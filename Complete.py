from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import differential_evolution, minimize


TWO_PI = 2.0 * np.pi
M_PARAM_COUNT = 5
WAVE_PARAM_COUNT = 5


class PhysicsBasis:
    @staticmethod
    def wave_kernel(t: np.ndarray, params: np.ndarray, wave_type: str) -> np.ndarray:
        """生成单个波形 (L 或 H)。"""
        amp, tau, lam, freq, phase = params
        dt = t - tau
        mask = np.where(dt >= 0.0, 1.0, 0.0)
        dt = np.maximum(dt, 0.0)

        if wave_type == "L":
            envelope = (1.0 + dt) ** (-lam)
        elif wave_type == "H":
            envelope = np.exp(-lam * dt)
        else:
            raise ValueError(f"不支持的波类型: {wave_type}")

        oscillation = np.cos(TWO_PI * freq * dt + phase)
        return amp * envelope * oscillation * mask

    @staticmethod
    def m_band_kernel(t: np.ndarray, params: np.ndarray) -> np.ndarray:
        """生成 24h+12h 的 M-Band 背景节律。"""
        b0, b1, b2, b3, b4 = params
        omega_24 = TWO_PI * t / 24.0
        omega_12 = TWO_PI * t / 12.0
        return (
            b0
            + b1 * np.cos(omega_24)
            + b2 * np.sin(omega_24)
            + b3 * np.cos(omega_12)
            + b4 * np.sin(omega_12)
        )


def build_m_design_matrix(t: np.ndarray) -> np.ndarray:
    """构造 M-Band 的线性最小二乘设计矩阵。"""
    omega_24 = TWO_PI * t / 24.0
    omega_12 = TWO_PI * t / 12.0
    return np.column_stack(
        (
            np.ones_like(t),
            np.cos(omega_24),
            np.sin(omega_24),
            np.cos(omega_12),
            np.sin(omega_12),
        )
    )


def fit_m_band_least_squares(t: np.ndarray, y_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """通过线性最小二乘拟合 M-Band，并返回 y_rhythm 与 residual。"""
    x = build_m_design_matrix(t)
    coeffs, _, _, _ = np.linalg.lstsq(x, y_obs, rcond=None)
    y_rhythm = x @ coeffs
    residual = y_obs - y_rhythm
    return coeffs, y_rhythm, residual


def calculate_bic(residual: np.ndarray, num_params: int, n_obs: int) -> float:
    """BIC 准则。"""
    mse = np.mean(residual**2)
    mse = max(mse, 1e-15)
    return float(n_obs * np.log(mse) + num_params * np.log(n_obs))


def _wave_bounds(wave_type: str, t_max: float, amp_upper: float) -> list[tuple[float, float]]:
    if wave_type == "L":
        return [
            (0.0, amp_upper),
            (0.0, t_max),
            (0.1, 2.0),
            (0.001, 0.05),
            (0.0, TWO_PI),
        ]
    if wave_type == "H":
        return [
            (0.0, amp_upper),
            (0.0, t_max),
            (0.1, 1.0),
            (0.05, 0.5),
            (0.0, TWO_PI),
        ]
    raise ValueError(f"不支持的波类型: {wave_type}")


def _m_bounds(y_scale: float) -> list[tuple[float, float]]:
    limit = max(1.0, 2.0 * y_scale)
    return [(-limit, limit)] * M_PARAM_COUNT


def fit_single_wave(
    t: np.ndarray,
    residual: np.ndarray,
    wave_type: str,
    amp_upper: float,
    seed: int,
    de_popsize: int,
    de_maxiter: int,
) -> tuple[np.ndarray, float]:
    """在残差上拟合单个最优波形。"""
    bounds = _wave_bounds(wave_type, float(np.max(t)), amp_upper)
    amp_min, amp_max = bounds[0]

    def solve_amp_phase(tau: float, lam: float, freq: float) -> tuple[float, float, float]:
        dt = t - tau
        mask = np.where(dt >= 0.0, 1.0, 0.0)
        dt_pos = np.maximum(dt, 0.0)

        if wave_type == "L":
            envelope = (1.0 + dt_pos) ** (-lam)
        else:
            envelope = np.exp(-lam * dt_pos)

        cos_term = np.cos(TWO_PI * freq * dt_pos)
        sin_term = np.sin(TWO_PI * freq * dt_pos)
        c_vec = envelope * cos_term * mask
        s_vec = envelope * sin_term * mask

        cc = float(np.dot(c_vec, c_vec))
        ss = float(np.dot(s_vec, s_vec))
        cs = float(np.dot(c_vec, s_vec))
        cr = float(np.dot(c_vec, residual))
        sr = float(np.dot(s_vec, residual))

        det = cc * ss - cs * cs
        if det <= 1e-12:
            b_val, c_val = 0.0, 0.0
        else:
            b_val = (cr * ss - sr * cs) / det
            c_val = (sr * cc - cr * cs) / det

        amp = float(np.hypot(b_val, c_val))
        phase = float(np.mod(np.arctan2(-c_val, b_val), TWO_PI))
        amp = float(np.clip(amp, amp_min, amp_max))

        wave = PhysicsBasis.wave_kernel(t, np.array([amp, tau, lam, freq, phase]), wave_type)
        mse = float(np.mean((residual - wave) ** 2))
        return mse, amp, phase

    def obj(theta: np.ndarray) -> float:
        tau, lam, freq = theta
        mse, _, _ = solve_amp_phase(float(tau), float(lam), float(freq))
        return mse

    de_result = differential_evolution(
        obj,
        bounds=bounds[1:4],
        strategy="best1bin",
        popsize=de_popsize,
        maxiter=de_maxiter,
        tol=1e-4,
        seed=seed,
        polish=False,
        updating="deferred",
    )

    tau_opt, lam_opt, freq_opt = [float(v) for v in de_result.x]
    mse, amp_opt, phase_opt = solve_amp_phase(tau_opt, lam_opt, freq_opt)
    params = np.array([amp_opt, tau_opt, lam_opt, freq_opt, phase_opt], dtype=float)
    return params, mse


def _flatten_params(m_params: np.ndarray, waves: list[dict[str, Any]]) -> np.ndarray:
    flat = list(m_params)
    for wave in waves:
        flat.extend(wave["params"])
    return np.array(flat, dtype=float)


def _unpack_params(flat_params: np.ndarray, wave_types: list[str]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    m_params = np.array(flat_params[:M_PARAM_COUNT], dtype=float)
    waves: list[dict[str, Any]] = []
    wave_flat = flat_params[M_PARAM_COUNT:]
    for idx, wave_type in enumerate(wave_types):
        start = idx * WAVE_PARAM_COUNT
        end = (idx + 1) * WAVE_PARAM_COUNT
        params = np.array(wave_flat[start:end], dtype=float)
        waves.append({"type": wave_type, "params": params})
    return m_params, waves


def _global_bounds(wave_types: list[str], t_max: float, amp_upper: float, y_scale: float) -> list[tuple[float, float]]:
    bounds = _m_bounds(y_scale)
    for wave_type in wave_types:
        bounds.extend(_wave_bounds(wave_type, t_max, amp_upper))
    return bounds


def reconstruct_components(
    t: np.ndarray,
    m_params: np.ndarray,
    waves: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """根据参数重构 y_rhythm、L_sum、H_sum 与 recon。"""
    y_rhythm = PhysicsBasis.m_band_kernel(t, m_params)
    l_sum = np.zeros_like(t, dtype=float)
    h_sum = np.zeros_like(t, dtype=float)

    for wave in waves:
        signal = PhysicsBasis.wave_kernel(t, wave["params"], wave["type"])
        if wave["type"] == "L":
            l_sum += signal
        else:
            h_sum += signal

    recon = y_rhythm + l_sum + h_sum
    return y_rhythm, l_sum, h_sum, recon


def _global_mse_objective(
    flat_params: np.ndarray,
    t: np.ndarray,
    y_obs: np.ndarray,
    wave_types: list[str],
) -> float:
    m_params, waves = _unpack_params(flat_params, wave_types)
    _, _, _, recon = reconstruct_components(t, m_params, waves)
    return float(np.mean((y_obs - recon) ** 2))


def run_stage1_greedy(
    t: np.ndarray,
    residual: np.ndarray,
    max_waves: int,
    seed: int,
    amp_upper: float,
    de_popsize: int,
    de_maxiter: int,
    verbose: bool,
) -> tuple[list[dict[str, Any]], np.ndarray, list[dict[str, Any]], float]:
    """Stage I：残差上贪婪增量搜索，BIC 判停。"""
    n_obs = len(t)
    current_residual = residual.copy()
    identified_waves: list[dict[str, Any]] = []
    bic_history: list[dict[str, Any]] = []

    current_bic = calculate_bic(current_residual, M_PARAM_COUNT, n_obs)
    bic_history.append(
        {
            "step": 0,
            "candidate_type": "init",
            "bic_before": current_bic,
            "bic_after": current_bic,
            "accepted": True,
        }
    )

    if verbose:
        print(f"[Stage I] Init BIC: {current_bic:.6f}")

    for step in range(1, max_waves + 1):
        l_seed = seed + step * 1000 + 11
        h_seed = seed + step * 1000 + 29

        p_l, err_l = fit_single_wave(
            t=t,
            residual=current_residual,
            wave_type="L",
            amp_upper=amp_upper,
            seed=l_seed,
            de_popsize=de_popsize,
            de_maxiter=de_maxiter,
        )
        p_h, err_h = fit_single_wave(
            t=t,
            residual=current_residual,
            wave_type="H",
            amp_upper=amp_upper,
            seed=h_seed,
            de_popsize=de_popsize,
            de_maxiter=de_maxiter,
        )

        if err_l <= err_h:
            candidate = {"type": "L", "params": p_l}
            candidate_wave = PhysicsBasis.wave_kernel(t, p_l, "L")
        else:
            candidate = {"type": "H", "params": p_h}
            candidate_wave = PhysicsBasis.wave_kernel(t, p_h, "H")

        temp_residual = current_residual - candidate_wave
        k_params = M_PARAM_COUNT + (len(identified_waves) + 1) * WAVE_PARAM_COUNT
        new_bic = calculate_bic(temp_residual, k_params, n_obs)
        accepted = new_bic < current_bic

        bic_item = {
            "step": step,
            "candidate_type": candidate["type"],
            "bic_before": current_bic,
            "bic_after": new_bic,
            "accepted": accepted,
            "mse_l": float(err_l),
            "mse_h": float(err_h),
        }
        bic_history.append(bic_item)

        if verbose:
            print(
                f"[Stage I] step={step} candidate={candidate['type']} "
                f"BIC {current_bic:.6f} -> {new_bic:.6f} accepted={accepted}"
            )

        if accepted:
            identified_waves.append(candidate)
            current_residual = temp_residual
            current_bic = new_bic
        else:
            break

    return identified_waves, current_residual, bic_history, current_bic


def run_stage2_memetic(
    t: np.ndarray,
    y_obs: np.ndarray,
    m_init: np.ndarray,
    waves_init: list[dict[str, Any]],
    seed: int,
    amp_upper: float,
    y_scale: float,
    de_popsize: int,
    de_maxiter: int,
    verbose: bool,
) -> tuple[np.ndarray, list[dict[str, Any]], Any, Any]:
    """Stage II：全局 DE + 局部 L-BFGS-B 的模因精修。"""
    wave_types = [wave["type"] for wave in waves_init]
    x0 = _flatten_params(m_init, waves_init)
    bounds = _global_bounds(wave_types, float(np.max(t)), amp_upper, y_scale)

    def obj(flat_params: np.ndarray) -> float:
        return _global_mse_objective(flat_params, t, y_obs, wave_types)

    de_result = differential_evolution(
        obj,
        bounds=bounds,
        strategy="best1bin",
        popsize=de_popsize,
        maxiter=de_maxiter,
        tol=1e-4,
        seed=seed + 70000,
        polish=False,
        updating="deferred",
    )

    lbfgsb_result = minimize(
        obj,
        x0=de_result.x,
        method="L-BFGS-B",
        bounds=bounds,
    )

    final_m, final_waves = _unpack_params(lbfgsb_result.x, wave_types)

    if verbose:
        print(f"[Stage II] DE loss: {float(de_result.fun):.6f}")
        print(f"[Stage II] L-BFGS-B loss: {float(lbfgsb_result.fun):.6f}")

    return final_m, final_waves, de_result, lbfgsb_result


def decomposition_solver(
    t: np.ndarray,
    y_obs: np.ndarray,
    max_waves: int = 6,
    seed: int = 42,
    de_popsize: int = 12,
    de_maxiter: int = 60,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    双阶段分解求解器：
    1) 24h/12h 最小二乘剥离 M-Band
    2) Stage I 贪婪增量搜索 + BIC 判停
    3) Stage II 全局 DE + L-BFGS-B 联合优化
    """
    t = np.asarray(t, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)

    if t.ndim != 1 or y_obs.ndim != 1:
        raise ValueError("t 与 y_obs 必须为一维数组")
    if len(t) != len(y_obs):
        raise ValueError("t 与 y_obs 长度不一致")
    if len(t) < 20:
        raise ValueError("样本长度过短，至少需要 20 个点")

    amp_upper = max(15.0, float(np.max(np.abs(y_obs)) * 1.5 + np.std(y_obs)))
    y_scale = max(1.0, float(np.max(np.abs(y_obs))))

    if verbose:
        print("[Step 0] Pre-processing: Least-squares M-Band (24h/12h)")

    m_ls, y_rhythm_ls, residual_ls = fit_m_band_least_squares(t, y_obs)

    stage1_waves, stage1_residual, bic_history, final_bic = run_stage1_greedy(
        t=t,
        residual=residual_ls,
        max_waves=max_waves,
        seed=seed,
        amp_upper=amp_upper,
        de_popsize=de_popsize,
        de_maxiter=de_maxiter,
        verbose=verbose,
    )

    if verbose:
        print("[Stage II] Global memetic refinement (DE + L-BFGS-B)")

    final_m, final_waves, de_result, lbfgsb_result = run_stage2_memetic(
        t=t,
        y_obs=y_obs,
        m_init=m_ls,
        waves_init=stage1_waves,
        seed=seed,
        amp_upper=amp_upper,
        y_scale=y_scale,
        de_popsize=de_popsize,
        de_maxiter=de_maxiter,
        verbose=verbose,
    )

    y_rhythm, l_sum, h_sum, recon = reconstruct_components(t, final_m, final_waves)

    return {
        "m_params": final_m,
        "waves": final_waves,
        "y_rhythm": y_rhythm,
        "residual": y_obs - y_rhythm,
        "L_sum": l_sum,
        "H_sum": h_sum,
        "recon": recon,
        "stage1": {
            "bic_history": bic_history,
            "final_bic": float(final_bic),
            "residual_norm": float(np.linalg.norm(stage1_residual)),
        },
        "loss": {
            "de": float(de_result.fun),
            "lbfgsb": float(lbfgsb_result.fun),
        },
        "settings": {
            "max_waves": int(max_waves),
            "seed": int(seed),
            "de_popsize": int(de_popsize),
            "de_maxiter": int(de_maxiter),
        },
        "y_rhythm_ls": y_rhythm_ls,
        "residual_ls": residual_ls,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    t_demo = np.arange(0.0, 240.0, 1.0)
    m_demo = np.array([1.5, 0.9, -0.4, 0.3, 0.2])
    l_demo = np.array([9.0, 24.0, 0.7, 0.02, 0.1])
    h_demo = np.array([5.5, 120.0, 0.5, 0.2, 1.2])
    y_demo = (
        PhysicsBasis.m_band_kernel(t_demo, m_demo)
        + PhysicsBasis.wave_kernel(t_demo, l_demo, "L")
        + PhysicsBasis.wave_kernel(t_demo, h_demo, "H")
        + rng.normal(0.0, 0.25, size=len(t_demo))
    )

    result_demo = decomposition_solver(t_demo, y_demo, max_waves=4, seed=42, verbose=True)
    print(f"[Done] waves={len(result_demo['waves'])} final_loss={result_demo['loss']['lbfgsb']:.6f}")
