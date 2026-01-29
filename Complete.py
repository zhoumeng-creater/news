import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize


# ==========================================
# 1. 物理引擎 (Physics Engine)
# ==========================================
class PhysicsBasis:
    @staticmethod
    def wave_kernel(t, params, wave_type):
        """生成单个波形 (L 或 H)"""
        # params: [A, tau, lam, f, phi]
        A, tau, lam, f, phi = params
        dt = t - tau
        mask = np.where(dt >= 0, 1.0, 0.0)
        dt = np.maximum(dt, 0)

        if wave_type == 'L':  # 机构惯性 (Power-law)
            envelope = (1 + dt) ** (-lam)
        elif wave_type == 'H':  # 社交脉冲 (Exponential)
            envelope = np.exp(-lam * dt)

        oscillation = np.cos(2 * np.pi * f * dt + phi)
        return A * envelope * oscillation * mask

    @staticmethod
    def m_band_kernel(t, params):
        """生成 M-Band (背景节律)"""
        # params: [A, phi, offset]
        A, phi, offset = params
        f_day = 1.0 / 24.0  # 锁定 24h 周期
        return A * np.cos(2 * np.pi * f_day * t + phi) + offset

    @staticmethod
    def build_full_model(t, m_params, waves_list):
        """根据参数列表重建完整信号"""
        y_recon = PhysicsBasis.m_band_kernel(t, m_params)
        for wave in waves_list:
            y_recon += PhysicsBasis.wave_kernel(t, wave['params'], wave['type'])
        return y_recon


# ==========================================
# 2. 辅助算子 (Operators)
# ==========================================
def calculate_bic(residual, k, n):
    """BIC 准则: 判定是否停止搜索"""
    mse = np.mean(residual ** 2)
    if mse <= 1e-15: mse = 1e-15
    return n * np.log(mse) + k * np.log(n)


def phase_grid_search(t, residual, base_bounds, wave_type):
    """
    [特异性算子] 相位网格扫描
    在 DE 之前粗筛出最佳相位，解决非凸性问题。
    """
    best_phi = 0.0
    min_mse = np.inf

    # 构造一个测试用的"平均参数"来扫相位
    # A, tau, lam, f 取边界的中值
    test_params = [np.mean(b) for b in base_bounds[:4]]

    phases = np.linspace(0, 2 * np.pi, 12)  # 尝试 12 个相位角
    for phi in phases:
        p = test_params + [phi]
        wave = PhysicsBasis.wave_kernel(t, p, wave_type)
        mse = np.mean((residual - wave) ** 2)
        if mse < min_mse:
            min_mse = mse
            best_phi = phi
    return best_phi


# ==========================================
# 3. 核心算法: Stage I & Stage II
# ==========================================
def fit_single_wave(t, residual, wave_type):
    """Stage I 的子程序: 拟合单个波"""
    # 1. 定义物理边界
    if wave_type == 'L':  # 低频
        bounds = [(0, 15), (0, 100), (0.1, 2.0), (0.001, 0.05), (0, 2 * np.pi)]
    else:  # 高频
        bounds = [(0, 15), (0, 100), (0.1, 1.0), (0.05, 0.5), (0, 2 * np.pi)]

    a_min, a_max = bounds[0]

    def solve_amp_phase(tau, lam, f):
        dt = t - tau
        mask = np.where(dt >= 0, 1.0, 0.0)
        dt_pos = np.maximum(dt, 0)

        if wave_type == 'L':
            envelope = (1 + dt_pos) ** (-lam)
        else:
            envelope = np.exp(-lam * dt_pos)

        cos_term = np.cos(2 * np.pi * f * dt_pos)
        sin_term = np.sin(2 * np.pi * f * dt_pos)
        c = envelope * cos_term * mask
        s = envelope * sin_term * mask

        cc = np.dot(c, c)
        ss = np.dot(s, s)
        cs = np.dot(c, s)
        cr = np.dot(c, residual)
        sr = np.dot(s, residual)

        det = cc * ss - cs * cs
        if det <= 1e-12:
            B, C = 0.0, 0.0
        else:
            B = (cr * ss - sr * cs) / det
            C = (sr * cc - cr * cs) / det

        A = np.hypot(B, C)
        phi = np.mod(np.arctan2(-C, B), 2 * np.pi)
        A_clipped = np.clip(A, a_min, a_max)

        w = PhysicsBasis.wave_kernel(t, [A_clipped, tau, lam, f, phi], wave_type)
        mse = np.mean((residual - w) ** 2)
        return mse, A_clipped, phi

    # 2. 差分进化仅搜索 (tau, lam, f)，相位与振幅解析对齐
    def obj(theta):
        tau, lam, f = theta
        mse, _, _ = solve_amp_phase(tau, lam, f)
        return mse

    result = differential_evolution(obj, bounds[1:4], strategy='best1bin', popsize=20, tol=1e-3, seed=None)

    tau_opt, lam_opt, f_opt = result.x
    mse, A_opt, phi_opt = solve_amp_phase(tau_opt, lam_opt, f_opt)
    return np.array([A_opt, tau_opt, lam_opt, f_opt, phi_opt]), mse


def messd_solver(t, y_obs, max_waves=5):
    """M-ESSD 主求解器 (Stage I + Stage II)"""
    n = len(t)
    current_residual = y_obs.copy()
    identified_waves = []

    print("🚀 [Step 0] Pre-processing: Removing M-Band...")

    # 0. 剥离 M-Band
    def m_obj(theta):
        return np.mean((y_obs - PhysicsBasis.m_band_kernel(t, theta)) ** 2)

    m_res = differential_evolution(m_obj, [(0, 5), (0, 2 * np.pi), (-2, 2)], tol=1e-3)
    m_params_init = m_res.x

    current_residual -= PhysicsBasis.m_band_kernel(t, m_params_init)
    current_bic = calculate_bic(current_residual, 3, n)
    print(f"   -> Init BIC: {current_bic:.2f}")

    # ================= Stage I: Incremental Greedy Search =================
    print("\n🚀 [Stage I] Incremental Search with BIC...")
    for k in range(max_waves):
        # 竞争: 试探 L 和 H
        p_L, err_L = fit_single_wave(t, current_residual, 'L')
        p_H, err_H = fit_single_wave(t, current_residual, 'H')

        # 择优
        if err_L < err_H:
            candidate = {'type': 'L', 'params': p_L}
            cand_wave = PhysicsBasis.wave_kernel(t, p_L, 'L')
        else:
            candidate = {'type': 'H', 'params': p_H}
            cand_wave = PhysicsBasis.wave_kernel(t, p_H, 'H')

        # BIC 检查
        temp_residual = current_residual - cand_wave
        # 参数个数: M(3) + 已有波(k*5) + 新波(5)
        k_params = 3 + (len(identified_waves) + 1) * 5
        new_bic = calculate_bic(temp_residual, k_params, n)

        print(f"   -> Wave {k + 1} candidate: {candidate['type']}-Band. BIC: {current_bic:.2f} -> {new_bic:.2f}")

        if new_bic < current_bic:
            identified_waves.append(candidate)
            current_residual = temp_residual
            current_bic = new_bic
            print("      ✅ Accepted.")
        else:
            print("      🛑 Rejected (BIC increased). Stopping Stage I.")
            break

    # ================= Stage II: Global Memetic Refinement =================
    print("\n🚀 [Stage II] Global Memetic Refinement (L-BFGS-B)...")

    # 1. 拼接所有参数: [M_params, Wave1_params, Wave2_params...]
    # 这是一个变长的参数向量
    x0 = list(m_params_init)
    for w in identified_waves:
        x0.extend(w['params'])
    x0 = np.array(x0)

    # 2. 定义全局目标函数
    def global_obj(flat_params):
        # 解包参数
        m_p = flat_params[:3]
        wave_p_list = flat_params[3:]

        y_recon = PhysicsBasis.m_band_kernel(t, m_p)

        # 每 5 个一组解包波形
        num_waves = len(identified_waves)
        for i in range(num_waves):
            p = wave_p_list[i * 5: (i + 1) * 5]
            w_type = identified_waves[i]['type']
            y_recon += PhysicsBasis.wave_kernel(t, p, w_type)

        return np.mean((y_obs - y_recon) ** 2)

    # 3. L-BFGS-B 优化
    # 需要动态生成边界
    bounds = [(0, 5), (0, 2 * np.pi), (-2, 2)]  # M-Band bounds
    for w in identified_waves:
        if w['type'] == 'L':
            bounds.extend([(0, 15), (0, 100), (0.1, 2.0), (0.001, 0.05), (0, 2 * np.pi)])
        else:
            bounds.extend([(0, 15), (0, 100), (0.1, 1.0), (0.05, 0.5), (0, 2 * np.pi)])

    result_refine = minimize(global_obj, x0, method='L-BFGS-B', bounds=bounds)

    print(f"   -> Final MSE: {result_refine.fun:.6f}")

    # 4. 解析最终参数
    final_m = result_refine.x[:3]
    final_waves = []
    wave_params_flat = result_refine.x[3:]
    for i in range(len(identified_waves)):
        p = wave_params_flat[i * 5: (i + 1) * 5]
        final_waves.append({'type': identified_waves[i]['type'], 'params': p})

    return final_m, final_waves


# ==========================================
# 4. 实验验证与绘图
# ==========================================
if __name__ == "__main__":
    # --- A. 生成合成数据 ---
    t = np.linspace(0, 100, 500)
    gt_L = [8.0, 10.0, 0.6, 0.02, 0.0]  # t=10
    gt_H = [6.0, 50.0, 0.4, 0.2, np.pi / 2]  # t=50
    gt_M = [1.5, 0.0, 0.5]

    # 构造带噪信号
    y_true_L = PhysicsBasis.wave_kernel(t, gt_L, 'L')
    y_true_H = PhysicsBasis.wave_kernel(t, gt_H, 'H')
    y_true_M = PhysicsBasis.m_band_kernel(t, gt_M)
    y_obs = y_true_L + y_true_H + y_true_M + np.random.normal(0, 0.3, len(t))

    # --- B. 运行 M-ESSD ---
    est_m, est_waves = messd_solver(t, y_obs)

    # --- C. 重建结果 ---
    y_est_M = PhysicsBasis.m_band_kernel(t, est_m)
    y_est_L = np.zeros_like(t)
    y_est_H = np.zeros_like(t)

    for w in est_waves:
        wave_sig = PhysicsBasis.wave_kernel(t, w['params'], w['type'])
        if w['type'] == 'L':
            y_est_L += wave_sig
        else:
            y_est_H += wave_sig

    # --- D. 归因计算 ---
    vol_L = np.sum(np.abs(y_est_L))
    vol_H = np.sum(np.abs(y_est_H))
    vol_M = np.sum(np.abs(y_est_M))
    total = vol_L + vol_H + vol_M
    pct_L, pct_H, pct_M = vol_L / total * 100, vol_H / total * 100, vol_M / total * 100

    # --- E. 终极绘图 ---
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.5, 1])

    # 1. Total
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, y_obs, 'k.', alpha=0.2, label='Observed')
    ax1.plot(t, y_est_L + y_est_H + y_est_M, 'r-', lw=2, label='M-ESSD')
    ax1.set_title('A. Total Reconstruction')
    ax1.legend()
    ax1.set_xticks([])

    # 2. L-Band
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t, y_true_L, 'b--', alpha=0.5)
    ax2.fill_between(t, y_est_L, color='blue', alpha=0.3, label=f'L-Band: {pct_L:.1f}%')
    ax2.set_title('B. Institutional Inertia')
    ax2.legend()
    ax2.set_xticks([])

    # 3. H-Band
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(t, y_true_H, 'g--', alpha=0.5)
    ax3.fill_between(t, y_est_H, color='green', alpha=0.3, label=f'H-Band: {pct_H:.1f}%')
    ax3.set_title('C. Social Impulse')
    ax3.legend()
    ax3.set_xticks([])

    # 4. M-Band
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.plot(t, y_true_M, 'orange', ls='--')
    ax4.plot(t, y_est_M, color='orange', lw=2, label=f'M-Band: {pct_M:.1f}%')
    ax4.set_title('D. Circadian Rhythm')
    ax4.legend()

    # 5. Pie Chart
    ax_pie = fig.add_subplot(gs[:, 1])
    labels = [f'Institutional\n{pct_L:.1f}%', f'Social\n{pct_H:.1f}%', f'Rhythm\n{pct_M:.1f}%']
    ax_pie.pie([vol_L, vol_H, vol_M], labels=labels, colors=['#6495ED', '#90EE90', '#FFDEAD'],
               explode=(0.05, 0.05, 0), autopct='%1.1f%%', shadow=True)
    ax_pie.set_title('E. Causal Attribution')

    plt.tight_layout()
    plt.show()
