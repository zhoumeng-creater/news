from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from Complete import PhysicsBasis, decomposition_solver, fit_m_band_least_squares, run_stage1_greedy


ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "data" / "raw" / "chatgpt_release_2022w48"
SCRIPT_PATH = ROOT / "scripts" / "run_experiment.py"


def _synthetic_series(n: int = 240, noise_std: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(float(n), dtype=float)
    m_true = np.array([1.8, 1.0, -0.6, 0.4, 0.2], dtype=float)
    l_true = np.array([8.0, 28.0, 0.7, 0.02, 0.3], dtype=float)
    h_true = np.array([5.5, 132.0, 0.5, 0.18, 1.1], dtype=float)

    rng = np.random.default_rng(2026)
    y = (
        PhysicsBasis.m_band_kernel(t, m_true)
        + PhysicsBasis.wave_kernel(t, l_true, "L")
        + PhysicsBasis.wave_kernel(t, h_true, "H")
        + rng.normal(0.0, noise_std, size=n)
    )
    return t, y


def _run_cli(case_dir: Path, output_root: Path, seed: int) -> Path:
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--case-dir",
        str(case_dir),
        "--output-root",
        str(output_root),
        "--max-waves",
        "2",
        "--seed",
        str(seed),
        "--de-popsize",
        "6",
        "--de-maxiter",
        "8",
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["MPLBACKEND"] = "Agg"
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)
    return output_root / case_dir.name


def test_m_band_least_squares_recovers_24h_12h_rhythm() -> None:
    t = np.arange(0.0, 336.0, 1.0)
    coeffs_true = np.array([2.0, 1.2, -0.8, 0.5, 0.3], dtype=float)
    y = PhysicsBasis.m_band_kernel(t, coeffs_true)
    coeffs_est, y_rhythm, residual = fit_m_band_least_squares(t, y)

    mse = float(np.mean((y_rhythm - y) ** 2))
    assert mse < 1e-12
    assert float(np.linalg.norm(residual)) < 1e-8
    np.testing.assert_allclose(coeffs_est, coeffs_true, rtol=1e-7, atol=1e-9)


def test_stage1_bic_acceptance_monotonic() -> None:
    t, y = _synthetic_series(n=220, noise_std=0.1)
    _, _, residual = fit_m_band_least_squares(t, y)
    amp_upper = max(15.0, float(np.max(np.abs(y)) * 1.5 + np.std(y)))

    _, _, bic_history, _ = run_stage1_greedy(
        t=t,
        residual=residual,
        max_waves=4,
        seed=12,
        amp_upper=amp_upper,
        de_popsize=6,
        de_maxiter=10,
        verbose=False,
    )

    accepted_steps = [item for item in bic_history if item["step"] > 0 and item["accepted"]]
    for item in accepted_steps:
        assert item["bic_after"] < item["bic_before"]

    rejected_steps = [item for item in bic_history if item["step"] > 0 and not item["accepted"]]
    if rejected_steps:
        assert len(rejected_steps) == 1
        assert rejected_steps[0]["step"] == bic_history[-1]["step"]


def test_stage2_lbfgsb_loss_not_worse_than_de() -> None:
    t, y = _synthetic_series(n=200, noise_std=0.15)
    result = decomposition_solver(
        t=t,
        y_obs=y,
        max_waves=3,
        seed=42,
        de_popsize=8,
        de_maxiter=10,
        verbose=False,
    )
    assert result["loss"]["lbfgsb"] <= result["loss"]["de"] + 1e-9


def test_integration_outputs_expected_files(tmp_path: Path) -> None:
    out_dir = _run_cli(CASE_DIR, tmp_path / "processed", seed=42)
    assert (out_dir / "fit.png").exists()
    assert (out_dir / "components.png").exists()
    assert (out_dir / "decomp_hourly.csv").exists()
    assert (out_dir / "run_meta.json").exists()
    assert (out_dir / "实验报告.md").exists()

    decomp_df = pd.read_csv(out_dir / "decomp_hourly.csv")
    expected_columns = {
        "ts_utc",
        "trends",
        "y_rhythm",
        "residual",
        "L_sum",
        "H_sum",
        "recon",
        "gdelt_vol",
        "twitter_vol",
    }
    assert expected_columns.issubset(set(decomp_df.columns))
    assert len(decomp_df) > 0


def test_reproducible_with_same_seed(tmp_path: Path) -> None:
    out_dir_1 = _run_cli(CASE_DIR, tmp_path / "run1", seed=77)
    out_dir_2 = _run_cli(CASE_DIR, tmp_path / "run2", seed=77)

    meta_1 = json.loads((out_dir_1 / "run_meta.json").read_text(encoding="utf-8"))
    meta_2 = json.loads((out_dir_2 / "run_meta.json").read_text(encoding="utf-8"))

    assert meta_1["k_waves"] == meta_2["k_waves"]
    assert abs(float(meta_1["loss"]["de"]) - float(meta_2["loss"]["de"])) < 1e-10
    assert abs(float(meta_1["loss"]["lbfgsb"]) - float(meta_2["loss"]["lbfgsb"])) < 1e-10

    df_1 = pd.read_csv(out_dir_1 / "decomp_hourly.csv")
    df_2 = pd.read_csv(out_dir_2 / "decomp_hourly.csv")
    cols = ["y_rhythm", "L_sum", "H_sum", "recon"]
    np.testing.assert_allclose(df_1[cols].to_numpy(), df_2[cols].to_numpy(), rtol=0.0, atol=1e-10)
