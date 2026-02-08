from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Complete import decomposition_solver
from scripts.generate_experiment_report import generate_report


DEFAULT_CASE_DIR = Path("data/raw/chatgpt_release_2022w48")
DEFAULT_OUTPUT_ROOT = Path("data/processed")
DEFAULT_INPUT_NAME = "panel_hourly.csv"
REQUIRED_COLUMNS = ["ts_utc", "trends", "gdelt_raw", "twitter_count"]


def _load_panel(panel_path: Path) -> pd.DataFrame:
    if not panel_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {panel_path}")
    df = pd.read_csv(panel_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"panel_hourly.csv 缺少字段: {missing}")
    if df.empty:
        raise ValueError("panel_hourly.csv 为空")
    return df[REQUIRED_COLUMNS].copy()


def _output_dir(case_dir: Path, output_root: Path) -> Path:
    return output_root / case_dir.name


def _save_fit_plot(ts: pd.Series, trends: np.ndarray, recon: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5), dpi=150)
    ax.plot(ts, trends, color="#1f77b4", linewidth=1.5, label="trends")
    ax.plot(ts, recon, color="#d62728", linewidth=1.7, label="recon")
    ax.set_title("Fit: trends vs recon")
    ax.set_xlabel("ts_utc")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_components_plot(
    ts: pd.Series,
    y_rhythm: np.ndarray,
    l_sum: np.ndarray,
    h_sum: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 8.5), dpi=150, sharex=True)

    axes[0].plot(ts, y_rhythm, color="#ff7f0e", linewidth=1.6, label="M (rhythm)")
    axes[0].set_ylabel("M")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper left")

    axes[1].plot(ts, l_sum, color="#1f77b4", linewidth=1.6, label="L_sum")
    axes[1].set_ylabel("L")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper left")

    axes[2].plot(ts, h_sum, color="#2ca02c", linewidth=1.6, label="H_sum")
    axes[2].set_ylabel("H")
    axes[2].set_xlabel("ts_utc")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="upper left")

    fig.suptitle("Decomposed Components")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_decomp_csv(
    ts_utc: pd.Series,
    trends: np.ndarray,
    y_rhythm: np.ndarray,
    residual: np.ndarray,
    l_sum: np.ndarray,
    h_sum: np.ndarray,
    recon: np.ndarray,
    gdelt_vol: np.ndarray,
    twitter_vol: np.ndarray,
    out_path: Path,
) -> None:
    out_df = pd.DataFrame(
        {
            "ts_utc": ts_utc,
            "trends": trends,
            "y_rhythm": y_rhythm,
            "residual": residual,
            "L_sum": l_sum,
            "H_sum": h_sum,
            "recon": recon,
            "gdelt_vol": gdelt_vol,
            "twitter_vol": twitter_vol,
        }
    )
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")


def run_experiment(
    case_dir: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    input_name: str = DEFAULT_INPUT_NAME,
    max_waves: int = 6,
    seed: int = 42,
    de_popsize: int = 12,
    de_maxiter: int = 60,
) -> dict[str, Any]:
    panel_path = case_dir / input_name
    panel_df = _load_panel(panel_path)

    ts_utc = panel_df["ts_utc"].astype(str)
    ts_for_plot = pd.to_datetime(ts_utc, utc=True)
    trends = pd.to_numeric(panel_df["trends"], errors="coerce").astype(float).to_numpy()
    gdelt_vol = pd.to_numeric(panel_df["gdelt_raw"], errors="coerce").fillna(0.0).astype(float).to_numpy()
    twitter_vol = (
        pd.to_numeric(panel_df["twitter_count"], errors="coerce").fillna(0.0).astype(float).to_numpy()
    )

    t = np.arange(len(panel_df), dtype=float)
    result = decomposition_solver(
        t=t,
        y_obs=trends,
        max_waves=max_waves,
        seed=seed,
        de_popsize=de_popsize,
        de_maxiter=de_maxiter,
        verbose=True,
    )

    y_rhythm = np.asarray(result["y_rhythm"], dtype=float)
    residual = trends - y_rhythm
    l_sum = np.asarray(result["L_sum"], dtype=float)
    h_sum = np.asarray(result["H_sum"], dtype=float)
    recon = np.asarray(result["recon"], dtype=float)

    out_dir = _output_dir(case_dir, output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    fit_path = out_dir / "fit.png"
    components_path = out_dir / "components.png"
    decomp_path = out_dir / "decomp_hourly.csv"
    meta_path = out_dir / "run_meta.json"
    report_path = out_dir / "实验报告.md"

    legacy_paths = [
        out_dir / "L_vs_gdelt.png",
        out_dir / "H_vs_twitter.png",
    ]
    for legacy_path in legacy_paths:
        if legacy_path.exists():
            legacy_path.unlink()

    _save_fit_plot(ts_for_plot, trends, recon, fit_path)
    _save_components_plot(ts_for_plot, y_rhythm, l_sum, h_sum, components_path)

    _save_decomp_csv(
        ts_utc=ts_utc,
        trends=trends,
        y_rhythm=y_rhythm,
        residual=residual,
        l_sum=l_sum,
        h_sum=h_sum,
        recon=recon,
        gdelt_vol=gdelt_vol,
        twitter_vol=twitter_vol,
        out_path=decomp_path,
    )

    meta = {
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": case_dir.name,
        "input_file": str(panel_path),
        "output_dir": str(out_dir),
        "seed": int(seed),
        "max_waves": int(max_waves),
        "de_popsize": int(de_popsize),
        "de_maxiter": int(de_maxiter),
        "n_obs": int(len(panel_df)),
        "k_waves": int(len(result["waves"])),
        "wave_types": [wave["type"] for wave in result["waves"]],
        "waves": [
            {"type": wave["type"], "params": [float(v) for v in wave["params"]]}
            for wave in result["waves"]
        ],
        "m_params": [float(v) for v in result["m_params"]],
        "loss": result["loss"],
        "stage1": result["stage1"],
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write("\n")

    generate_report(output_dir=out_dir)

    return {
        "fit_path": fit_path,
        "components_path": components_path,
        "decomp_path": decomp_path,
        "meta_path": meta_path,
        "report_path": report_path,
        "output_dir": out_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="运行小时级分解实验并导出图表与分解结果")
    parser.add_argument(
        "--case-dir",
        type=str,
        default=str(DEFAULT_CASE_DIR),
        help="输入数据目录，需包含 panel_hourly.csv",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="输出根目录，结果将写入 data/processed/<case>/",
    )
    parser.add_argument(
        "--input-name",
        type=str,
        default=DEFAULT_INPUT_NAME,
        help="输入面板文件名",
    )
    parser.add_argument(
        "--max-waves",
        type=int,
        default=6,
        help="Stage I 最大增量波数量",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--de-popsize",
        type=int,
        default=12,
        help="DE 种群规模因子",
    )
    parser.add_argument(
        "--de-maxiter",
        type=int,
        default=60,
        help="DE 最大迭代轮次",
    )
    args = parser.parse_args()

    outputs = run_experiment(
        case_dir=Path(args.case_dir),
        output_root=Path(args.output_root),
        input_name=args.input_name,
        max_waves=args.max_waves,
        seed=args.seed,
        de_popsize=args.de_popsize,
        de_maxiter=args.de_maxiter,
    )

    print(f"[OK] fit.png: {outputs['fit_path']}")
    print(f"[OK] components.png: {outputs['components_path']}")
    print(f"[OK] decomp_hourly.csv: {outputs['decomp_path']}")
    print(f"[OK] run_meta.json: {outputs['meta_path']}")
    print(f"[OK] 实验报告.md: {outputs['report_path']}")


if __name__ == "__main__":
    main()
