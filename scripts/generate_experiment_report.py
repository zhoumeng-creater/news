from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_DIR = Path("data/processed/chatgpt_release_2022w48")


def _safe_float(value: float) -> float:
    return float(np.round(float(value), 6))


def _series_profile(values: pd.Series) -> dict[str, float | int]:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    return {
        "nonzero": int((numeric != 0.0).sum()),
        "min": _safe_float(numeric.min()),
        "max": _safe_float(numeric.max()),
        "mean": _safe_float(numeric.mean()),
    }


def _compute_metrics(df: pd.DataFrame) -> dict[str, float]:
    trends = df["trends"].to_numpy(dtype=float)
    recon = df["recon"].to_numpy(dtype=float)
    residual = trends - recon

    mse = float(np.mean(residual**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residual)))
    ss_res = float(np.sum((trends - recon) ** 2))
    ss_tot = float(np.sum((trends - np.mean(trends)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    l_sum = df["L_sum"].to_numpy(dtype=float)
    h_sum = df["H_sum"].to_numpy(dtype=float)
    m_sum = df["y_rhythm"].to_numpy(dtype=float)
    vol_l = float(np.sum(np.abs(l_sum)))
    vol_h = float(np.sum(np.abs(h_sum)))
    vol_m = float(np.sum(np.abs(m_sum)))
    vol_total = vol_l + vol_h + vol_m

    if vol_total <= 1e-15:
        share_l, share_h, share_m = 0.0, 0.0, 0.0
    else:
        share_l = vol_l / vol_total * 100.0
        share_h = vol_h / vol_total * 100.0
        share_m = vol_m / vol_total * 100.0

    return {
        "mse": _safe_float(mse),
        "rmse": _safe_float(rmse),
        "mae": _safe_float(mae),
        "r2": _safe_float(r2),
        "share_l_pct": _safe_float(share_l),
        "share_h_pct": _safe_float(share_h),
        "share_m_pct": _safe_float(share_m),
        "h_abs_max": _safe_float(np.max(np.abs(h_sum))),
    }


def _build_report_text(output_dir: Path, meta: dict, df: pd.DataFrame) -> str:
    metrics = _compute_metrics(df)

    profiles = {
        "trends": _series_profile(df["trends"]),
        "gdelt_vol": _series_profile(df["gdelt_vol"]),
        "twitter_vol": _series_profile(df["twitter_vol"]),
    }

    bic_history = meta["stage1"]["bic_history"]
    accepted_steps = [item for item in bic_history if item.get("step", 0) > 0 and item.get("accepted")]
    rejected_steps = [item for item in bic_history if item.get("step", 0) > 0 and (not item.get("accepted"))]

    lines: list[str] = []
    lines.append("# 小时级分解实验报告")
    lines.append("")
    lines.append("## 1. 报告信息")
    lines.append(f"- 报告生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}")
    lines.append("- 执行者：Codex")
    lines.append(f"- 案例 ID：`{meta['case_id']}`")
    lines.append(f"- 输入文件：`{meta['input_file']}`")
    lines.append(f"- 输出目录：`{meta['output_dir']}`")
    lines.append("")

    lines.append("## 2. 数据概况")
    lines.append(f"- 样本点数：{meta['n_obs']}（小时级）")
    lines.append(f"- 时间范围：`{df['ts_utc'].iloc[0]}` 到 `{df['ts_utc'].iloc[-1]}`")
    lines.append("- 参与建模的目标序列：`trends`")
    lines.append("- 其他字段（`gdelt_vol`、`twitter_vol`）作为普通并行数据保留在结果表中，不参与目标函数加权。")
    lines.append("")
    lines.append("| 字段 | 非零小时数 | 最小值 | 最大值 | 均值 |")
    lines.append("|---|---:|---:|---:|---:|")
    for name in ["trends", "gdelt_vol", "twitter_vol"]:
        item = profiles[name]
        lines.append(
            f"| `{name}` | {item['nonzero']} | {item['min']:.6f} | {item['max']:.6f} | {item['mean']:.6f} |"
        )
    lines.append("")

    lines.append("## 3. 方法与参数")
    lines.append("1. 24h/12h M-Band 最小二乘预处理，得到 `y_rhythm` 与残差。")
    lines.append("2. Stage I 贪婪增量搜索：每轮拟合 L/H 单波，使用 BIC 判停。")
    lines.append("3. Stage II 全局优化：先 DE，再 L-BFGS-B 联合优化全参数。")
    lines.append(f"- `seed={meta['seed']}`")
    lines.append(f"- `max_waves={meta['max_waves']}`")
    lines.append(f"- `de_popsize={meta['de_popsize']}`")
    lines.append(f"- `de_maxiter={meta['de_maxiter']}`")
    lines.append("")

    lines.append("## 4. 核心结果")
    lines.append(f"- 最终波数量：{meta['k_waves']}")
    lines.append(f"- 波型序列：`{meta['wave_types']}`")
    lines.append(f"- DE 损失：{meta['loss']['de']:.6f}")
    lines.append(f"- L-BFGS-B 损失：{meta['loss']['lbfgsb']:.6f}")
    lines.append("")
    lines.append("### 4.1 拟合指标")
    lines.append(f"- MSE：{metrics['mse']:.6f}")
    lines.append(f"- RMSE：{metrics['rmse']:.6f}")
    lines.append(f"- MAE：{metrics['mae']:.6f}")
    lines.append(f"- R²：{metrics['r2']:.6f}")
    lines.append("")
    lines.append("### 4.2 分量体量占比（按绝对值积分）")
    lines.append(f"- L 分量占比：{metrics['share_l_pct']:.6f}%")
    lines.append(f"- H 分量占比：{metrics['share_h_pct']:.6f}%")
    lines.append(f"- M 分量占比：{metrics['share_m_pct']:.6f}%")
    lines.append(f"- H 分量绝对最大值：{metrics['h_abs_max']:.6f}")
    lines.append("")

    lines.append("### 4.3 BIC 迭代摘要")
    lines.append(f"- 初始 BIC：{bic_history[0]['bic_after']:.6f}")
    lines.append(f"- Stage I 最终 BIC：{meta['stage1']['final_bic']:.6f}")
    lines.append(f"- 接受轮数：{len(accepted_steps)}")
    lines.append(f"- 拒绝轮数：{len(rejected_steps)}")
    if rejected_steps:
        item = rejected_steps[-1]
        lines.append(
            f"- 最后一次拒绝：step={item['step']}, "
            f"type={item['candidate_type']}, "
            f"BIC {item['bic_before']:.6f} -> {item['bic_after']:.6f}"
        )
    lines.append("")

    lines.append("## 5. 产物清单")
    lines.append("- `fit.png`：目标序列与重构序列对比图")
    lines.append("- `components.png`：M/L/H 三个分量时序图")
    lines.append("- `decomp_hourly.csv`：逐小时分解结果明细")
    lines.append("- `run_meta.json`：运行参数与优化轨迹元数据")
    lines.append("")

    lines.append("## 6. 结果图")
    lines.append("![fit](fit.png)")
    lines.append("")
    lines.append("![components](components.png)")
    lines.append("")

    lines.append("## 7. 复现实验命令")
    lines.append("```powershell")
    lines.append("python scripts/run_experiment.py --case-dir data/raw/chatgpt_release_2022w48 --output-root data/processed")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def generate_report(output_dir: Path, report_name: str = "实验报告.md") -> Path:
    output_dir = Path(output_dir)
    meta_path = output_dir / "run_meta.json"
    decomp_path = output_dir / "decomp_hourly.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"缺少元数据文件: {meta_path}")
    if not decomp_path.exists():
        raise FileNotFoundError(f"缺少分解结果文件: {decomp_path}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    df = pd.read_csv(decomp_path)
    if df.empty:
        raise ValueError("decomp_hourly.csv 为空，无法生成报告")

    text = _build_report_text(output_dir=output_dir, meta=meta, df=df)
    report_path = output_dir / report_name
    report_path.write_text(text, encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="根据实验输出自动生成中文实验报告")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="实验输出目录（需包含 run_meta.json 与 decomp_hourly.csv）",
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default="实验报告.md",
        help="输出报告文件名",
    )
    args = parser.parse_args()

    report_path = generate_report(Path(args.output_dir), args.report_name)
    print(f"[OK] report: {report_path}")


if __name__ == "__main__":
    main()
