from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_CASE_DIR = Path("data/raw/chatgpt_release_2022w48")
META_NAME = "meta.json"
TRENDS_FILE = "trends_hourly.csv"
GDELT_FILE = "gdelt_hourly.csv"
PANEL_FILE = "panel_hourly.csv"
QC_PLOT_FILE = "qc_plot.png"
TWITTER_CANDIDATES = [
    "twitter_hourly.csv",
    "kaggle_hourly.csv",
    "twitter_timeline_hourly.csv",
    "kaggle_timeline_hourly.csv",
    "kaggle_timeline.csv",
]


def _load_meta(meta_path: Path) -> dict:
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json 不存在: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    if not isinstance(meta, dict):
        raise ValueError("meta.json 不是对象结构")
    return meta


def _build_master_index(meta: dict) -> pd.DatetimeIndex:
    start_utc = meta.get("window_start_utc")
    end_utc = meta.get("window_end_utc")
    if not start_utc or not end_utc:
        raise ValueError("meta.json 缺少 window_start_utc / window_end_utc")

    start_dt = pd.to_datetime(start_utc, utc=True)
    end_dt = pd.to_datetime(end_utc, utc=True)
    if end_dt < start_dt:
        raise ValueError("window_end_utc 早于 window_start_utc")
    return pd.date_range(start=start_dt, end=end_dt, freq="h", tz="UTC")


def _pick_time_column(columns: list[str]) -> str:
    preferred = ["ts_utc", "datetime_utc", "datetime", "date", "time", "timestamp", "Date"]
    for col in preferred:
        if col in columns:
            return col
    for col in columns:
        lower = col.lower()
        if "time" in lower or "date" in lower:
            return col
    raise ValueError(f"无法识别时间列: {columns}")


def _pick_value_column(df: pd.DataFrame, preferred: list[str]) -> str:
    for col in preferred:
        if col in df.columns:
            return col
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) == 1:
        return numeric_cols[0]
    raise ValueError(f"无法识别数值列: {df.columns.tolist()}")


def _parse_to_hour_utc(
    series: pd.Series,
    source_name: str,
    allow_invalid: bool = False,
) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if parsed.isna().any() and not allow_invalid:
        raise ValueError(f"{source_name} 时间列存在无法解析的值")
    return parsed.dt.floor("h")


def _load_hourly_trends(case_dir: Path) -> pd.DataFrame:
    path = case_dir / TRENDS_FILE
    if not path.exists():
        raise FileNotFoundError(f"缺少文件: {path}")

    df = pd.read_csv(path)
    time_col = _pick_time_column(df.columns.tolist())
    value_col = _pick_value_column(df, ["trends", "value", "interest", "score"])

    ts_utc = _parse_to_hour_utc(df[time_col], "trends")
    values = pd.to_numeric(df[value_col], errors="coerce")

    out = (
        pd.DataFrame({"ts_utc": ts_utc, "trends": values})
        .groupby("ts_utc", as_index=False)["trends"]
        .mean()
    )
    return out


def _load_hourly_gdelt(case_dir: Path) -> pd.DataFrame:
    path = case_dir / GDELT_FILE
    if not path.exists():
        raise FileNotFoundError(f"缺少文件: {path}")

    df = pd.read_csv(path)
    time_col = _pick_time_column(df.columns.tolist())
    value_col = _pick_value_column(df, ["gdelt_raw", "value", "count", "volume", "raw", "Value"])

    ts_utc = _parse_to_hour_utc(df[time_col], "gdelt")
    values = pd.to_numeric(df[value_col], errors="coerce")

    out = (
        pd.DataFrame({"ts_utc": ts_utc, "gdelt_raw": values})
        .groupby("ts_utc", as_index=False)["gdelt_raw"]
        .sum(min_count=1)
    )
    return out


def _resolve_twitter_path(case_dir: Path) -> Path:
    for name in TWITTER_CANDIDATES:
        path = case_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"未找到 Twitter 数据文件，候选: {TWITTER_CANDIDATES}")


def _load_hourly_twitter(case_dir: Path) -> tuple[pd.DataFrame, Path]:
    path = _resolve_twitter_path(case_dir)
    df = pd.read_csv(path, low_memory=False)

    time_col = _pick_time_column(df.columns.tolist())
    ts_utc = _parse_to_hour_utc(df[time_col], "twitter", allow_invalid=True)
    valid_mask = ts_utc.notna()
    if not valid_mask.any():
        raise ValueError("twitter 时间列全部无法解析")

    count_col = None
    for col in ["twitter_count", "tweet_count", "count", "tweets", "value", "n"]:
        if col in df.columns:
            count_col = col
            break

    if count_col is None:
        values = pd.Series(1.0, index=df.index)
    else:
        values = pd.to_numeric(df[count_col], errors="coerce")
        if values.notna().sum() == 0:
            values = pd.Series(1.0, index=df.index)
    values = values[valid_mask]
    ts_utc = ts_utc[valid_mask]

    out = (
        pd.DataFrame({"ts_utc": ts_utc, "twitter_count": values})
        .groupby("ts_utc", as_index=False)["twitter_count"]
        .sum(min_count=1)
    )
    return out, path


def _merge_panel(
    index_utc: pd.DatetimeIndex,
    trends_df: pd.DataFrame,
    gdelt_df: pd.DataFrame,
    twitter_df: pd.DataFrame,
) -> pd.DataFrame:
    panel = pd.DataFrame({"ts_utc": index_utc})
    panel = panel.merge(trends_df, on="ts_utc", how="left")
    panel = panel.merge(gdelt_df, on="ts_utc", how="left")
    panel = panel.merge(twitter_df, on="ts_utc", how="left")

    panel = panel[["ts_utc", "trends", "gdelt_raw", "twitter_count"]]
    panel["trends"] = (
        pd.to_numeric(panel["trends"], errors="coerce")
        .interpolate(method="linear", limit_direction="both")
        .fillna(0.0)
    )
    panel["gdelt_raw"] = pd.to_numeric(panel["gdelt_raw"], errors="coerce").fillna(0.0)
    panel["twitter_count"] = (
        pd.to_numeric(panel["twitter_count"], errors="coerce").fillna(0.0).round().astype(int)
    )

    if panel[["trends", "gdelt_raw", "twitter_count"]].isna().any().any():
        raise ValueError("缺失值处理后仍存在 NaN")

    panel["ts_utc"] = panel["ts_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return panel


def _minmax_scale(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    min_val = values.min()
    max_val = values.max()
    if pd.isna(min_val) or pd.isna(max_val):
        return pd.Series(0.0, index=series.index)
    span = max_val - min_val
    if span <= 0:
        return pd.Series(0.0, index=series.index)
    return (values - min_val) / span


def _write_qc_plot(panel: pd.DataFrame, case_dir: Path) -> Path:
    plot_df = panel.copy()
    plot_df["ts_utc"] = pd.to_datetime(plot_df["ts_utc"], utc=True)
    plot_df = plot_df.sort_values("ts_utc")

    trends_scaled = _minmax_scale(plot_df["trends"])
    gdelt_scaled = _minmax_scale(plot_df["gdelt_raw"])
    twitter_scaled = _minmax_scale(plot_df["twitter_count"])

    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=150)
    ax.plot(plot_df["ts_utc"], trends_scaled, label="trends (scaled)", linewidth=1.8)
    ax.plot(plot_df["ts_utc"], gdelt_scaled, label="gdelt_raw (scaled)", linewidth=1.8)
    ax.plot(plot_df["ts_utc"], twitter_scaled, label="twitter_count (scaled)", linewidth=1.8)
    ax.set_title("Hourly Panel QC Plot")
    ax.set_xlabel("ts_utc")
    ax.set_ylabel("scaled value (min-max)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    plot_path = case_dir / QC_PLOT_FILE
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


def main() -> None:
    parser = argparse.ArgumentParser(description="按主时间轴合成 panel_hourly.csv")
    parser.add_argument(
        "--case-dir",
        type=str,
        default=str(DEFAULT_CASE_DIR),
        help="包含 meta.json 与小时级 CSV 的目录",
    )
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    meta = _load_meta(case_dir / META_NAME)
    index_utc = _build_master_index(meta)

    trends_df = _load_hourly_trends(case_dir)
    gdelt_df = _load_hourly_gdelt(case_dir)
    twitter_df, twitter_path = _load_hourly_twitter(case_dir)

    panel = _merge_panel(index_utc, trends_df, gdelt_df, twitter_df)
    out_path = case_dir / PANEL_FILE
    panel.to_csv(out_path, index=False, encoding="utf-8-sig")
    plot_path = _write_qc_plot(panel, case_dir)

    print(f"[OK] panel_hourly: {out_path}")
    print(f"[OK] qc_plot: {plot_path}")
    print(f"[INFO] twitter_source: {twitter_path}")
    print(f"[INFO] rows={len(panel)}")


if __name__ == "__main__":
    main()
