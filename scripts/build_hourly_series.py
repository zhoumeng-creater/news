from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd


DEFAULT_CASE_DIR = Path("data/raw/chatgpt_release_2022w48")
META_NAME = "meta.json"

TRENDS_IN_NAME = "trends_interest_over_time.csv"
GDELT_IN_NAME = "gdelt_timelinevolraw.csv"
TWITTER_IN_NAME = "kaggle_data.xlsx"
TWITTER_FALLBACK_CSV = "kaggle_timeline.csv"

TRENDS_OUT_NAME = "trends_hourly.csv"
GDELT_OUT_NAME = "gdelt_hourly.csv"
TWITTER_OUT_NAME = "twitter_hourly.csv"

TIME_CANDIDATES = ["datetime_utc", "datetime", "date", "time", "timestamp", "Date", "created_at"]
TEXT_CANDIDATES = ["text", "tweet", "content"]
LANG_CANDIDATES = ["lang", "language"]
RETWEET_BOOL_CANDIDATES = ["is_retweet", "retweeted", "retweet"]
CHATGPT_PATTERN = r"chat\s*gpt|chatgpt|openai"


def _load_meta(meta_path: Path) -> dict:
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json 不存在: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("meta.json 不是对象结构")
    return data


def _save_meta(meta_path: Path, meta: dict) -> None:
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _format_utc(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_index_from_bounds(start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DatetimeIndex:
    if end_dt < start_dt:
        raise ValueError("window_end_utc 早于 window_start_utc")
    return pd.date_range(start=start_dt, end=end_dt, freq="h", tz="UTC")


def _build_master_index_from_meta(meta: dict) -> pd.DatetimeIndex:
    start_utc = meta.get("window_start_utc")
    end_utc = meta.get("window_end_utc")
    if not start_utc or not end_utc:
        raise ValueError("meta.json 缺少 window_start_utc / window_end_utc")
    start_dt = pd.to_datetime(start_utc, utc=True)
    end_dt = pd.to_datetime(end_utc, utc=True)
    return _build_index_from_bounds(start_dt, end_dt)


def _pick_column(columns: list[str], candidates: list[str], partial: bool = True) -> str | None:
    lower_map = {str(col).strip().lower(): col for col in columns}
    for cand in candidates:
        matched = lower_map.get(cand.lower())
        if matched is not None:
            return matched
    if partial:
        for col in columns:
            lowered = str(col).strip().lower()
            for cand in candidates:
                if cand.lower() in lowered:
                    return col
    return None


def _pick_time_column(columns: list[str]) -> str:
    col = _pick_column(columns, TIME_CANDIDATES, partial=True)
    if col is None:
        raise ValueError(f"无法识别时间列: {columns}")
    return col


def _pick_value_column(df: pd.DataFrame, preferred: list[str]) -> str:
    for col in preferred:
        if col in df.columns:
            return col
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) == 1:
        return numeric_cols[0]
    raise ValueError(f"无法识别数值列: {df.columns.tolist()}")


def _tz_offset_label(ts: pd.Timestamp) -> str:
    offset = ts.strftime("%z")
    if not offset:
        return "UTC"
    return f"{offset[:3]}:{offset[3:]}"


def _infer_trends_tz(meta: dict) -> tuple[object, str]:
    trends_meta = meta.get("trends", {})
    export_time_local = trends_meta.get("export_time_local")
    if export_time_local:
        ts = pd.to_datetime(export_time_local, errors="coerce")
        if pd.notna(ts) and ts.tzinfo is not None:
            return ts.tzinfo, "meta.trends.export_time_local"
    return "UTC", "fallback_utc"


def _parse_trends_time(series: pd.Series, meta: dict, time_col: str) -> tuple[pd.Series, dict]:
    parsed = pd.to_datetime(series, errors="coerce", utc=False)
    if parsed.isna().any():
        raise ValueError("Trends 时间列存在无法解析的值")

    if parsed.dt.tz is None:
        assumed_tz, source = _infer_trends_tz(meta)
        parsed = parsed.dt.tz_localize(assumed_tz)
        tz_note = {
            "time_column": time_col,
            "mode": "assumed",
            "assumed_tz": _tz_offset_label(parsed.iloc[0]),
            "assumed_tz_source": source,
            "note": "时间列无时区，先按导出时区本地化后再转 UTC",
        }
    else:
        tz_note = {
            "time_column": time_col,
            "mode": "source",
            "source_tz": str(parsed.dt.tz),
            "note": "时间列自带时区标记，直接转 UTC",
        }

    parsed = parsed.dt.tz_convert("UTC")
    return parsed, tz_note


def _parse_gdelt_time(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=False)
    if parsed.isna().any():
        raise ValueError("GDELT 时间列存在无法解析的值")
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize("UTC")
    else:
        parsed = parsed.dt.tz_convert("UTC")
    return parsed


def _write_trends(case_dir: Path, index_utc: pd.DatetimeIndex, meta: dict) -> dict:
    trends_path = case_dir / TRENDS_IN_NAME
    df = pd.read_csv(trends_path)

    time_col = _pick_time_column(df.columns.tolist())
    value_col = _pick_value_column(df, ["value", "trends", "interest", "score"])

    parsed_time, tz_note = _parse_trends_time(df[time_col], meta, time_col)
    df = df.assign(ts_utc=parsed_time.dt.floor("h"))

    hourly = (
        df.groupby("ts_utc")[value_col]
        .mean()
        .sort_index()
        .reindex(index_utc)
        .astype(float)
        .interpolate(method="linear", limit_direction="both")
    )

    out_df = pd.DataFrame({"ts_utc": index_utc, "trends": hourly.values})
    out_df["ts_utc"] = out_df["ts_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out_df.to_csv(case_dir / TRENDS_OUT_NAME, index=False, encoding="utf-8-sig")
    return tz_note


def _write_gdelt(case_dir: Path, index_utc: pd.DatetimeIndex) -> None:
    gdelt_path = case_dir / GDELT_IN_NAME
    df = pd.read_csv(gdelt_path)

    time_col = _pick_time_column(df.columns.tolist())
    value_col = _pick_value_column(df, ["value", "count", "volume", "raw", "Value"])

    parsed_time = _parse_gdelt_time(df[time_col]).dt.floor("h")
    values = pd.to_numeric(df[value_col], errors="coerce")

    hourly = (
        pd.DataFrame({"ts_utc": parsed_time, "value": values})
        .groupby("ts_utc")["value"]
        .sum(min_count=1)
        .sort_index()
        .reindex(index_utc)
        .fillna(0.0)
    )

    out_df = pd.DataFrame({"ts_utc": index_utc, "gdelt_raw": hourly.values})
    out_df["ts_utc"] = out_df["ts_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out_df.to_csv(case_dir / GDELT_OUT_NAME, index=False, encoding="utf-8-sig")


def _read_excel(path: Path, header: int | None = 0) -> pd.DataFrame:
    try:
        return pd.read_excel(path, sheet_name=0, header=header)
    except PermissionError:
        temp_path = Path(tempfile.gettempdir()) / f"{path.stem}_codex_copy{path.suffix}"
        shutil.copy2(path, temp_path)
        return pd.read_excel(temp_path, sheet_name=0, header=header)


def _all_unnamed(columns: pd.Index) -> bool:
    names = [str(col).strip().lower() for col in columns]
    return bool(names) and all((not name) or name.startswith("unnamed") for name in names)


def _load_twitter_raw_dataframe(input_path: Path, fallback_csv: Path | None) -> tuple[pd.DataFrame, str]:
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    if input_path.suffix.lower() in {".xlsx", ".xls"}:
        try:
            df = _read_excel(input_path, header=0)
            source_note = str(input_path)
            if _all_unnamed(df.columns):
                df = _read_excel(input_path, header=None).dropna(how="all").reset_index(drop=True)
                rename_map = {0: "created_at", 1: "text", 2: "user", 3: "retweet_count"}
                df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
            return df, source_note
        except Exception as exc:
            if fallback_csv is None or not fallback_csv.exists():
                raise RuntimeError(f"读取 Excel 失败，且无可用 fallback CSV: {exc}") from exc
            return pd.read_csv(fallback_csv, low_memory=False), f"{fallback_csv} (fallback)"

    if input_path.suffix.lower() == ".csv":
        return pd.read_csv(input_path, low_memory=False), str(input_path)

    raise ValueError(f"不支持的输入格式: {input_path.suffix}")


def _parse_bool_like(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    out = pd.Series(pd.NA, index=series.index, dtype="boolean")

    true_tokens = {"true", "t", "yes", "y", "1"}
    false_tokens = {"false", "f", "no", "n", "0"}
    out[normalized.isin(true_tokens)] = True
    out[normalized.isin(false_tokens)] = False

    num = pd.to_numeric(series, errors="coerce")
    out[num == 1] = True
    out[num == 0] = False
    return out


def _parse_time_to_utc(series: pd.Series, assume_tz: str) -> tuple[pd.Series, str]:
    parsed = pd.to_datetime(series, errors="coerce", utc=False)
    if parsed.isna().all():
        raise ValueError("Twitter 时间列全部无法解析")

    try:
        source_tz = parsed.dt.tz
    except AttributeError:
        parsed_utc = pd.to_datetime(series, errors="coerce", utc=True)
        if parsed_utc.isna().all():
            raise ValueError("Twitter 时间列无法解析为 datetime")
        return parsed_utc, "mixed_or_unknown"

    if source_tz is None:
        localized = parsed.dt.tz_localize(assume_tz)
        return localized.dt.tz_convert("UTC"), f"assumed_{assume_tz}"
    return parsed.dt.tz_convert("UTC"), f"source_{source_tz}"


def _prepare_twitter_records(
    raw_df: pd.DataFrame,
    assume_tz: str,
    chatgpt_only: bool,
    drop_retweets: bool,
) -> tuple[pd.DataFrame, dict]:
    columns = [str(col) for col in raw_df.columns]
    time_col = _pick_column(columns, ["created_at", "date", "datetime", "timestamp", "time"], partial=True)
    text_col = _pick_column(columns, TEXT_CANDIDATES, partial=True)

    if time_col is None:
        raise ValueError(f"无法识别 Twitter 时间列，当前列: {columns}")
    if text_col is None:
        raise ValueError(f"无法识别 Twitter 文本列，当前列: {columns}")

    data = pd.DataFrame({"created_at_raw": raw_df[time_col], "text": raw_df[text_col]})
    data["text"] = data["text"].astype(str).str.strip()
    data = data[data["text"] != ""].copy()

    lang_col = _pick_column(columns, LANG_CANDIDATES, partial=True)
    if lang_col is not None:
        data["lang"] = raw_df[lang_col]

    is_retweet_col = _pick_column(columns, RETWEET_BOOL_CANDIDATES, partial=False)
    if is_retweet_col is not None:
        parsed_bool = _parse_bool_like(raw_df[is_retweet_col]).reindex(data.index)
        if parsed_bool.notna().any():
            data["is_retweet"] = parsed_bool.fillna(False).astype(bool)
    if "is_retweet" not in data.columns:
        data["is_retweet"] = data["text"].str.startswith("RT @", na=False)

    created_at_utc, tz_mode = _parse_time_to_utc(data["created_at_raw"], assume_tz)
    valid_mask = created_at_utc.notna()
    if not valid_mask.any():
        raise ValueError("Twitter 时间列可解析数据为空")

    data = data.loc[valid_mask].copy()
    data["created_at_utc"] = created_at_utc.loc[valid_mask]

    unique_days = data["created_at_utc"].dt.floor("D").nunique()
    if unique_days <= 1:
        raise ValueError(f"Twitter 时间分布仅覆盖 {unique_days} 天，疑似时间字段错误")

    if chatgpt_only:
        mask = data["text"].str.contains(CHATGPT_PATTERN, case=False, regex=True, na=False)
        data = data.loc[mask].copy()

    if drop_retweets:
        data = data.loc[~data["is_retweet"].astype(bool)].copy()

    if data.empty:
        raise ValueError("Twitter 过滤后数据为空，请检查过滤条件")

    return data, {
        "time_col": time_col,
        "text_col": text_col,
        "lang_col": lang_col,
        "is_retweet_col": is_retweet_col or "derived_from_text_prefix_RT",
        "tz_mode": tz_mode,
        "rows_after_filter": int(len(data)),
        "unique_days": int(unique_days),
    }


def _limit_to_first_days(
    data: pd.DataFrame,
    first_days: int,
) -> tuple[pd.DataFrame, pd.Timestamp | None, pd.Timestamp | None]:
    if first_days <= 0:
        return data, None, None

    start_day = data["created_at_utc"].min().floor("D")
    end_exclusive = start_day + pd.Timedelta(days=first_days)
    limited = data[(data["created_at_utc"] >= start_day) & (data["created_at_utc"] < end_exclusive)].copy()
    if limited.empty:
        raise ValueError(f"仅保留前 {first_days} 天后 Twitter 数据为空")
    return limited, start_day, end_exclusive - pd.Timedelta(hours=1)


def _write_twitter(
    case_dir: Path,
    output_name: str,
    input_name: str,
    fallback_csv_name: str,
    assume_tz: str,
    chatgpt_only: bool,
    drop_retweets: bool,
    first_days: int,
    window_start_utc: str | None,
    window_end_utc: str | None,
    index_utc: pd.DatetimeIndex | None,
) -> dict:
    input_path = case_dir / input_name
    fallback_csv = case_dir / fallback_csv_name if fallback_csv_name else None
    raw_df, source_note = _load_twitter_raw_dataframe(input_path, fallback_csv)

    data, summary = _prepare_twitter_records(
        raw_df=raw_df,
        assume_tz=assume_tz,
        chatgpt_only=chatgpt_only,
        drop_retweets=drop_retweets,
    )
    data, first_window_start, first_window_end = _limit_to_first_days(data, first_days)

    if index_utc is None:
        start_dt = pd.to_datetime(window_start_utc, utc=True) if window_start_utc else first_window_start
        end_dt = pd.to_datetime(window_end_utc, utc=True) if window_end_utc else first_window_end
        if start_dt is None:
            start_dt = data["created_at_utc"].min().floor("h")
        if end_dt is None:
            end_dt = data["created_at_utc"].max().floor("h")
        index_utc = _build_index_from_bounds(start_dt, end_dt)
    else:
        start_dt = index_utc.min()
        end_dt = index_utc.max()

    ts_hour = data["created_at_utc"].dt.floor("h")
    grouped = ts_hour.value_counts().sort_index().rename("twitter_count")
    hourly = grouped.reindex(index_utc, fill_value=0).astype(int)

    out_df = pd.DataFrame(
        {"ts_utc": index_utc.strftime("%Y-%m-%dT%H:%M:%SZ"), "twitter_count": hourly.values}
    )
    output_path = case_dir / output_name
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return {
        "output_path": output_path,
        "source": source_note,
        "time_col": summary["time_col"],
        "text_col": summary["text_col"],
        "lang_col": summary["lang_col"],
        "is_retweet_col": summary["is_retweet_col"],
        "tz_mode": summary["tz_mode"],
        "rows_after_filter": summary["rows_after_filter"],
        "unique_days": summary["unique_days"],
        "window_start_utc": _format_utc(start_dt),
        "window_end_utc": _format_utc(end_dt),
        "first_days": first_days,
        "output_rows": len(out_df),
        "index_utc": index_utc,
    }


def _assert_output_matches_index(output_path: Path, index_utc: pd.DatetimeIndex) -> None:
    df = pd.read_csv(output_path)
    if "ts_utc" not in df.columns:
        raise ValueError(f"{output_path} 缺少 ts_utc 列")

    ts = pd.to_datetime(df["ts_utc"], errors="coerce", utc=True)
    if ts.isna().any():
        raise ValueError(f"{output_path} 存在无法解析的时间值")
    if len(df) != len(index_utc):
        raise ValueError(f"{output_path} 行数与主时间轴不一致: {len(df)} != {len(index_utc)}")
    if ts.min() != index_utc.min() or ts.max() != index_utc.max():
        raise ValueError(
            f"{output_path} 时间范围不一致: "
            f"{ts.min()}~{ts.max()} vs {index_utc.min()}~{index_utc.max()}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="清洗并对齐小时级 Trends/GDELT/Twitter")
    parser.add_argument("--case-dir", type=str, default=str(DEFAULT_CASE_DIR), help="案例目录")
    parser.add_argument(
        "--with-twitter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否同时生成 twitter_hourly.csv",
    )
    parser.add_argument("--twitter-input-name", type=str, default=TWITTER_IN_NAME, help="Twitter 输入文件名")
    parser.add_argument(
        "--twitter-fallback-csv",
        type=str,
        default=TWITTER_FALLBACK_CSV,
        help="Twitter Excel 读取失败时 fallback 文件名",
    )
    parser.add_argument("--twitter-output-name", type=str, default=TWITTER_OUT_NAME, help="Twitter 输出文件名")
    parser.add_argument("--twitter-assume-tz", type=str, default="UTC", help="Twitter 无时区时默认时区")
    parser.add_argument(
        "--twitter-chatgpt-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Twitter 是否仅保留 ChatGPT/OpenAI 相关文本",
    )
    parser.add_argument(
        "--twitter-drop-retweets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Twitter 是否过滤 retweet",
    )
    parser.add_argument(
        "--twitter-use-meta-window",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Twitter 是否强制使用 meta.json 时间窗（默认否）",
    )
    parser.add_argument("--twitter-window-start-utc", type=str, default=None, help="Twitter 时间窗起点")
    parser.add_argument("--twitter-window-end-utc", type=str, default=None, help="Twitter 时间窗终点")
    parser.add_argument(
        "--twitter-first-days",
        type=int,
        default=7,
        help="Twitter 仅保留前 N 天（默认 7；传 0 关闭截取）",
    )
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    meta_path = case_dir / META_NAME
    meta = _load_meta(meta_path)

    twitter_result = None
    if args.with_twitter:
        index_from_meta = _build_master_index_from_meta(meta) if args.twitter_use_meta_window else None
        meta_start = meta.get("window_start_utc")
        meta_end = meta.get("window_end_utc")
        tw_start = args.twitter_window_start_utc or (meta_start if args.twitter_use_meta_window else None)
        tw_end = args.twitter_window_end_utc or (meta_end if args.twitter_use_meta_window else None)

        twitter_result = _write_twitter(
            case_dir=case_dir,
            output_name=args.twitter_output_name,
            input_name=args.twitter_input_name,
            fallback_csv_name=args.twitter_fallback_csv,
            assume_tz=args.twitter_assume_tz,
            chatgpt_only=args.twitter_chatgpt_only,
            drop_retweets=args.twitter_drop_retweets,
            first_days=args.twitter_first_days,
            window_start_utc=tw_start,
            window_end_utc=tw_end,
            index_utc=index_from_meta,
        )
        index_utc = twitter_result["index_utc"]
    else:
        index_utc = _build_master_index_from_meta(meta)

    tz_note = _write_trends(case_dir, index_utc, meta)
    _write_gdelt(case_dir, index_utc)

    trends_meta = meta.get("trends")
    if not isinstance(trends_meta, dict):
        trends_meta = {}
        meta["trends"] = trends_meta
    trends_meta["timezone_infer"] = tz_note

    meta["window_start_utc"] = _format_utc(index_utc.min())
    meta["window_end_utc"] = _format_utc(index_utc.max())
    processing = meta.get("processing")
    if not isinstance(processing, dict):
        processing = {}
        meta["processing"] = processing
    time_index = processing.get("time_index")
    if not isinstance(time_index, dict):
        time_index = {}
        processing["time_index"] = time_index
    time_index["type"] = "complete_hourly"
    time_index["freq"] = "1H"
    time_index["timezone"] = "UTC"
    time_index["basis"] = "window_start_utc~window_end_utc"

    _save_meta(meta_path, meta)

    _assert_output_matches_index(case_dir / TRENDS_OUT_NAME, index_utc)
    _assert_output_matches_index(case_dir / GDELT_OUT_NAME, index_utc)
    if args.with_twitter:
        _assert_output_matches_index(case_dir / args.twitter_output_name, index_utc)

    print(f"[OK] trends_hourly: {case_dir / TRENDS_OUT_NAME}")
    print(f"[OK] gdelt_hourly: {case_dir / GDELT_OUT_NAME}")
    if twitter_result is not None:
        print(f"[OK] twitter_hourly: {twitter_result['output_path']}")
        print(
            f"[INFO] twitter_window={twitter_result['window_start_utc']}~{twitter_result['window_end_utc']} "
            f"rows={twitter_result['output_rows']} first_days={twitter_result['first_days']}"
        )
        print(
            f"[INFO] twitter_cols time={twitter_result['time_col']} text={twitter_result['text_col']} "
            f"is_retweet={twitter_result['is_retweet_col']} lang={twitter_result['lang_col']}"
        )
    else:
        print("[INFO] twitter_hourly skipped")
    print(
        f"[INFO] unified_window={_format_utc(index_utc.min())}~{_format_utc(index_utc.max())} "
        f"hours={len(index_utc)}"
    )
    print(f"[OK] meta.json updated: {meta_path}")


if __name__ == "__main__":
    main()

