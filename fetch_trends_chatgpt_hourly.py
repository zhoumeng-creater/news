import time
from pathlib import Path
import pandas as pd
from pytrends.request import TrendReq


def _iter_time_windows(start_dt: pd.Timestamp, end_dt: pd.Timestamp, window_hours: int, overlap_hours: int):
    if window_hours <= 0:
        raise ValueError("window_hours must be positive")
    if overlap_hours < 0:
        raise ValueError("overlap_hours must be >= 0")
    if overlap_hours >= window_hours:
        raise ValueError("overlap_hours must be smaller than window_hours")

    step = pd.Timedelta(hours=window_hours)
    overlap = pd.Timedelta(hours=overlap_hours)

    current_start = start_dt
    while current_start < end_dt:
        current_end = min(current_start + step, end_dt)
        yield current_start, current_end
        if current_end >= end_dt:
            break
        current_start = current_end - overlap


def _format_timeframe(start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> str:
    return f"{start_dt.strftime('%Y-%m-%dT%H')} {end_dt.strftime('%Y-%m-%dT%H')}"


def _fetch_window(pytrends: TrendReq, keyword: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp,
                  category: int, geo: str, gprop: str, max_retries: int):
    last_err = None
    timeframe = _format_timeframe(start_dt, end_dt)

    for attempt in range(1, max_retries + 1):
        try:
            pytrends.build_payload([keyword], cat=category, geo=geo, gprop=gprop, timeframe=timeframe)
            df = pytrends.interest_over_time()

            if df is None or df.empty:
                raise RuntimeError("Empty data returned by Google Trends (可能被限流/风控或该时间窗无数据).")

            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])

            df = df.rename(columns={keyword: "value"})
            df.index = pd.to_datetime(df.index, utc=True)
            return df[["value"]]
        except Exception as e:
            last_err = e
            wait = min(10, 2 * attempt)
            print(f"[WARN] attempt {attempt}/{max_retries} failed: {e}")
            print(f"       retry in {wait}s ...")
            time.sleep(wait)

    raise RuntimeError(f"Failed window {timeframe} after {max_retries} retries. Last error: {last_err}")


def fetch_hourly_google_trends(
    keyword: str,
    start_utc: str,
    end_utc: str,
    save_dir: str,
    filename: str = "trends_interest_over_time.csv",
    geo: str = "",          # ""=Worldwide; "US","JP","CN" 等
    gprop: str = "",        # ""=web; "news","youtube","images","shopping"
    category: int = 0,      # 0=all categories
    sleep_seconds: float = 2.0,
    max_retries: int = 5,
):
    """
    Fetch hourly Google Trends data within a UTC window and save as CSV (utf-8-sig for Excel).
    start_utc/end_utc format: "YYYY-MM-DD HH:MM:SSZ"
    """

    # --- Parse times (UTC) ---
    start_dt = pd.to_datetime(start_utc, utc=True).floor("h")
    end_dt = pd.to_datetime(end_utc, utc=True).ceil("h")

    if end_dt <= start_dt:
        raise ValueError("end_utc must be later than start_utc")

    # --- Prepare output path ---
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    # --- Build pytrends client ---
    # tz=0 -> UTC (minutes offset)
    pytrends = TrendReq(hl="en-US", tz=0, retries=0, backoff_factor=0)

    window_hours = 24 * 7
    overlap_hours = 1

    combined = None
    last_window_scaled = None
    for w_start, w_end in _iter_time_windows(start_dt, end_dt, window_hours, overlap_hours):
        window_df = _fetch_window(
            pytrends=pytrends,
            keyword=keyword,
            start_dt=w_start,
            end_dt=w_end,
            category=category,
            geo=geo,
            gprop=gprop,
            max_retries=max_retries,
        )

        if last_window_scaled is None:
            scaled_df = window_df
        else:
            overlap = last_window_scaled.join(window_df, how="inner", lsuffix="_prev", rsuffix="_new")
            if overlap.empty:
                scale = 1.0
            else:
                valid = overlap[(overlap["value_prev"] > 0) & (overlap["value_new"] > 0)]
                if not valid.empty:
                    scale = valid["value_prev"].median() / valid["value_new"].median()
                else:
                    prev_mean = overlap["value_prev"].mean()
                    new_mean = overlap["value_new"].mean()
                    scale = prev_mean / new_mean if new_mean > 0 else 1.0

            scaled_df = window_df.copy()
            scaled_df["value"] = scaled_df["value"] * scale

        if combined is None:
            combined = scaled_df
        else:
            combined = pd.concat([combined, scaled_df])
            combined = combined[~combined.index.duplicated(keep="first")].sort_index()

        last_window_scaled = scaled_df
        if sleep_seconds and sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if combined is None or combined.empty:
        raise RuntimeError("No data collected from Google Trends.")

    combined = combined.loc[(combined.index >= start_dt) & (combined.index <= end_dt)]
    max_val = combined["value"].max()
    if pd.notna(max_val) and max_val > 0:
        combined["value"] = (combined["value"] / max_val * 100).round(3)

    combined.insert(0, "keyword", keyword)
    combined.insert(0, "datetime_utc", combined.index.strftime("%Y-%m-%d %H:%M:%SZ"))

    # 额外给一列日本时间（可选；你在东京时区，方便对齐本地事件）
    combined["datetime_jst"] = combined.index.tz_convert("Asia/Tokyo").strftime("%Y-%m-%d %H:%M:%S%z")

    combined = combined[["datetime_utc", "datetime_jst", "keyword", "value"]]

    combined.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[OK] Saved: {out_path}")
    print(combined.head(5).to_string(index=False))
    return combined


if __name__ == "__main__":
    keyword = "ChatGPT"
    start_utc = "2022-11-28 00:00:00Z"
    end_utc = "2022-12-05 00:00:00Z"

    save_dir = r"E:\OneDrive - Coventry University\个人工作\新闻归因\code\data\raw\chatgpt_release_2022w48"
    filename = "trends_interest_over_time.csv"

    # 你也可以把 geo 改成 "US"/"JP" 等看特定国家的趋势
    fetch_hourly_google_trends(
        keyword=keyword,
        start_utc=start_utc,
        end_utc=end_utc,
        save_dir=save_dir,
        filename=filename,
        geo="",
        gprop="",
        category=0,
        sleep_seconds=2.0,
        max_retries=5,
    )
