import os
import time
import math
import json
import requests
import pandas as pd
from datetime import datetime, timezone

# =========================
# User config
# =========================
KEYWORD = "ChatGPT"
LANGUAGE = "eng"  # English only

START_Z = "2022-11-28 00:00:00Z"
END_Z   = "2022-12-05 00:00:00Z"

OUT_DIR = r"E:\OneDrive - Coventry University\个人工作\新闻归因\code\data\raw\chatgpt_release_2022w48"
OUT_FILE = "gdelt_timelinevolraw.csv"
META_FILE = "meta.json"

# =========================
# GDELT config
# =========================
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# 每页大小：Doc API支持maxrecords，常用 250；太大可能被拒
MAX_RECORDS = 100

# 轻微限速，避免被暂时封
SLEEP_SECONDS = 0.2

# 网络请求超时
TIMEOUT = 30

# 重试配置（429/5xx 会触发）
MAX_RETRIES = 6
BACKOFF_BASE_SECONDS = 2
BACKOFF_CAP_SECONDS = 60

SESSION = requests.Session()

# =========================
# Helpers
# =========================
def to_gdelt_datetime(z_str: str) -> str:
    """
    Convert 'YYYY-mm-dd HH:MM:SSZ' -> 'YYYYmmddHHMMSS' (UTC)
    """
    dt = datetime.strptime(z_str, "%Y-%m-%d %H:%M:%SZ").replace(tzinfo=timezone.utc)
    return dt.strftime("%Y%m%d%H%M%S")

def to_iso_z(z_str: str) -> str:
    """
    Convert 'YYYY-mm-dd HH:MM:SSZ' -> 'YYYY-mm-ddTHH:MM:SSZ'
    """
    dt = datetime.strptime(z_str, "%Y-%m-%d %H:%M:%SZ").replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def update_meta_json(meta_path: str, query: str, start_z: str, end_z: str, download_time_utc: str):
    # 仅在文件存在时更新，避免误创建
    if not os.path.exists(meta_path):
        print(f"[WARN] meta.json not found: {meta_path}")
        return

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if not isinstance(meta, dict):
        print("[WARN] meta.json is not an object; skip update")
        return

    gdelt = meta.get("gdelt")
    if not isinstance(gdelt, dict):
        gdelt = {}
        meta["gdelt"] = gdelt

    gdelt["query"] = query
    gdelt["start_utc"] = to_iso_z(start_z)
    gdelt["end_utc"] = to_iso_z(end_z)
    gdelt["download_time_utc"] = download_time_utc

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"[INFO] meta.json updated: {meta_path}")

def fetch_page(query: str, startdt: str, enddt: str, start: int):
    """
    Fetch one page of articles from GDELT Doc API.
    We request JSON and then aggregate locally for strict hourly buckets.
    """
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "startdatetime": startdt,
        "enddatetime": enddt,
        "language": LANGUAGE,
        "maxrecords": MAX_RECORDS,
        "startrecord": start,     # 1-based index
        "sort": "HybridRel"       # 相关性+时间混合排序（你也可改成 'datedesc'）
    }

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        r = SESSION.get(GDELT_DOC_API, params=params, timeout=TIMEOUT)

        if r.status_code == 429 or 500 <= r.status_code < 600:
            # 429/5xx：退避重试（优先使用 Retry-After）
            retry_after = r.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                sleep_seconds = int(retry_after)
            else:
                sleep_seconds = min(BACKOFF_CAP_SECONDS, BACKOFF_BASE_SECONDS ** attempt)

            print(f"[WARN] HTTP {r.status_code}, retry in {sleep_seconds}s (attempt {attempt}/{MAX_RETRIES})")
            time.sleep(sleep_seconds)
            last_error = r
            continue

        r.raise_for_status()
        return r.json()

    # 超过最大重试仍失败
    if last_error is not None:
        last_error.raise_for_status()
    raise RuntimeError("GDELT request failed without response")

def parse_seendate_to_hour_utc(seendate_str: str) -> datetime:
    """
    GDELT seendate usually like: '20221128001500' (UTC)
    Return floored-to-hour datetime with tzinfo UTC.
    """
    dt = datetime.strptime(seendate_str, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    dt_hour = dt.replace(minute=0, second=0, microsecond=0)
    return dt_hour

# =========================
# Main
# =========================
def main():
    ensure_dir(OUT_DIR)
    out_path = os.path.join(OUT_DIR, OUT_FILE)

    startdt = to_gdelt_datetime(START_Z)
    enddt   = to_gdelt_datetime(END_Z)

    # GDELT query：精确短语匹配更干净
    # 你也可以用 query = KEYWORD（不带引号）扩大召回
    query = f"\"{KEYWORD}\" sourcelang:{LANGUAGE}"

    # 先拉第一页，得到总量与分页信息
    startrecord = 1
    all_hours = []

    print(f"[INFO] Query={query}, language={LANGUAGE}")
    print(f"[INFO] Range UTC: {startdt} -> {enddt}")
    print(f"[INFO] Output: {out_path}")

    total_articles = None
    fetched = 0

    while True:
        data = fetch_page(query, startdt, enddt, startrecord)

        # total articles
        if total_articles is None:
            total_articles = int(data.get("totalArticles", 0))
            print(f"[INFO] totalArticles={total_articles}")
            if total_articles == 0:
                # 输出空的 hourly 框架也可以（按窗口生成）
                break

        articles = data.get("articles", [])
        if not articles:
            break

        # 解析每条文章的 seendate，floor到小时
        for a in articles:
            seendate = a.get("seendate")
            if not seendate:
                continue
            try:
                h = parse_seendate_to_hour_utc(seendate)
                all_hours.append(h)
            except Exception:
                # 某些记录字段异常时跳过
                continue

        fetched += len(articles)
        print(f"[INFO] fetched={fetched}/{total_articles} (startrecord={startrecord})")

        # 下一页
        startrecord += len(articles)
        if startrecord > total_articles:
            break

        time.sleep(SLEEP_SECONDS)

    # 构造严格 hourly 时间轴（左闭右开：start <= t < end）
    start_dt = datetime.strptime(START_Z, "%Y-%m-%d %H:%M:%SZ").replace(tzinfo=timezone.utc)
    end_dt   = datetime.strptime(END_Z,   "%Y-%m-%d %H:%M:%SZ").replace(tzinfo=timezone.utc)

    hourly_index = pd.date_range(start=start_dt, end=end_dt, freq="H", inclusive="left", tz="UTC")

    if total_articles and total_articles > 0 and all_hours:
        s = pd.Series(all_hours)
        counts = s.value_counts().sort_index()
        counts.index = pd.to_datetime(counts.index).tz_convert("UTC")
        df = pd.DataFrame({"datetime_utc": hourly_index})
        df["gdelt_doc_count"] = df["datetime_utc"].map(counts).fillna(0).astype(int)
    else:
        df = pd.DataFrame({"datetime_utc": hourly_index, "gdelt_doc_count": 0})

    # 你要求“时区列是Z”，所以把输出格式写成 ISO-8601 Z
    df["datetime_utc"] = df["datetime_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # 保存
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved: {out_path} (rows={len(df)})")

    # 记录到 meta.json
    download_time_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta_path = os.path.join(OUT_DIR, META_FILE)
    update_meta_json(meta_path, query, START_Z, END_Z, download_time_utc)

if __name__ == "__main__":
    main()
