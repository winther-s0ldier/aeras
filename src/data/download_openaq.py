"""
OpenAQ v3 bulk downloader for Delhi NCR stations (2018-2025).

Discovers all CPCB-linked stations within the Delhi NCR bounding box,
then downloads PM2.5, NO2, O3, SO2 hourly measurements year by year.

Output: data/raw/cpcb/openaq/openaq_{year}.parquet per year.
Schema: timestamp (UTC tz-aware), station, pm25, no2, o3, so2, latitude, longitude.
        Matches the return schema of preprocess.load_station_hour_data().

Run:
    .\.venv\Scripts\python.exe src/data/download_openaq.py
    or
    .\.venv\Scripts\python.exe main.py download-openaq   (once wired in main.py)

Re-run is safe: skips years whose parquet already exists.
"""

import sys
import time
import json
import calendar
from pathlib import Path

import requests
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import RAW_DIR, DELHI_LAT_RANGE, DELHI_LON_RANGE

# ── Config ─────────────────────────────────────────────────────────────────────
OPENAQ_BASE    = "https://api.openaq.org/v3"
PARAMETERS     = ["pm25", "no2", "o3", "so2"]
DOWNLOAD_YEARS = list(range(2018, 2026))     # 2018 inclusive, 2026 exclusive = 2018-2025
OUTPUT_DIR     = RAW_DIR / "cpcb" / "openaq"
PAGE_SIZE      = 1000
REQUEST_DELAY  = 1.2   # seconds; well under 60 req/min API-key limit


# ── Session ────────────────────────────────────────────────────────────────────
def _make_session() -> requests.Session:
    import os
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
    api_key = os.getenv("OPENAQ_API", "")
    if not api_key:
        raise RuntimeError(
            "OPENAQ_API not found in .env. "
            "Get a key at https://explore.openaq.org and add OPENAQ_API=<key> to .env"
        )
    s = requests.Session()
    s.headers.update({"X-API-Key": api_key, "Accept": "application/json"})
    return s


# ── HTTP helper ────────────────────────────────────────────────────────────────
def _get(session: requests.Session, url: str, params: dict, retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 60))
                print(f"[OPENAQ] Rate limited - waiting {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code in (500, 502, 503, 504):
                print(f"[OPENAQ] Server error {resp.status_code}, retry {attempt + 1}/{retries}")
                time.sleep(15)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt == retries - 1:
                print(f"[OPENAQ] Request failed after {retries} attempts: {exc}")
                return {}
            time.sleep(10)
    return {}


# ── Location discovery ─────────────────────────────────────────────────────────
def discover_locations(session: requests.Session) -> list:
    """
    Returns all Delhi NCR locations that have at least one of our target parameters.
    Each entry: {id, name, latitude, longitude, sensors: {param: sensor_id}}.
    Cached to openaq/locations.json after first run.
    """
    cache_path = OUTPUT_DIR / "locations.json"
    if cache_path.exists():
        locations = json.loads(cache_path.read_text())
        print(f"[OPENAQ] Loaded {len(locations)} locations from cache ({cache_path.name})")
        return locations

    lat_min, lat_max = DELHI_LAT_RANGE
    lon_min, lon_max = DELHI_LON_RANGE
    bbox = f"{lon_min},{lat_min},{lon_max},{lat_max}"

    locations = []
    page = 1
    while True:
        data = _get(session, f"{OPENAQ_BASE}/locations", {
            "bbox":  bbox,
            "limit": PAGE_SIZE,
            "page":  page,
        })
        time.sleep(REQUEST_DELAY)

        results = data.get("results", [])
        if not results:
            break

        for loc in results:
            sensors = {}
            for sensor in loc.get("sensors", []):
                param_info = sensor.get("parameter") or {}
                param = param_info.get("name", "").lower()
                if param in PARAMETERS:
                    sensors[param] = sensor["id"]

            if not sensors:
                continue

            coords = loc.get("coordinates") or {}
            lat = coords.get("latitude")
            lon = coords.get("longitude")
            if lat is None or lon is None:
                continue

            locations.append({
                "id":        loc["id"],
                "name":      loc.get("name", f"openaq_{loc['id']}"),
                "latitude":  float(lat),
                "longitude": float(lon),
                "sensors":   sensors,
            })

        meta  = data.get("meta", {})
        found = int(meta.get("found", 0))
        if not results or page * PAGE_SIZE >= found:
            break
        page += 1

    print(f"[OPENAQ] Found {len(locations)} Delhi NCR locations with target parameters")
    cache_path.write_text(json.dumps(locations, indent=2))
    print(f"[OPENAQ] Location list cached to {cache_path.name}")
    return locations


# ── Measurement download ───────────────────────────────────────────────────────
def _parse_utc(result: dict) -> str | None:
    """
    Extract a UTC datetime string from a measurement result.
    Handles both v3 hourly-aggregate format (period.datetimeTo.utc)
    and raw-measurement format (date.utc).
    """
    period = result.get("period") or {}
    dt_to = (period.get("datetimeTo") or {}).get("utc")
    if dt_to:
        return dt_to
    date = result.get("date") or {}
    return date.get("utc")


def _download_sensor_month(
    session: requests.Session,
    sensor_id: int,
    year: int,
    month: int,
) -> list:
    """Download all measurements for one sensor in one calendar month."""
    _, n_days = calendar.monthrange(year, month)
    date_from = f"{year}-{month:02d}-01T00:00:00Z"
    date_to   = f"{year}-{month:02d}-{n_days:02d}T23:59:59Z"

    rows = []
    page = 1
    while True:
        data = _get(session, f"{OPENAQ_BASE}/sensors/{sensor_id}/measurements", {
            "datetime_from": date_from,
            "datetime_to":   date_to,
            "limit":         PAGE_SIZE,
            "page":          page,
        })
        time.sleep(REQUEST_DELAY)

        results = data.get("results", [])
        for r in results:
            dt  = _parse_utc(r)
            val = r.get("value")
            if dt is not None and val is not None:
                rows.append({"timestamp_utc": dt, "value": float(val)})

        meta  = data.get("meta", {})
        found = int(meta.get("found", 0))
        if not results or page * PAGE_SIZE >= found:
            break
        page += 1

    return rows


# ── Year assembly ──────────────────────────────────────────────────────────────
def _download_year(
    session: requests.Session,
    locations: list,
    year: int,
) -> pd.DataFrame:
    """
    Download one full year for all locations.
    Returns DataFrame: timestamp, station, pm25, no2, o3, so2, latitude, longitude.
    """
    station_frames = []

    for loc in locations:
        name = loc["name"]
        lat  = loc["latitude"]
        lon  = loc["longitude"]

        param_series = {}
        for param, sensor_id in loc["sensors"].items():
            monthly = []
            for month in range(1, 13):
                monthly.extend(_download_sensor_month(session, sensor_id, year, month))

            if not monthly:
                continue

            s = (
                pd.DataFrame(monthly)
                  .drop_duplicates("timestamp_utc")
                  .set_index("timestamp_utc")["value"]
                  .rename(param)
            )
            param_series[param] = s
            print(f"  [{year}] {name} / {param.upper()}: {len(s):,} readings")

        if not param_series:
            continue

        df_loc = pd.concat(param_series.values(), axis=1)
        df_loc.index = pd.to_datetime(df_loc.index, utc=True)
        df_loc.index.name = "timestamp"
        df_loc = df_loc.reset_index()
        df_loc["station"]   = name
        df_loc["latitude"]  = lat
        df_loc["longitude"] = lon
        station_frames.append(df_loc)

    if not station_frames:
        return pd.DataFrame()

    df = pd.concat(station_frames, ignore_index=True)

    # Ensure all four pollutant columns exist even if a parameter had no data
    for col in PARAMETERS:
        if col not in df.columns:
            df[col] = float("nan")

    col_order = ["timestamp", "station", "pm25", "no2", "o3", "so2", "latitude", "longitude"]
    return df[col_order]


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("  OpenAQ v3 - Delhi NCR Download (2018-2025)")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    session   = _make_session()
    locations = discover_locations(session)

    if not locations:
        print("[OPENAQ] No locations found. Check bounding box and API key.")
        return

    failed = []
    for year in DOWNLOAD_YEARS:
        out_path = OUTPUT_DIR / f"openaq_{year}.parquet"
        if out_path.exists():
            size_mb = out_path.stat().st_size / 1e6
            print(f"[OPENAQ] {out_path.name} exists ({size_mb:.1f} MB) - skipping")
            continue

        print(f"[OPENAQ] --- {year} -------------------------------------------")
        try:
            df = _download_year(session, locations, year)
            if df.empty:
                print(f"[OPENAQ] {year}: no data returned - skipping")
                continue

            df.to_parquet(out_path, index=False)
            n_rows     = len(df)
            n_stations = df["station"].nunique()
            pm25_cov   = df["pm25"].notna().mean() * 100
            print(
                f"[OPENAQ] Saved {out_path.name}: "
                f"{n_rows:,} rows | {n_stations} stations | PM2.5 coverage {pm25_cov:.1f}%"
            )
        except Exception as exc:
            print(f"[OPENAQ] {year} failed: {exc}")
            failed.append(year)

    print("\n" + "=" * 60)
    if failed:
        print(f"[OPENAQ] Failed years: {failed}")
        print("[OPENAQ] Re-run to retry - existing files are skipped automatically.")
    else:
        print("[OPENAQ] All years complete.")
        print(f"[OPENAQ] Output: {OUTPUT_DIR}")
        print("[OPENAQ] Next step: update src/data/preprocess.py to merge OpenAQ data.")


if __name__ == "__main__":
    main()


