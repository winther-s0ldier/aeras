"""
download_modis.py  —  Download MCD19A2 (MODIS MAIAC AOD) for Delhi NCR
Uses NASA CMR API for search + Bearer token for direct download.
No earthaccess needed.
"""

import os, calendar, time
from pathlib import Path
from dotenv import load_dotenv
import requests

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# ── Config ─────────────────────────────────────────────────────────
RAW_DIR    = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
MODIS_DIR  = RAW_DIR / "modis"

# Full Delhi NCR bounding box  (W, S, E, N)
WEST, SOUTH, EAST, NORTH = 76.5, 27.5, 77.8, 29.0

# MCD19A2 = MAIAC AOD, 1 km, daily (best AOD product for urban areas)
SHORT_NAME = "MCD19A2"
VERSION    = "061"                      # current collection

TARGET_MONTHS = [10, 11, 12, 1]        # Oct, Nov, Dec, Jan (Diwali + Winter)
YEARS         = [2018, 2019, 2020, 2021, 2022]

CMR_URL  = "https://cmr.earthdata.nasa.gov/search/granules.json"
TOKEN    = os.getenv("EARTHDATA_TOKEN", "")

# ── Helpers ─────────────────────────────────────────────────────────
def search_granules(start_date: str, end_date: str, page: int = 1):
    params = {
        "short_name":   SHORT_NAME,
        "version":      VERSION,
        "temporal":     f"{start_date}T00:00:00Z,{end_date}T23:59:59Z",
        "bounding_box": f"{WEST},{SOUTH},{EAST},{NORTH}",
        "page_size":    200,
        "page_num":     page,
    }
    r = requests.get(CMR_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("feed", {}).get("entry", [])


def get_download_url(entry):
    """Pick the best download link from a CMR granule entry."""
    links = entry.get("links", [])
    for link in links:
        href = link.get("href", "")
        rel  = link.get("rel", "")
        # Prefer direct HDF download links
        if "data#" in rel and href.endswith(".hdf"):
            return href
    # Fallback: first https link that ends in .hdf
    for link in links:
        href = link.get("href", "")
        if href.startswith("https") and href.endswith(".hdf"):
            return href
    return None


def download_file(url: str, dest: Path, token: str):
    headers = {"Authorization": f"Bearer {token}"}
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        dest.write_bytes(r.content)
    return dest.stat().st_size


# ── Main ─────────────────────────────────────────────────────────────
def download_modis_data(token: str = None):
    token = token or TOKEN
    if not token:
        raise ValueError("No EARTHDATA_TOKEN. Set it in .env or pass directly.")

    MODIS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[MODIS] Product: {SHORT_NAME} v{VERSION}")
    print(f"[MODIS] Bbox: W={WEST} S={SOUTH} E={EAST} N={NORTH}")
    print(f"[MODIS] Output: {MODIS_DIR}\n")

    total_found = total_downloaded = total_skipped = 0

    for year in YEARS:
        for month in TARGET_MONTHS:
            actual_year = year if month >= 10 else year + 1
            if actual_year > 2022:
                continue

            last_day   = calendar.monthrange(actual_year, month)[1]
            start_date = f"{actual_year}-{month:02d}-01"
            end_date   = f"{actual_year}-{month:02d}-{last_day:02d}"

            granules = search_granules(start_date, end_date)
            total_found += len(granules)
            print(f"[MODIS] {actual_year}-{month:02d}: {len(granules)} granules found")

            for g in granules:
                url = get_download_url(g)
                if not url:
                    total_skipped += 1
                    continue

                fname = url.split("/")[-1]
                dest  = MODIS_DIR / fname

                if dest.exists() and dest.stat().st_size > 1_000_000:
                    total_skipped += 1
                    continue

                try:
                    sz = download_file(url, dest, token)
                    total_downloaded += 1
                    print(f"  OK {fname}  ({sz//1024:,} KB)")
                except Exception as e:
                    print(f"  FAIL {fname}: {e}")
                    total_skipped += 1
                    time.sleep(2)   # back-off on error

    print(f"\n[MODIS] Done.")
    print(f"  Found: {total_found}  |  Downloaded: {total_downloaded}  |  Skipped: {total_skipped}")
    print(f"  Files in {MODIS_DIR}: {len(list(MODIS_DIR.glob('*.hdf')))}")


if __name__ == "__main__":
    TOKEN_CLI = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6InJ1ZHJhazIxIiwiZXhwIjoxNzg1MDc4ODk2LCJpYXQiOjE3Nzk4OTQ4OTYsImlzcyI6Imh0dHBzOi8vdXJzLmVhcnRoZGF0YS5uYXNhLmdvdiIsImlkZW50aXR5X3Byb3ZpZGVyIjoiZWRsX29wcyIsImFjciI6ImVkbCIsImFzc3VyYW5jZV9sZXZlbCI6M30.BT-n3w8apFJyt4pmomcb4GSzB8dCT1oTDnTknf9h27tMWtzOEMF2KF1b3gPdR2ova8kSpfP6JRms0jnEwJAL2oCgUqJwWP4NR-LWnezaEhVbBkQpoep4PWqF8Qmv35dRkW0iWr3WDyn2E_LkKFDY_Nn4XQgzQLdIQ37MCgFghacVNQBJi1pcRYVbN_Ja9Bd7EmFtCOqBL9q-mPJXIR3cUV0y-TyygyblomUp2GgOgk8XFNpXttFJhk11hJrs3JcefTAtD5MR_cEtchYZjR1b3DvUN9uB7QHcIo2qvnbd5xzil0eQ_TuXXPHqDGkG-3G6JllWqsOPo_aAFqZZo5JFBg"
    download_modis_data(token=TOKEN_CLI)
