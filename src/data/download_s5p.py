import os
import sys
import requests
import json
from pathlib import Path
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import RAW_DIR, YEARS

# Bounding box: West, South, East, North
BBOX = "74.0,27.0,78.0,32.0"
WKT_POLYGON = "POLYGON((74.0 27.0, 78.0 27.0, 78.0 32.0, 74.0 32.0, 74.0 27.0))"

def get_cdse_token(email, password):
    print("[S5P] Authenticating with Copernicus Data Space Ecosystem (CDSE)...")
    token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    data = {
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": email,
        "password": password
    }
    resp = requests.post(token_url, data=data)
    resp.raise_for_status()
    return resp.json()["access_token"]

def download_s5p_data():
    load_dotenv()
    email = os.getenv("CDSE_USERNAME")
    password = os.getenv("CDSE_PASSWORD")

    if not email or not password or email == "__FILL_IN__":
        print("[ERROR] CDSE_USERNAME or CDSE_PASSWORD not set in .env")
        return

    s5p_dir = RAW_DIR / "s5p"
    s5p_dir.mkdir(parents=True, exist_ok=True)

    product_type = "L2__AER_AI"
    print(f"[S5P] Searching CDSE for {product_type} over bounding box...")

    search_years = [y for y in YEARS if y >= 2018]

    # ── Phase 1: collect all products to download (search only, no auth needed) ──
    all_products = []
    for year in search_years:
        for month in [10, 11, 12, 1]:  # Diwali + Winter months
            if month == 1 and year == 2018:
                continue  # S5P not fully operational Jan 2018

            start_date = f"{year}-{month:02d}-01T00:00:00.000Z"
            end_month, end_year = month + 1, year
            if end_month == 13:
                end_month, end_year = 1, year + 1
            end_date = f"{end_year}-{end_month:02d}-01T00:00:00.000Z"

            query = (
                f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
                f"$filter=Collection/Name eq 'SENTINEL-5P' "
                f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
                f"    and att/OData.CSC.StringAttribute/Value eq '{product_type}') "
                f"and OData.CSC.Intersects(area=geography'SRID=4326;{WKT_POLYGON}') "
                f"and ContentDate/Start ge {start_date} and ContentDate/Start lt {end_date}&$top=10"
            )

            print(f"[S5P] Searching {year}-{month:02d}...")
            resp = requests.get(query, timeout=30)
            if resp.status_code == 200:
                products = resp.json().get("value", [])
                print(f"       Found {len(products)} products.")
                for prod in products:
                    out_path = s5p_dir / f"{prod['Name']}.nc"
                    if out_path.exists():
                        print(f"  [SKIP] {prod['Name']}")
                    else:
                        all_products.append(prod)
            else:
                print(f"[S5P] Search failed: {resp.status_code} {resp.text[:200]}")

    print(f"\n[S5P] {len(all_products)} new files to download.")
    if not all_products:
        print("[S5P] Nothing to download — all files already present.")
        return

    # ── Phase 2: download each file with a FRESH TOKEN per file ──────────────
    # CDSE OAuth tokens expire after ~10 minutes. Refreshing before every
    # download prevents 401 errors on long multi-hour runs.
    downloaded = 0
    failed = 0
    for i, prod in enumerate(all_products):
        prod_name = prod["Name"]
        prod_id   = prod["Id"]
        out_path  = s5p_dir / f"{prod_name}.nc"

        print(f"  [{i+1}/{len(all_products)}] Downloading {prod_name}...")

        # Fresh token for every file (avoids expiry)
        try:
            access_token = get_cdse_token(email, password)
        except Exception as e:
            print(f"  [ERROR] Token refresh failed: {e} — skipping.")
            failed += 1
            continue

        download_url = (
            f"https://zipper.dataspace.copernicus.eu/odata/v1/"
            f"Products({prod_id})/$value"
        )
        try:
            dl_resp = requests.get(
                download_url,
                headers={"Authorization": f"Bearer {access_token}"},
                stream=True,
                timeout=300,
            )
            dl_resp.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in dl_resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            size_mb = out_path.stat().st_size / 1024 / 1024
            print(f"  [OK]    {size_mb:.1f} MB")
            downloaded += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            if out_path.exists():
                out_path.unlink()  # Remove partial file
            failed += 1

    total_in_folder = len(list(s5p_dir.glob("*.nc")))
    print(f"\n[S5P] Done. Downloaded: {downloaded} | Failed: {failed} | "
          f"Total in folder: {total_in_folder}")

if __name__ == "__main__":
    download_s5p_data()
