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
    
    try:
        access_token = get_cdse_token(email, password)
    except Exception as e:
        print(f"[ERROR] CDSE Authentication failed: {e}")
        return

    headers = {"Authorization": f"Bearer {access_token}"}
    
    # We want Sentinel-5P Level 2 Aerosol Index or NO2
    # Product type: L2__AER_AI or L2__NO2___
    product_type = "L2__AER_AI"
    
    print(f"[S5P] Searching CDSE for {product_type} over bounding box...")
    
    # S5P data started in mid-2018. We will try 2018-2020.
    search_years = [y for y in YEARS if y >= 2018]
    
    for year in search_years:
        for month in [10, 11, 12, 1]:  # Focus on Diwali and Winter months
            if month == 1 and year == 2018:
                continue # S5P wasn't fully operational in Jan 2018
                
            start_date = f"{year}-{month:02d}-01T00:00:00.000Z"
            end_month = month + 1
            end_year = year
            if end_month == 13:
                end_month = 1
                end_year += 1
            end_date = f"{end_year}-{end_month:02d}-01T00:00:00.000Z"
            
            # OData API Query
            query = (
                f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
                f"$filter=Collection/Name eq 'SENTINEL-5P' "
                f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq '{product_type}') "
                f"and OData.CSC.Intersects(area=geography'SRID=4326;{WKT_POLYGON}') "
                f"and ContentDate/Start ge {start_date} and ContentDate/Start lt {end_date}&$top=10"
            )
            
            print(f"[S5P] Searching {year}-{month:02d}...")
            resp = requests.get(query)
            if resp.status_code == 200:
                products = resp.json().get("value", [])
                print(f"[S5P] Found {len(products)} products for {year}-{month:02d} (top 10 capped).")
                for prod in products:
                    prod_id = prod["Id"]
                    prod_name = prod["Name"]
                    out_path = s5p_dir / f"{prod_name}.nc"
                    
                    if out_path.exists():
                        print(f"  [SKIP] {prod_name}")
                        continue
                        
                    download_url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({prod_id})/$value"
                    print(f"  [DOWNLOAD] {prod_name}...")
                    
                    try:
                        dl_resp = requests.get(download_url, headers=headers, stream=True)
                        dl_resp.raise_for_status()
                        with open(out_path, "wb") as f:
                            for chunk in dl_resp.iter_content(chunk_size=8192):
                                f.write(chunk)
                    except Exception as e:
                        print(f"  [ERROR] Download failed: {e}")
            else:
                print(f"[S5P] Search failed: {resp.status_code} {resp.text}")

if __name__ == "__main__":
    download_s5p_data()
