import os
import sys
from pathlib import Path
import requests
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import RAW_DIR, YEARS

# FIRMS bbox: [West, South, East, North]
# Capturing Punjab/Haryana/Delhi
BBOX = "74.0,27.0,78.0,32.0"

def download_firms_data():
    load_dotenv()
    api_key = os.getenv("FIRMS_API_KEY")
    
    if not api_key or api_key == "__FILL_IN__":
        print("[ERROR] FIRMS_API_KEY is not set in .env")
        return
        
    firms_dir = RAW_DIR / "firms"
    firms_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[FIRMS] Starting download for years {YEARS}")
    print(f"[FIRMS] Bounding box: {BBOX}")
    
    # We use VIIRS SNPP (VIIRS_SNPP_NRT and VIIRS_SNPP_SP) or MODIS (MODIS_NRT / MODIS_SP)
    # The standard historical archive is usually accessed via SP (Standard Processing)
    source = "VIIRS_SNPP_SP" 
    
    for year in YEARS:
        # We will download the crucial burning months: October and November
        for month in [10, 11]:
            start_date = f"{year}-{month:02d}-01"
            # FIRMS API limit is max 5 days per request, so we chunk it
            # 1-5, 6-10, 11-15, 16-20, 21-25, 26-31
            chunks = [
                (f"{year}-{month:02d}-01", 5),
                (f"{year}-{month:02d}-06", 5),
                (f"{year}-{month:02d}-11", 5),
                (f"{year}-{month:02d}-16", 5),
                (f"{year}-{month:02d}-21", 5),
                (f"{year}-{month:02d}-26", 6 if month == 10 else 5)
            ]
            
            for date_str, days in chunks:
                url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/{source}/{BBOX}/{days}/{date_str}"
                out_path = firms_dir / f"firms_{source}_{date_str}_{days}d.csv"
                
                if out_path.exists():
                    print(f"[FIRMS] Skipping {out_path.name}, already exists.")
                    continue
                    
                print(f"[FIRMS] Requesting {date_str} for {days} days...")
                try:
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(response.text)
                    print(f"[FIRMS] Saved {out_path.name}")
                except Exception as e:
                    print(f"[ERROR] Failed to download {date_str}: {e}")

if __name__ == "__main__":
    download_firms_data()
