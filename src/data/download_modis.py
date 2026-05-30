import os
from pathlib import Path
from dotenv import load_dotenv
from src.config import RAW_DIR, YEARS
import earthaccess

def download_modis_data():
    load_dotenv()
    username = os.getenv("EARTHDATA_USERNAME")
    password = os.getenv("EARTHDATA_PASSWORD")
    
    if not username or not password or username == "__FILL_IN__":
        print("[ERROR] EARTHDATA_USERNAME or EARTHDATA_PASSWORD not set in .env")
        return

    modis_dir = RAW_DIR / "modis"
    modis_dir.mkdir(parents=True, exist_ok=True)
    
    # We want MCD19A2 (MAIAC AOD)
    short_name = "MCD19A2"
    
    print(f"[MODIS] Logging into Earthdata via earthaccess...")
    try:
        earthaccess.login(strategy="environment", persist=True)
    except Exception as e:
        print(f"[MODIS] Could not login via environment: {e}")
        # Try passing directly
        os.environ["EARTHDATA_USERNAME"] = username
        os.environ["EARTHDATA_PASSWORD"] = password
        earthaccess.login(strategy="environment", persist=True)

    print(f"[MODIS] Querying NASA CMR for {short_name} over bounding box...")
    
    for year in YEARS:
        for month in [10, 11, 12, 1]:  # Focus on Diwali and Winter months
            actual_year = year if month >= 10 else year + 1
            if actual_year > 2020:
                continue
                
            start_date = f"{actual_year}-{month:02d}-01"
            # Get last day of month
            if month in [10, 12, 1]:
                end_date = f"{actual_year}-{month:02d}-31"
            else:
                end_date = f"{actual_year}-{month:02d}-30"
                
            print(f"[MODIS] Searching {actual_year}-{month:02d}...")
            
            # Delhi bounding box: W, S, E, N
            bounding_box = (76.8, 28.4, 77.4, 28.9)
            
            results = earthaccess.search_data(
                short_name=short_name,
                bounding_box=bounding_box,
                temporal=(start_date, end_date),
                count=100
            )
            
            if not results:
                print("  No granules found.")
                continue
                
            print(f"  Found {len(results)} granules. Downloading...")
            earthaccess.download(results, local_path=str(modis_dir))
    
    print("\n[MODIS] Download complete.")

if __name__ == "__main__":
    download_modis_data()
