import sys
from pathlib import Path
import urllib.request
import zipfile
import xarray as xr
import numpy as np
import tempfile
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import RAW_DIR, CHECKPOINTS_DIR

EDGAR_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/EDGAR/datasets/v61_AP/PM2.5/TOTALS/EDGARv6.1_PM2.5_2018_TOTALS.0.1x0.1.zip"

def main():
    try:
        edgar_dir = RAW_DIR / "edgar"
        edgar_dir.mkdir(parents=True, exist_ok=True)
        out_file = edgar_dir / "edgar_delhi_pm25.npy"
    except OSError:
        out_file = CHECKPOINTS_DIR / "edgar_delhi_pm25.npy"
    
    if out_file.exists():
        print(f"[EDGAR] Baseline already exists at {out_file}")
        return

    print("[EDGAR] Downloading EDGAR v6.1 PM2.5 Total Emissions (2018)...")
    print("This is a ~6MB global grid, so it should be fast.")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        zip_path = tmp_dir / "edgar.zip"
        
        try:
            req = urllib.request.Request(
                EDGAR_URL, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_f:
                shutil.copyfileobj(response, out_f)
                
            print("[EDGAR] Unzipping...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmp_dir)
                
            nc_files = list(tmp_dir.glob("*.nc"))
            if not nc_files:
                raise FileNotFoundError("Could not find the .nc file inside the zip archive.")
            nc_path = nc_files[0]
            
            print(f"[EDGAR] Processing {nc_path.name}...")
            
            ds = xr.open_dataset(nc_path)
            
            lats = ds.lat.values
            if lats[0] > lats[-1]:
                ds_delhi = ds.sel(lat=slice(28.9, 28.2), lon=slice(76.8, 77.4))
            else:
                ds_delhi = ds.sel(lat=slice(28.2, 28.9), lon=slice(76.8, 77.4))
                
            var_name = [v for v in ds.data_vars if 'emi' in v.lower()][0]
            delhi_grid = ds_delhi[var_name].values.squeeze()
            
            delhi_grid = np.nan_to_num(delhi_grid, 0.0)
            
            if delhi_grid.max() > 0:
                delhi_grid = delhi_grid / delhi_grid.max()
                
            np.save(out_file, delhi_grid)
            print(f"[EDGAR] Successfully saved Delhi baseline: {out_file} (Shape: {delhi_grid.shape})")
            
        except Exception as e:
            print(f"[EDGAR] Error processing baseline: {e}")

if __name__ == "__main__":
    main()
