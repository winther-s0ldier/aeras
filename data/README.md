# aeras — Data Directory

All raw and processed data files are **gitignored** to keep the repo lightweight.
This file documents exactly how to reconstruct every dataset from scratch.

> **Shortcut:** The fully processed, ready-to-train splits already exist as Kaggle datasets.
> You only need the sections below if you want to re-run preprocessing from raw sources.

---

## Kaggle Datasets (fastest path — no downloading required)

| Dataset | Kaggle slug | Contents |
|---|---|---|
| Training splits | `b1042rudrakumar/kaggle-dataset-fixed` | `train/val/test_*.parquet` |
| Processed data | `b1042rudrakumar/aeras-data` | `merged_hourly.parquet`, `station_coords.csv`, `collocation_points.npy`, `normalized_params.json` |

Download via Kaggle CLI:
```bash
kaggle datasets download b1042rudrakumar/kaggle-dataset-fixed -p data/splits --unzip
kaggle datasets download b1042rudrakumar/aeras-data -p data/processed --unzip
```

---

## 1. CPCB Station Data (PM2.5 / NO2 / O3 / SO2)

**Source:** Central Pollution Control Board — Delhi NCR  
**Years used:** 2018–2022  
**Stations:** ~40 across Delhi NCR  
**Resolution:** Hourly

### Download options

**Option A — Kaggle backup (easiest):**
```bash
kaggle datasets download deepaksirohiwal/delhi-air-quality -p data/raw/cpcb --unzip
```

**Option B — CPCB portal (official):**
1. Go to https://app.cpcbccr.com/ccr
2. Select station → Download → CSV (hourly, PM2.5/NO2/O3/SO2)
3. Save to `data/raw/cpcb/<station_name>.csv`

**Script:** `src/data/download_cpcb.py`

---

## 2. ERA5 Wind & Meteorology

**Source:** Copernicus Climate Data Store (CDS)  
**Variables:** `10m_u_component_of_wind`, `10m_v_component_of_wind`, `2m_temperature`, `boundary_layer_height`  
**Bounding box:** 27°N–32°N, 76°E–78°E (Delhi NCR)  
**Resolution:** 0.25° × 0.25°, hourly

### Setup
```bash
pip install cdsapi
# Create ~/.cdsapirc with your CDS API key:
# url: https://cds.climate.copernicus.eu/api/v2
# key: <UID>:<API-KEY>   (get from https://cds.climate.copernicus.eu/user)
```

### Download
```bash
python src/data/download_era5.py
# Output: data/raw/era5/era5_delhi_ncr.nc  (~36 MB for 2018-2022)
```

---

## 3. MODIS MAIAC AOD (MCD19A2)

**Source:** NASA Earthdata — LP DAAC  
**Product:** MCD19A2 v061 — 1 km daily aerosol optical depth  
**Tile:** h24v06 (covers Delhi NCR)  
**Months downloaded:** Oct/Nov/Dec/Jan (Diwali + Winter seasons)  
**Files:** ~579 HDF4 files, ~7 GB total

### Setup
1. Create account at https://urs.earthdata.nasa.gov
2. Authorize apps: **LP DAAC Data Pool** and **LAADS DAAC**
3. Generate a Bearer token at https://urs.earthdata.nasa.gov/users/<your-uid>/user_tokens
4. Put token in `src/data/download_modis.py` → `TOKEN_CLI` variable

### Download raw HDF4 files
```bash
python src/data/download_modis.py
# Output: data/raw/modis/*.hdf  (~7 GB)
```

### Process HDF4 → station daily CSV
Requires pyhdf (needs conda, NOT pip — native DLLs required on Windows):
```bash
conda install -c conda-forge pyhdf pyarrow -y
conda run python process_modis_aod.py
# Output: data/modis_aod_station_daily.csv  (already committed to repo)
```

> **Note:** `data/modis_aod_station_daily.csv` is already committed to the repo.
> You only need to re-run this if you expand to full-year MODIS data.

### Full-year MODIS (future work)
To get all 12 months (improves train coverage from 16% → ~45%):
```python
# In src/data/download_modis.py, change:
TARGET_MONTHS = list(range(1, 13))   # all 12 months instead of [10, 11, 12, 1]
# Then re-run download + process_modis_aod.py
# Expected: ~1,740 files, ~21 GB
```

---

## 4. EDGAR Emission Inventory

**Source:** European Commission JRC — https://edgar.jrc.ec.europa.eu  
**Dataset:** EDGAR v6.1 — gridded PM2.5 emissions by sector  
**Resolution:** 0.1° × 0.1°  
**Coverage:** Global (crop to Delhi NCR bbox: 27.5–29.0°N, 76.5–77.8°E)

### Download
1. Go to https://edgar.jrc.ec.europa.eu/dataset_ap61
2. Select: PM2.5 → All sectors → 2019 → Download NetCDF
3. Save to `data/raw/edgar/`
4. Cropped output already in Kaggle dataset as `edgar_delhi_pm25.npy` (15×13 grid)

---

## 5. FIRMS Fire Counts (VIIRS)

**Source:** NASA FIRMS — https://firms.modaps.eosdis.nasa.gov  
**Dataset:** VIIRS S-NPP 375m active fire detections  
**Use:** Stubble burning proxy (Punjab/Haryana fires affecting Delhi AQ)

### Download
```bash
# API key from https://firms.modaps.eosdis.nasa.gov/api/area/
# Bounding box: 27.5,76.5,30.5,78.5 (wider, captures Punjab)
curl "https://firms.modaps.eosdis.nasa.gov/api/area/csv/<API_KEY>/VIIRS_SNPP_NRT/27.5,76.5,30.5,78.5/2018-01-01/2022-12-31" \
     -o data/raw/firms/viirs_firms_2018_2022.csv
```

---

## 6. Sentinel-5P NO2 Column (S5P)

**Source:** Copernicus Open Access Hub / Google Earth Engine  
**Product:** TROPOMI L2 NO2 tropospheric column  
**Resolution:** 3.5 × 5.5 km, daily

### Download (scaffolded, not yet integrated)
```bash
python src/data/download_s5p.py
# Output: data/raw/s5p/
```

---

## Preprocessing Pipeline (reconstruct splits from raw)

Once raw data is downloaded, run in order:

```bash
# 1. Merge CPCB + ERA5 + FIRMS into hourly parquet
python src/data/preprocess.py

# 2. Create train/val/test splits
python src/data/split.py

# 3. Merge MODIS AOD into splits
python merge_modis.py

# Output: data/splits/train.parquet, val.parquet, test_diwali.parquet,
#         test_winter.parquet, test_random.parquet
```

---

## Directory Structure (after reconstruction)

```
data/
├── README.md                          ← this file (committed)
├── modis_aod_station_daily.csv        ← committed to repo
├── raw/
│   ├── cpcb/                          ← ~297 MB
│   ├── era5/                          ← ~36 MB
│   ├── modis/                         ← ~7 GB (HDF4 files)
│   ├── edgar/
│   ├── firms/
│   └── s5p/
├── processed/
│   ├── merged_hourly.parquet
│   ├── station_coords.csv
│   ├── collocation_points.npy
│   └── normalized_params.json
└── splits/
    ├── train.parquet
    ├── val.parquet
    ├── test_diwali.parquet
    ├── test_winter.parquet
    └── test_random.parquet
```
