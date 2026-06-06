# process_modis_aod.py  --  Extract MODIS MCD19A2 AOD at CPCB station locations
# Run with conda Python: miniconda/python.exe process_modis_aod.py

import numpy as np
import pandas as pd
from pathlib import Path
from pyhdf.SD import SD, SDC
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
import warnings, time
warnings.filterwarnings("ignore")

REPO      = Path(__file__).resolve().parent
MODIS_DIR = REPO / "data" / "raw" / "modis"
SPLITS_DIR= REPO / "data" / "splits"
COORDS_F  = REPO / "data" / "processed" / "station_coords.csv"
OUT_CSV   = REPO / "checkpoints" / "modis_aod_station_daily.csv"

# MODIS sinusoidal projection constants
R          = 6371007.181   # sphere radius (m)
T          = 1111950.5196  # tile size (m) = 10° at R
PIXEL_SIZE = 926.625433    # 1km pixel size (m) = T/1200
TILE_H, TILE_V = 24, 6    # Delhi NCR is in h24v06
NROWS = NCOLS = 1200

# Scale factor and fill for MCD19A2 Optical_Depth_047
AOD_SCALE  = 0.001
AOD_FILL   = -28672         # raw fill value
AOD_VALID  = (-100, 5000)   # raw valid range → × 0.001 = -0.1 to 5.0

SAVE_EVERY = 50

# ── Projection: lat/lon → pixel (row, col) in tile h24v06 ──────────────────
def latlon_to_pixel(lat, lon):
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    x     = R * lon_r * np.cos(lat_r)
    y     = R * lat_r
    x_ul  = (TILE_H - 18) * T
    y_ul  = (9 - TILE_V)  * T
    col   = int((x - x_ul) / PIXEL_SIZE)
    row   = int((y_ul - y) / PIXEL_SIZE)
    return row, col

# ── Load station coords ────────────────────────────────────────────────────
coords = pd.read_csv(COORDS_F)
print(f"[MODIS] Stations: {len(coords)}")

# Pre-compute pixel index for each station
station_pixels = []
for _, row in coords.iterrows():
    r, c = latlon_to_pixel(row["latitude"], row["longitude"])
    # Clamp to valid range (stations near edge)
    r = max(0, min(NROWS - 1, r))
    c = max(0, min(NCOLS - 1, c))
    station_pixels.append((row["station"], r, c))

# ── Load PM2.5 daily means ─────────────────────────────────────────────────
print("[MODIS] Loading PM2.5 data...")
dfs = []
for split in ["train", "val", "test_diwali", "test_winter", "test_random"]:
    p = SPLITS_DIR / f"{split}.parquet"
    if p.exists():
        dfs.append(pd.read_parquet(p, columns=["station","timestamp","pm25"]))
all_data = pd.concat(dfs).drop_duplicates()
all_data["date"] = pd.to_datetime(all_data["timestamp"]).dt.date
daily_pm25 = (all_data.groupby(["station","date"])
              .agg(pm25=("pm25","mean")).reset_index())
date_station_map = {(r.date, r.station): r.pm25
                    for r in daily_pm25.itertuples()}
print(f"[MODIS] Daily PM2.5 records: {len(daily_pm25):,}")

# ── Process HDF4 files ─────────────────────────────────────────────────────
hdf_files = sorted(MODIS_DIR.glob("*.hdf"))
print(f"[MODIS] HDF files: {len(hdf_files)}\n")

results = []
skipped = 0

for fi, f in enumerate(hdf_files):
    try:
        # Parse date from filename: MCD19A2.AYYYYDDD.*
        # e.g. MCD19A2.A2018274.h24v06...
        parts  = f.stem.split(".")
        jdate  = next((p for p in parts if p.startswith("A") and len(p)==8), None)
        if jdate is None:
            skipped += 1; continue
        year   = int(jdate[1:5])
        doy    = int(jdate[5:8])
        date   = (pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=doy-1)).date()

        # Open HDF4
        sd   = SD(str(f), SDC.READ)
        sds  = sd.select("Optical_Depth_047")
        data = sds.get()     # shape: (nobs, NROWS, NCOLS) or (NROWS, NCOLS)
        sds.endaccess()
        sd.end()

        # Handle time dimension if present
        if data.ndim == 3:
            # Take mean across observations (usually 2: Terra AM + PM)
            aod_raw = data.astype(float)
            aod_raw[aod_raw == AOD_FILL] = np.nan
            aod_grid = np.nanmean(aod_raw, axis=0)
        else:
            aod_grid = data.astype(float)
            aod_grid[aod_grid == AOD_FILL] = np.nan

        aod_grid *= AOD_SCALE  # apply scale factor

        # Extract at each station pixel
        for station, row, col in station_pixels:
            aod_val = aod_grid[row, col]
            if np.isnan(aod_val) or aod_val < -0.05 or aod_val > 5.0:
                continue
            pm25_val = date_station_map.get((date, station))
            if pm25_val is None or np.isnan(pm25_val):
                continue
            results.append({
                "date":    date,
                "station": station,
                "aod_047": round(float(aod_val), 4),
                "pm25":    float(pm25_val),
            })

    except Exception as e:
        skipped += 1
        continue

    if (fi + 1) % SAVE_EVERY == 0:
        print(f"  {fi+1:>4}/{len(hdf_files)}  matched={len(results):>6}  skipped={skipped}")
        if results:
            pd.DataFrame(results).to_csv(OUT_CSV, index=False)

# ── Correlation analysis ───────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"TOTAL MATCHED: {len(results)}  |  Skipped: {skipped}")

if len(results) < 20:
    print("Too few matches — check HDF structure.")
else:
    df = pd.DataFrame(results).dropna()
    df.to_csv(OUT_CSV, index=False)

    r_p, pval = pearsonr(df["aod_047"], df["pm25"])
    r_s, _    = spearmanr(df["aod_047"], df["pm25"])

    print(f"\nOVERALL  n={len(df)}")
    print(f"  Pearson  r = {r_p:+.4f}  (p={pval:.2e})")
    print(f"  Spearman r = {r_s:+.4f}")
    print(f"\nAOD_047: mean={df['aod_047'].mean():.3f}  std={df['aod_047'].std():.3f}  "
          f"min={df['aod_047'].min():.3f}  max={df['aod_047'].max():.3f}")
    print(f"PM2.5:   mean={df['pm25'].mean():.1f}  std={df['pm25'].std():.1f}")

    df["month"] = pd.to_datetime(df["date"]).dt.month
    print(f"\nBY MONTH:")
    for m in sorted(df["month"].unique()):
        sub = df[df["month"]==m]
        if len(sub) < 5: continue
        r, _ = pearsonr(sub["aod_047"], sub["pm25"])
        print(f"  Month {m:02d}: r={r:+.4f}  n={len(sub):>5}  "
              f"PM2.5={sub['pm25'].mean():>6.1f}  AOD={sub['aod_047'].mean():>6.3f}")

    diwali = df[df["month"].isin([10,11])]
    winter = df[df["month"].isin([12,1])]
    print(f"\nSEASONAL:")
    for label, sub in [("Diwali Oct-Nov", diwali), ("Winter Dec-Jan", winter)]:
        if len(sub) < 5: continue
        r, _ = pearsonr(sub["aod_047"], sub["pm25"])
        print(f"  {label}: r={r:+.4f}  n={len(sub)}  PM2.5={sub['pm25'].mean():.1f}")

    print(f"\n{'='*60}")
    if r_p > 0.5:
        print(f"VERDICT: r={r_p:.3f} > 0.5  -> STRONG — proceed with v24 (dual satellite)")
    elif r_p > 0.3:
        print(f"VERDICT: r={r_p:.3f} 0.3-0.5 -> MODERATE — worth trying v24")
    else:
        print(f"VERDICT: r={r_p:.3f} < 0.3  -> WEAK — skip v24, finalise v22")

    print(f"\nSaved -> {OUT_CSV}")
