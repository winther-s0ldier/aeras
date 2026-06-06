"""
sanity_check_aer_ai.py  —  AER_AI vs PM2.5 correlation check (fast version)
Uses netCDF4 directly (faster than xarray), saves every 50 files,
skips files that take too long.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import netCDF4 as nc
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
import signal, time, json

REPO         = Path(__file__).resolve().parent
S5P_DIR      = REPO / "data" / "raw" / "s5p"
SPLITS_DIR   = REPO / "data" / "splits"
OUT_CSV      = REPO / "checkpoints" / "aer_ai_pm25_correlation.csv"

MAX_DIST_DEG = 0.15
QA_THRESHOLD = 0.5
SAVE_EVERY   = 50
FILE_TIMEOUT = 30   # seconds per file before skipping

# ── Load station PM2.5 ────────────────────────────────────────────────────────
print("[CHECK] Loading PM2.5 data...")
dfs = []
for split in ["train", "val", "test_diwali", "test_winter", "test_random"]:
    p = SPLITS_DIR / f"{split}.parquet"
    if p.exists():
        dfs.append(pd.read_parquet(p, columns=["station","timestamp","pm25","latitude","longitude"]))
all_data = pd.concat(dfs).drop_duplicates()
all_data["date"] = pd.to_datetime(all_data["timestamp"]).dt.date

daily_pm25 = (all_data.groupby(["station","date"])
              .agg(pm25=("pm25","mean"), latitude=("latitude","first"),
                   longitude=("longitude","first"))
              .reset_index())
print(f"[CHECK] Daily PM2.5 records: {len(daily_pm25):,}")

# Pre-build date index for fast lookup
date_groups = {d: grp for d, grp in daily_pm25.groupby("date")}

# ── Process S5P files ─────────────────────────────────────────────────────────
s5p_files = sorted(S5P_DIR.glob("*.nc"))
print(f"[CHECK] Found {len(s5p_files)} S5P files\n")

results = []
skipped = 0

for fi, f in enumerate(s5p_files):
    t_start = time.time()
    try:
        # Parse date from filename
        parts = f.stem.replace(".nc","").split("_")
        date_str = next((p[:8] for p in parts
                         if len(p) >= 8 and p[:4] in ["2018","2019","2020","2021","2022"]), None)
        if date_str is None:
            skipped += 1; continue

        date = pd.Timestamp(date_str).date()
        if date not in date_groups:
            skipped += 1; continue

        day_df = date_groups[date]

        # Read with netCDF4 (much faster than xarray)
        ds = nc.Dataset(f)
        grp = ds["PRODUCT"]

        # Use masked arrays so fill value (9.96921e+36) → NaN automatically
        lat = np.ma.filled(grp["latitude"][0],  fill_value=np.nan)
        lon = np.ma.filled(grp["longitude"][0], fill_value=np.nan)
        aer = np.ma.filled(grp["aerosol_index_354_388"][0], fill_value=np.nan)
        qa  = np.ma.filled(grp["qa_value"][0],  fill_value=0.0)

        ds.close()

        # Quality filter — mask low-QA pixels
        aer = np.where(qa >= QA_THRESHOLD, aer, np.nan)

        lat_f = lat.flatten().astype(float)
        lon_f = lon.flatten().astype(float)
        aer_f = aer.flatten()
        ok    = np.isfinite(aer_f) & np.isfinite(lat_f) & np.isfinite(lon_f)

        if ok.sum() < 5:
            skipped += 1; continue

        # KD-tree lookup
        tree = cKDTree(np.column_stack([lat_f[ok], lon_f[ok]]))
        aer_ok = aer_f[ok]

        st_coords = np.column_stack([day_df["latitude"].values,
                                     day_df["longitude"].values])
        dists, idxs = tree.query(st_coords, k=1)

        for i, (dist, idx) in enumerate(zip(dists, idxs)):
            if dist > MAX_DIST_DEG:
                continue
            pm25_val = day_df["pm25"].iloc[i]
            if np.isnan(pm25_val):
                continue
            results.append({
                "date":    date,
                "station": day_df["station"].iloc[i],
                "aer_ai":  float(aer_ok[idx]),
                "pm25":    float(pm25_val),
                "dist_deg": float(dist),
            })

        elapsed = time.time() - t_start
        if elapsed > FILE_TIMEOUT:
            print(f"  [WARN] File {f.name} took {elapsed:.1f}s")

    except Exception as e:
        skipped += 1
        continue

    if (fi + 1) % SAVE_EVERY == 0:
        elapsed_total = time.time()
        print(f"  {fi+1:>5}/{len(s5p_files)}  matched={len(results):>6}  skipped={skipped}")
        # Save intermediate
        if results:
            pd.DataFrame(results).to_csv(OUT_CSV, index=False)

# ── Correlation analysis ──────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"TOTAL MATCHED: {len(results)}  |  Skipped: {skipped}")

if len(results) < 20:
    print("Too few matches.")
else:
    df = pd.DataFrame(results).dropna(subset=["aer_ai","pm25"])
    df.to_csv(OUT_CSV, index=False)

    r_p, pval_p = pearsonr(df["aer_ai"], df["pm25"])
    r_s, _      = spearmanr(df["aer_ai"], df["pm25"])

    print(f"\nOVERALL  n={len(df)}")
    print(f"  Pearson  r = {r_p:+.4f}  (p={pval_p:.2e})")
    print(f"  Spearman r = {r_s:+.4f}")
    print(f"\nAER_AI: mean={df['aer_ai'].mean():.3f}  std={df['aer_ai'].std():.3f}  "
          f"min={df['aer_ai'].min():.3f}  max={df['aer_ai'].max():.3f}")
    print(f"PM2.5:  mean={df['pm25'].mean():.1f}   std={df['pm25'].std():.1f}")

    df["month"] = pd.to_datetime(df["date"]).dt.month
    print(f"\nBY MONTH:")
    for m in sorted(df["month"].unique()):
        sub = df[df["month"]==m]
        if len(sub) < 5: continue
        r, _ = pearsonr(sub["aer_ai"], sub["pm25"])
        print(f"  Month {m:02d}: r={r:+.4f}  n={len(sub):>4}  "
              f"PM2.5={sub['pm25'].mean():>6.1f}  AER_AI={sub['aer_ai'].mean():>6.3f}")

    diwali = df[df["month"].isin([10,11])]
    winter = df[df["month"].isin([12,1])]
    print(f"\nSEASONAL:")
    for label, sub in [("Diwali Oct-Nov", diwali), ("Winter Dec-Jan", winter)]:
        if len(sub) < 5: continue
        r, _ = pearsonr(sub["aer_ai"], sub["pm25"])
        print(f"  {label}: r={r:+.4f}  n={len(sub)}  PM2.5={sub['pm25'].mean():.1f}")

    print(f"\n{'='*60}")
    if r_p > 0.5:
        print(f"VERDICT: r={r_p:.3f} > 0.5  -> PROCEED with pixel observation loss")
    elif r_p > 0.3:
        print(f"VERDICT: r={r_p:.3f} 0.3-0.5  -> MODERATE, limited gain expected")
    else:
        print(f"VERDICT: r={r_p:.3f} < 0.3  -> WEAK, satellite observation loss unlikely to help")

    print(f"\nSaved -> {OUT_CSV}")
