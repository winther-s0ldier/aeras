# merge_modis.py -- Add MODIS AOD to all splits
import pandas as pd
import numpy as np
from pathlib import Path

REPO       = Path(__file__).resolve().parent
SPLITS_DIR = REPO / "data" / "splits"
MODIS_CSV  = REPO / "checkpoints" / "modis_aod_station_daily.csv"

# Load MODIS daily AOD
modis = pd.read_csv(MODIS_CSV)
modis["date"] = pd.to_datetime(modis["date"]).dt.date
modis = modis[["station","date","aod_047"]].rename(columns={"aod_047":"modis_aod"})
print(f"[MERGE] MODIS records: {len(modis):,}  |  unique dates: {modis['date'].nunique()}")

# Compute normalization from train split only
train = pd.read_parquet(SPLITS_DIR / "train.parquet")
# Drop any old failed modis columns
old_cols = [c for c in train.columns if "modis" in c.lower()]
if old_cols:
    print(f"[MERGE] Dropping old columns: {old_cols}")
    train = train.drop(columns=old_cols)
train["date"] = pd.to_datetime(train["timestamp"]).dt.date
train_merged = train.merge(modis, on=["station","date"], how="left")
modis_mean = train_merged["modis_aod"].mean()
modis_std  = train_merged["modis_aod"].std()
print(f"[MERGE] Train MODIS coverage: {train_merged['modis_aod'].notna().mean()*100:.1f}%")
print(f"[MERGE] Normalization: mean={modis_mean:.4f}  std={modis_std:.4f}")

splits = ["train","val","test_diwali","test_winter","test_random"]
for split in splits:
    df = pd.read_parquet(SPLITS_DIR / f"{split}.parquet")
    # Drop any old failed modis columns
    old_cols = [c for c in df.columns if "modis" in c.lower()]
    if old_cols:
        df = df.drop(columns=old_cols)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    df = df.merge(modis, on=["station","date"], how="left")

    # Binary availability flag
    df["modis_aod_available"] = df["modis_aod"].notna().astype(np.float32)

    # Normalize: mean-fill missing with training mean
    df["modis_aod_norm"] = ((df["modis_aod"].fillna(modis_mean) - modis_mean)
                            / modis_std).astype(np.float32)

    df = df.drop(columns=["date","modis_aod"])

    df.to_parquet(SPLITS_DIR / f"{split}.parquet", index=False)

    cov = df["modis_aod_available"].mean() * 100
    print(f"  {split}: {len(df):>7} rows  |  MODIS coverage: {cov:.1f}%")

print("\n[MERGE] Done. All splits updated with modis_aod_norm + modis_aod_available")
print(f"  mean_fill={modis_mean:.4f}  std={modis_std:.4f}  (save these for inference)")
