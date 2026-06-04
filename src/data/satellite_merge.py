"""
satellite_merge.py
==================
Standalone script: reads already-downloaded MODIS and S5P files,
merges them into the existing parquet splits, and writes updated splits.

Run:  python src/data/satellite_merge.py

Does NOT re-download anything. Does NOT run training.
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

# ── repo root on sys.path ────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from src.config import RAW_DIR, SPLITS_DIR, DELHI_LAT_RANGE, DELHI_LON_RANGE

MODIS_DIR  = RAW_DIR / "modis"
S5P_DIR    = RAW_DIR / "s5p"

# Delhi NCR bounding box
LAT_LO, LAT_HI = DELHI_LAT_RANGE   # e.g. 27.5, 29.0
LON_LO, LON_HI = DELHI_LON_RANGE   # e.g. 76.5, 77.8

# Nearest-neighbour radius (degrees) — ~25 km MODIS, ~50 km S5P
MODIS_RADIUS_DEG = 0.23   # ~25 km at Delhi latitude
S5P_RADIUS_DEG   = 0.45   # ~50 km at Delhi latitude


# ─────────────────────────────────────────────────────────────────────────────
# MODIS parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_modis_rasterio(path: Path) -> pd.DataFrame | None:
    """Read a single MCD19A2 HDF4 file using rasterio (bundles GDAL+HDF4)."""
    try:
        import rasterio
        # MCD19A2 subdataset names
        subdataset_candidates = [
            f'HDF4_EOS:EOS_GRID:"{path}":grid1km:Optical_Depth_047',
            f'HDF4_EOS:EOS_GRID:"{path}":grid1km:Optical_Depth_055',
        ]
        # Parse date from filename: MCD19A2.AYYDDD.*
        stem  = path.stem.split(".")
        try:
            ydoy_part = next(p for p in stem if p.startswith("A") and len(p) == 8)
            year = int(ydoy_part[1:5])
            doy  = int(ydoy_part[5:8])
            file_date = datetime(year, 1, 1) + timedelta(days=doy - 1)
        except Exception:
            return None

        aod_047 = aod_055 = None
        for sds in subdataset_candidates:
            try:
                with rasterio.open(sds) as src:
                    data   = src.read(1).astype(np.float32)
                    nodata = src.nodata
                    tags   = src.tags()
                    scale  = float(tags.get("scale_factor", 0.001))
                    if nodata is not None:
                        data = np.where(data == nodata, np.nan, data)
                    data = data * scale
                    # filter to physical range 0–5
                    data = np.where((data < 0) | (data > 5), np.nan, data)

                    # Get lat/lon grid from transform
                    rows, cols = np.meshgrid(
                        np.arange(src.height), np.arange(src.width), indexing="ij"
                    )
                    xs, ys = rasterio.transform.xy(src.transform, rows.ravel(), cols.ravel())
                    lons = np.array(xs)
                    lats = np.array(ys)
                    vals = data.ravel()

                    # Filter to Delhi NCR
                    mask = (
                        (lats >= LAT_LO) & (lats <= LAT_HI) &
                        (lons >= LON_LO) & (lons <= LON_HI) &
                        np.isfinite(vals)
                    )
                    if mask.sum() == 0:
                        continue

                    if "047" in sds:
                        aod_047 = (lats[mask], lons[mask], vals[mask])
                    else:
                        aod_055 = (lats[mask], lons[mask], vals[mask])
            except Exception:
                continue

        # Build per-pixel DataFrame with whichever band we got
        src_data = aod_047 if aod_047 is not None else aod_055
        if src_data is None:
            return None

        lats_d, lons_d, vals_d = src_data
        df = pd.DataFrame({
            "lat":       lats_d,
            "lon":       lons_d,
            "datetime":  file_date,
            "aod_047":   vals_d if aod_047 is not None else np.nan,
            "aod_055":   vals_d if aod_055 is not None else np.nan,
        })
        return df

    except ImportError:
        return None
    except Exception as e:
        return None


def parse_modis_all(modis_dir: Path) -> pd.DataFrame:
    """
    Parse all MODIS HDF4 files. Tries rasterio first. Returns a DataFrame
    with columns [lat, lon, datetime, aod_047, aod_055] for all Delhi pixels.
    """
    hdf_files = sorted(modis_dir.rglob("*.hdf"))
    if not hdf_files:
        print(f"[MODIS] No .hdf files found in {modis_dir}")
        return pd.DataFrame()

    print(f"[MODIS] Found {len(hdf_files)} HDF4 files. Attempting to parse...")

    # Test first file to see which reader works
    test_result = _parse_modis_rasterio(hdf_files[0])
    if test_result is not None:
        reader = _parse_modis_rasterio
        reader_name = "rasterio"
    else:
        print("[MODIS] WARNING: rasterio reader failed on test file.")
        print("[MODIS] pyhdf/GDAL/rasterio are unavailable on this machine.")
        print("[MODIS] MODIS AOD will be left as NaN in the splits.")
        print("[MODIS] To fix: install pyhdf via conda (conda install -c conda-forge pyhdf)")
        return pd.DataFrame()

    print(f"[MODIS] Using reader: {reader_name}")

    frames = []
    failed = 0
    for i, f in enumerate(hdf_files):
        if i % 100 == 0 and i > 0:
            print(f"[MODIS]   {i}/{len(hdf_files)} files processed ({len(frames)} successful)...")
        result = reader(f)
        if result is not None and len(result) > 0:
            frames.append(result)
        else:
            failed += 1

    if failed > 0:
        print(f"[MODIS] {failed}/{len(hdf_files)} files failed or had no Delhi pixels.")

    if not frames:
        print("[MODIS] No valid MODIS data extracted.")
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["datetime"] = pd.to_datetime(out["datetime"])
    print(f"[MODIS] Parsed {len(out):,} Delhi pixels across "
          f"{out['datetime'].nunique()} unique dates.")
    aod_col = "aod_047" if out["aod_047"].notna().any() else "aod_055"
    valid = out[aod_col].dropna()
    print(f"[MODIS] AOD range: {valid.min():.3f} – {valid.max():.3f}, "
          f"mean={valid.mean():.3f} (n={len(valid):,})")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# S5P parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_s5p_all(s5p_dir: Path) -> pd.DataFrame:
    """
    Parse all Sentinel-5P NetCDF files. Returns a DataFrame with columns
    [lat, lon, datetime, aer_ai] for all Delhi pixels.
    """
    import netCDF4 as nc4

    nc_files = sorted(s5p_dir.rglob("*.nc"))
    if not nc_files:
        print(f"[S5P] No .nc files found in {s5p_dir}")
        return pd.DataFrame()

    print(f"[S5P] Found {len(nc_files)} NetCDF files.")

    var_candidates = [
        "absorbing_aerosol_index",
        "aerosol_index_354_388",
        "aerosol_index_340_380",
    ]

    frames = []
    failed = 0

    for i, f in enumerate(nc_files):
        try:
            with nc4.Dataset(str(f), "r") as ds:
                grp = ds.groups.get("PRODUCT", ds)

                # Find variable
                var_name = next(
                    (v for v in var_candidates if v in grp.variables), None
                )
                if var_name is None:
                    failed += 1
                    continue

                lat  = grp.variables["latitude"][:].data.flatten()
                lon  = grp.variables["longitude"][:].data.flatten()
                vals = grp.variables[var_name][:].data.flatten()

                # QA filter if available
                if "qa_value" in grp.variables:
                    qa = grp.variables["qa_value"][:].data.flatten()
                    vals = np.where(qa < 0.5, np.nan, vals)

                # Fill value / physical range filter for AER_AI
                fv = getattr(grp.variables[var_name], "_FillValue",
                             getattr(grp.variables[var_name], "missing_value", None))
                if fv is not None:
                    vals = np.where(vals == fv, np.nan, vals)
                vals = np.where((vals < -5) | (vals > 20), np.nan, vals)

                # Delhi bbox filter
                mask = (
                    (lat >= LAT_LO) & (lat <= LAT_HI) &
                    (lon >= LON_LO) & (lon <= LON_HI) &
                    np.isfinite(vals)
                )
                if mask.sum() == 0:
                    # No Delhi pixels in this overpass
                    failed += 1
                    continue

                # Extract date from filename
                # S5P_OFFL_L2__AER_AI_20181008T064057_...
                fname = f.stem
                date_str = None
                for part in fname.split("_"):
                    if len(part) == 15 and part[8] == "T":
                        date_str = part[:8]
                        break
                if date_str is None:
                    failed += 1
                    continue

                file_dt = datetime.strptime(date_str, "%Y%m%d")

                frame = pd.DataFrame({
                    "lat":      lat[mask],
                    "lon":      lon[mask],
                    "datetime": file_dt,
                    "aer_ai":   vals[mask].astype(np.float32),
                })
                frames.append(frame)

        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"[S5P] Failed {f.name}: {e}")

    if failed > 0:
        pct = failed / len(nc_files) * 100
        print(f"[S5P] {failed}/{len(nc_files)} files failed or had no Delhi pixels "
              f"({pct:.0f}% — expected if outside orbit coverage).")

    if not frames:
        print("[S5P] No valid S5P data extracted.")
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["datetime"] = pd.to_datetime(out["datetime"])
    print(f"[S5P] Parsed {len(out):,} Delhi pixels across "
          f"{out['datetime'].nunique()} unique dates.")
    valid = out["aer_ai"].dropna()
    print(f"[S5P] AER_AI range: {valid.min():.3f} – {valid.max():.3f}, "
          f"mean={valid.mean():.3f} (n={len(valid):,})")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Spatial nearest-neighbour merge
# ─────────────────────────────────────────────────────────────────────────────

def _nn_merge_daily(
    splits_df: pd.DataFrame,
    sat_df: pd.DataFrame,
    sat_col: str,
    radius_deg: float,
    out_col: str,
) -> pd.Series:
    """
    For each row in splits_df, find the nearest satellite pixel on the same
    date within radius_deg and assign its value. Returns a Series aligned to
    splits_df.index.
    """
    if sat_df.empty:
        return pd.Series(np.nan, index=splits_df.index)

    result = pd.Series(np.nan, index=splits_df.index, dtype=np.float32)

    # Group satellite data by date
    sat_df = sat_df.copy()
    sat_df["_date"] = sat_df["datetime"].dt.normalize()
    splits_df = splits_df.copy()
    splits_df["_date"] = splits_df["timestamp"].dt.normalize()

    unique_dates = splits_df["_date"].unique()
    print(f"  [{out_col}] Matching across {len(unique_dates)} unique dates...")

    matched_dates = 0
    for date in unique_dates:
        sat_day = sat_df[sat_df["_date"] == date]
        if sat_day.empty:
            continue

        row_idx = splits_df.index[splits_df["_date"] == date]
        if len(row_idx) == 0:
            continue

        # Build KDTree from satellite pixels
        tree = cKDTree(sat_day[["lat", "lon"]].values)

        # Query for each split row
        station_locs = splits_df.loc[row_idx, ["latitude", "longitude"]].values
        dists, idxs  = tree.query(station_locs, k=1, workers=-1)

        # Apply radius filter
        valid_mask = dists <= radius_deg
        if valid_mask.any():
            vals = sat_day[sat_col].values[idxs]
            vals[~valid_mask] = np.nan
            result.loc[row_idx] = vals.astype(np.float32)
            matched_dates += 1

    print(f"  [{out_col}] Matched {matched_dates}/{len(unique_dates)} dates with satellite coverage.")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _log1p_minmax_norm(series: pd.Series, q_lo: float = 0.01, q_hi: float = 0.99):
    """log1p transform then min-max scale to [0,1]. NaN → stays NaN."""
    log_s    = np.log1p(series.clip(lower=0))
    vmin     = float(log_s.quantile(q_lo))
    vmax     = float(log_s.quantile(q_hi))
    if vmax - vmin < 1e-8:
        return pd.Series(0.0, index=series.index), vmin, vmax
    normed = ((log_s - vmin) / (vmax - vmin)).clip(0, 1)
    return normed, vmin, vmax


def _linear_minmax_norm(series: pd.Series, q_lo: float = 0.01, q_hi: float = 0.99):
    """Linear min-max scale to [0,1]. NaN → stays NaN."""
    vmin = float(series.quantile(q_lo))
    vmax = float(series.quantile(q_hi))
    if vmax - vmin < 1e-8:
        return pd.Series(0.0, index=series.index), vmin, vmax
    normed = ((series - vmin) / (vmax - vmin)).clip(0, 1)
    return normed, vmin, vmax


# ─────────────────────────────────────────────────────────────────────────────
# Main merge + re-split
# ─────────────────────────────────────────────────────────────────────────────

def merge_and_update_splits(modis_df: pd.DataFrame, s5p_df: pd.DataFrame):
    """
    Load the current merged_hourly.parquet (or train+val+test parquets),
    attach satellite features, renormalise, and write updated split parquets.
    """
    from src.config import PROCESSED_DIR

    merged_path = PROCESSED_DIR / "merged_hourly.parquet"
    if merged_path.exists():
        print(f"[MERGE] Loading full merged dataset from {merged_path.name}...")
        df = pd.read_parquet(merged_path)
    else:
        print("[MERGE] merged_hourly.parquet not found — assembling from split files...")
        parts = []
        for split in ["train", "val", "test_diwali", "test_winter", "test_random"]:
            p = SPLITS_DIR / f"{split}.parquet"
            if p.exists():
                chunk = pd.read_parquet(p)
                chunk["_split"] = split
                parts.append(chunk)
        if not parts:
            raise FileNotFoundError("No split parquets found in data/splits/")
        df = pd.concat(parts, ignore_index=True)

    print(f"[MERGE] Working dataset: {len(df):,} rows, {df.shape[1]} columns.")

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ── Step 1: MODIS AOD spatial merge ──────────────────────────────────────
    if not modis_df.empty:
        print("\n[MODIS] Merging AOD via spatial nearest-neighbour...")
        aod_raw = _nn_merge_daily(
            df, modis_df, sat_col="aod_047" if "aod_047" in modis_df.columns else "aod_055",
            radius_deg=MODIS_RADIUS_DEG, out_col="modis_aod",
        )
        df["modis_aod"] = aod_raw

        # Forward-fill within station (3-day window) to improve coverage
        df = df.sort_values(["station", "timestamp"]).reset_index(drop=True)
        df["modis_aod"] = df.groupby("station")["modis_aod"].transform(
            lambda s: s.ffill(limit=72).bfill(limit=72)
        )

        non_nan_pct = df["modis_aod"].notna().mean() * 100
        print(f"[MODIS] Coverage after forward-fill: {non_nan_pct:.1f}%")

        # Normalise (log1p + min-max)
        norm_series, vmin, vmax = _log1p_minmax_norm(df["modis_aod"].dropna())
        df["modis_aod_norm"] = np.nan
        df.loc[df["modis_aod"].notna(), "modis_aod_norm"] = norm_series.values
        print(f"[MODIS] modis_aod_norm: log1p scale params [{vmin:.4f}, {vmax:.4f}]")
    else:
        print("[MODIS] No MODIS data — setting modis_aod_norm to NaN.")
        df["modis_aod"]      = np.nan
        df["modis_aod_norm"] = np.nan

    # ── Step 2: S5P AER_AI spatial merge ─────────────────────────────────────
    if not s5p_df.empty:
        print("\n[S5P] Merging AER_AI via spatial nearest-neighbour...")
        aer_raw = _nn_merge_daily(
            df, s5p_df, sat_col="aer_ai",
            radius_deg=S5P_RADIUS_DEG, out_col="aer_ai_new",
        )
        df["aer_ai_new"] = aer_raw

        # Forward-fill within station (3-day window)
        df["aer_ai_new"] = df.groupby("station")["aer_ai_new"].transform(
            lambda s: s.ffill(limit=72).bfill(limit=72)
        )

        non_nan_pct = df["aer_ai_new"].notna().mean() * 100
        print(f"[S5P] Coverage after forward-fill: {non_nan_pct:.1f}%")

        # Replace old zero-filled column with properly-merged values
        df["aer_ai"] = df["aer_ai_new"]
        df = df.drop(columns=["aer_ai_new"])

        # Renormalise (linear min-max on physical values)
        norm_series, vmin, vmax = _linear_minmax_norm(df["aer_ai"].dropna())
        df["aer_ai_norm"] = np.nan
        df.loc[df["aer_ai"].notna(), "aer_ai_norm"] = norm_series.values
        print(f"[S5P] aer_ai_norm: linear scale params [{vmin:.4f}, {vmax:.4f}]")
    else:
        print("[S5P] No S5P data — keeping existing aer_ai_norm as-is.")

    # ── Step 3: Coverage summary ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("COVERAGE SUMMARY")
    print("="*60)
    for col in ["modis_aod_norm", "aer_ai_norm", "aer_ai"]:
        if col in df.columns:
            pct = df[col].notna().mean() * 100
            mean_v = df[col].mean()
            print(f"  {col:20s}: {pct:5.1f}% non-NaN, mean={mean_v:.4f}")

    # ── Step 4: Re-create splits ──────────────────────────────────────────────
    print("\n[SPLIT] Re-writing parquet files with updated satellite features...")

    if "_split" in df.columns:
        # Came from assembling splits — write each back
        for split_name in ["train", "val", "test_diwali", "test_winter", "test_random"]:
            mask = df["_split"] == split_name
            if mask.any():
                out_df = df[mask].drop(columns=["_split"]).reset_index(drop=True)
                out_path = SPLITS_DIR / f"{split_name}.parquet"
                out_df.to_parquet(out_path, index=False)
                print(f"  Wrote {split_name}.parquet  ({mask.sum():,} rows)")
        # Also write combined test.parquet
        test_mask = df["_split"].str.startswith("test")
        if test_mask.any():
            test_df = df[test_mask].drop(columns=["_split"]).reset_index(drop=True)
            test_df.to_parquet(SPLITS_DIR / "test.parquet", index=False)
            print(f"  Wrote test.parquet  ({test_mask.sum():,} rows)")
    else:
        # Came from merged_hourly — use same split logic as preprocess.py
        diwali_2018  = (df["timestamp"] >= "2018-11-04") & (df["timestamp"] <= "2018-11-11")
        diwali_2019  = (df["timestamp"] >= "2019-10-24") & (df["timestamp"] <= "2019-10-31")
        diwali_2020  = (df["timestamp"] >= "2020-11-14") & (df["timestamp"] <= "2020-11-21")
        diwali_2021  = (df["timestamp"] >= "2021-11-04") & (df["timestamp"] <= "2021-11-11")
        diwali_mask  = diwali_2018 | diwali_2019 | diwali_2020 | diwali_2021

        winter_2018_19 = (df["timestamp"] >= "2018-12-01") & (df["timestamp"] <= "2019-01-31")
        winter_2019_20 = (df["timestamp"] >= "2019-12-01") & (df["timestamp"] <= "2020-01-31")
        winter_mask    = winter_2018_19 | winter_2019_20

        recent_pool = df["timestamp"].dt.year.isin([2019, 2020]) & ~winter_mask & ~diwali_mask
        random_idx  = df[recent_pool].sample(frac=0.15, random_state=42).index
        random_mask = df.index.isin(random_idx)

        test_mask    = diwali_mask | winter_mask | random_mask
        train_pool   = ~test_mask
        val_idx      = df[train_pool].sample(frac=0.10, random_state=42).index
        val_mask     = df.index.isin(val_idx)
        train_mask   = ~test_mask & ~val_mask

        df[train_mask].to_parquet(SPLITS_DIR / "train.parquet",        index=False)
        df[val_mask  ].to_parquet(SPLITS_DIR / "val.parquet",          index=False)
        df[test_mask ].to_parquet(SPLITS_DIR / "test.parquet",         index=False)
        df[diwali_mask].to_parquet(SPLITS_DIR / "test_diwali.parquet", index=False)
        df[winter_mask].to_parquet(SPLITS_DIR / "test_winter.parquet", index=False)
        df[random_mask].to_parquet(SPLITS_DIR / "test_random.parquet", index=False)

        print(f"  train: {train_mask.sum():,} | val: {val_mask.sum():,} | "
              f"test: {test_mask.sum():,}")

    # ── Step 5: Verify ────────────────────────────────────────────────────────
    print("\n[VERIFY] Checking train.parquet...")
    train = pd.read_parquet(SPLITS_DIR / "train.parquet")
    print(f"  Columns: {train.shape[1]} | Rows: {len(train):,}")

    modis_pct = train["modis_aod_norm"].notna().mean() * 100 if "modis_aod_norm" in train.columns else 0
    aer_pct   = train["aer_ai_norm"].notna().mean() * 100    if "aer_ai_norm" in train.columns else 0

    print(f"  modis_aod_norm non-NaN: {modis_pct:.1f}%  (target: >10%)")
    print(f"  aer_ai_norm    non-NaN: {aer_pct:.1f}%    (target: >20%)")

    if modis_pct >= 10:
        print("  [MODIS] SUCCESS -- target coverage met")
    else:
        print(f"  [MODIS] BELOW TARGET ({modis_pct:.1f}% < 10%)")

    if aer_pct >= 20:
        print("  [S5P]   SUCCESS -- target coverage met")
    else:
        print(f"  [S5P]   BELOW TARGET ({aer_pct:.1f}% < 20%)")

    print("\n[DONE] Satellite merge complete.")
    return train


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  aeras — Satellite Feature Integration")
    print("=" * 60)

    # Step 1: Parse MODIS
    print("\n[PHASE 1] Parsing MODIS AOD files...")
    modis_df = parse_modis_all(MODIS_DIR)

    # Step 2: Parse S5P
    print("\n[PHASE 2] Parsing Sentinel-5P AER_AI files...")
    s5p_df = parse_s5p_all(S5P_DIR)

    # Step 3+4: Merge and update splits
    print("\n[PHASE 3] Merging into splits and saving...")
    merge_and_update_splits(modis_df, s5p_df)
