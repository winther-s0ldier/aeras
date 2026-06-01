import sys
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    RAW_DIR, PROCESSED_DIR, SPLITS_DIR,
    POLLUTANTS, TARGET_POLLUTANT,
    MAX_GAP_HOURS_INTERPOLATE, MIN_MONTHLY_COVERAGE,
    OUTLIER_SIGMA, PM25_MIN, PM25_MAX,
    DELHI_LAT_RANGE, DELHI_LON_RANGE,
)

warnings.filterwarnings("ignore", category=FutureWarning)


DELHI_CPCB_COORDS = {
    "anand vihar":           (28.6471, 77.3159),
    "ashok vihar":           (28.6888, 77.1834),
    "aya nagar":             (28.4744, 77.1100),
    "burari":                (28.7363, 77.1810),
    "civil lines":           (28.6860, 77.2168),
    "karni singh":           (28.5028, 77.0996),
    "dtu":                   (28.7501, 77.1159),
    "dwarka":                (28.5688, 77.0664),
    "igi airport":           (28.5665, 77.0986),
    "ito":                   (28.6283, 77.2423),
    "jahangirpuri":          (28.7299, 77.1672),
    "jawaharlal nehru stad": (28.5798, 77.2364),
    "lodhi road":            (28.5920, 77.2258),
    "mandir marg":           (28.6354, 77.1972),
    "mathura road":          (28.5937, 77.2497),
    "dhyan chand":           (28.6074, 77.2423),
    "mundka":                (28.7035, 77.0577),
    "nsit dwarka":           (28.6113, 77.0340),
    "nehru nagar":           (28.5591, 77.2548),
    "north campus":          (28.7041, 77.1025),
    "okhla":                 (28.5312, 77.2765),
    "patparganj":            (28.6341, 77.2893),
    "pusa":                  (28.6354, 77.1507),
    "punjabi bagh":          (28.6680, 77.1317),
    "rk puram":              (28.5644, 77.1761),
    "rohini":                (28.7340, 77.0987),
    "siri fort":             (28.5500, 77.2167),
    "sirifort":              (28.5500, 77.2167),
    "sri aurobindo":         (28.5481, 77.1861),
    "vivek vihar":           (28.6696, 77.3074),
    "wazirpur":              (28.6988, 77.1613),

    "alipur":                (28.8164, 77.1330),
    "bawana":                (28.7899, 77.0390),
    "east arjun nagar":      (28.6488, 77.2793),
    "ihbas":                 (28.6820, 77.3050),
    "dilshad garden":        (28.6820, 77.3050),
    "najafgarh":             (28.6092, 76.9797),
    "narela":                (28.8522, 77.0938),
    "shadipur":              (28.6438, 77.1497),
    "sonia vihar":           (28.7373, 77.2604),
    "r k puram":             (28.5644, 77.1761),
    "crri":                  (28.5937, 77.2497),

    "delhi ncr aggregate":   (28.6139, 77.2090),
    "delhi":                 (28.6139, 77.2090),

    "faridabad":             (28.4033, 77.3153),
    "gurgaon":               (28.4560, 77.0488),
    "gurugram":              (28.4560, 77.0488),
    "noida sector 62":       (28.6263, 77.3641),
    "noida sector 125":      (28.5412, 77.3404),
    "greater noida":         (28.4756, 77.5036),
    "noida":                 (28.5355, 77.3910),
    "ghaziabad":             (28.6671, 77.3742),
    "ballabhgarh":           (28.3390, 77.3187),
}


def add_station_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    if "latitude" in df.columns and "longitude" in df.columns:

        existing = df["latitude"].notna().sum()
        print(f"[COORDS] Lat/lon already present ({existing} non-null rows).")
        return df

    print("[COORDS] Lat/lon missing — looking up from built-in station table...")

    DELHI_NCR_CENTER = (28.6139, 77.2090)

    def lookup_coords(station_name: str):
        name_lower = str(station_name).lower()
        for key, (lat, lon) in DELHI_CPCB_COORDS.items():
            if key in name_lower:
                return lat, lon
        return DELHI_NCR_CENTER

    if "station" not in df.columns:
        print("[WARN] No 'station' column — assigning Delhi NCR centroid to all rows.")
        df["latitude"]  = DELHI_NCR_CENTER[0]
        df["longitude"] = DELHI_NCR_CENTER[1]
        return df

    coords = df["station"].apply(lookup_coords)
    df["latitude"]  = coords.apply(lambda c: c[0])
    df["longitude"] = coords.apply(lambda c: c[1])

    matched   = df["station"].apply(
        lambda s: any(k in str(s).lower() for k in DELHI_CPCB_COORDS)
    ).sum()
    total     = len(df)
    unmatched = total - matched
    print(f"[COORDS] Matched {matched}/{total} rows to known stations "
          f"({unmatched} rows used centroid fallback).")


    if unmatched > 0:
        unknown = df.loc[
            ~df["station"].apply(
                lambda s: any(k in str(s).lower() for k in DELHI_CPCB_COORDS)
            ), "station"
        ].unique()
        print(f"[COORDS] Unrecognized stations (using centroid): {list(unknown[:10])}")
        if len(unknown) > 10:
            print(f"  ... and {len(unknown) - 10} more")

    return df


def load_cpcb_data(cpcb_dir: Path = None) -> pd.DataFrame:
    if cpcb_dir is None:
        cpcb_dir = RAW_DIR / "cpcb"

    csv_files = list(cpcb_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {cpcb_dir}")

    print(f"[CPCB] Loading {len(csv_files)} CSV files...")
    dfs = []

    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False)

            df.columns = df.columns.str.strip().str.lower()


            time_col_priority = ["from date", "date", "timestamp", "datetime"]
            time_cols = [c for c in time_col_priority if c in df.columns]
            if not time_cols:

                time_cols = [c for c in df.columns if any(
                    k in c for k in ["date", "time", "timestamp"]
                )]
            if time_cols:
                df["timestamp"] = pd.to_datetime(df[time_cols[0]], errors="coerce",
                                                  dayfirst=True)
            else:
                print(f"  [WARN] No timestamp column found in {f.name} — skipping.")
                continue


            station_col_priority = ["station", "station name", "site", "site_name",
                                     "location", "name"]
            station_cols = [c for c in station_col_priority if c in df.columns]
            if station_cols:
                df["station"] = df[station_cols[0]].astype(str).str.strip()
            else:


                raw_stem = f.stem.split(",")[0].strip()


                known_keywords = ["vihar", "nagar", "bagh", "puri", "pur", "road",
                                  "marg", "fort", "lines", "campus", "airport", "stadium"]
                if any(k in raw_stem.lower() for k in known_keywords):
                    station_name = raw_stem
                else:
                    station_name = "Delhi NCR Aggregate"
                df["station"] = station_name

            dfs.append(df)
        except Exception as e:
            print(f"  [WARN] Failed to load {f.name}: {e}")

    if not dfs:
        raise RuntimeError(
            f"[CPCB] All CSVs failed to load from {cpcb_dir}. "
            "Check file format and column names."
        )

    combined = pd.concat(dfs, ignore_index=True)
    print(f"[CPCB] Loaded {len(combined):,} total rows from {len(dfs)} files.")
    print(f"[CPCB] Columns present: {combined.columns.tolist()[:15]}")
    print(f"[CPCB] Stations: {combined['station'].nunique()} unique stations detected.")
    return combined


def clean_cpcb_data(df: pd.DataFrame) -> pd.DataFrame:
    print("[CLEAN] Cleaning CPCB data...")


    pm25_cols = [c for c in df.columns
                 if "pm2.5" in c.lower() or "pm25" in c.lower() or "pm2_5" in c.lower()]
    if pm25_cols:
        df["pm25"] = pd.to_numeric(df[pm25_cols[0]], errors="coerce")
    elif "pm25" not in df.columns:
        print("[WARN] PM2.5 column not found. Available columns:", df.columns.tolist())
        return df

    initial_count = len(df)


    df.loc[df["pm25"] < PM25_MIN, "pm25"] = np.nan
    df.loc[df["pm25"] > PM25_MAX, "pm25"] = np.nan


    if "station" in df.columns and "timestamp" in df.columns:
        df["year_month"] = df["timestamp"].dt.to_period("M")

        def remove_outliers(group):
            mean = group["pm25"].mean()
            std = group["pm25"].std()
            if std > 0:
                mask = (group["pm25"] - mean).abs() > OUTLIER_SIGMA * std
                group.loc[mask, "pm25"] = np.nan
            return group

        df = df.groupby(["station", "year_month"], group_keys=False).apply(remove_outliers)
        df = df.drop(columns=["year_month"])

    removed = initial_count - df["pm25"].notna().sum()
    print(f"[CLEAN] Removed/NaN'd {removed} values ({removed/initial_count*100:.1f}%)")


    if "station" in df.columns and "timestamp" in df.columns:
        df = df.set_index("timestamp").sort_index()
        stations = []
        for name, group in df.groupby("station"):
            hourly = group.resample("1h").mean(numeric_only=True)

            hourly["pm25"] = hourly["pm25"].interpolate(
                method="linear",
                limit=MAX_GAP_HOURS_INTERPOLATE
            )
            hourly["station"] = name
            stations.append(hourly)

        df = pd.concat(stations).reset_index()


    if "station" in df.columns:
        df["year_month"] = df["timestamp"].dt.to_period("M")
        coverage = df.groupby(["station", "year_month"])["pm25"].apply(
            lambda x: x.notna().mean()
        )
        valid_pairs = coverage[coverage >= MIN_MONTHLY_COVERAGE].index
        df = df.set_index(["station", "year_month"])
        df = df.loc[df.index.isin(valid_pairs)].reset_index()
        df = df.drop(columns=["year_month"])
        print(f"[CLEAN] After coverage filter: {df['station'].nunique()} stations remain.")

    return df


def load_era5_data(era5_dir: Path = None) -> "xarray.Dataset":
    import xarray as xr

    if era5_dir is None:
        era5_dir = RAW_DIR / "era5"

    nc_files = sorted(era5_dir.glob("era5_delhi_*.nc"))
    if not nc_files:
        print("[ERA5] No NetCDF files found. Run download_era5.py first.")
        return None

    print(f"[ERA5] Loading {len(nc_files)} files...")
    try:
        ds = xr.open_mfdataset(nc_files, combine="by_coords")
    except ImportError:
        print("[ERA5] dask not available — loading files sequentially...")
        datasets = [xr.open_dataset(f) for f in nc_files]
        ds = xr.concat(datasets, dim="valid_time")
        for d in datasets:
            d.close()


    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})

    print(f"[ERA5] Variables: {list(ds.data_vars)}")
    t0, t1 = ds.time.values[0], ds.time.values[-1]
    print(f"[ERA5] Time range: {t0} to {t1}")


    import pandas as pd
    from src.config import YEARS
    era5_years = set(pd.to_datetime(ds.time.values).year)
    overlap = era5_years & set(YEARS)
    if not overlap:
        print(f"[ERA5] WARNING: ERA5 years {sorted(era5_years)} don't overlap "
              f"CPCB YEARS {YEARS}.")
        print("[ERA5] Merge will produce zero rows. Skipping ERA5.")
        print(f"[ERA5] Re-download ERA5 for years {YEARS} — run `python main.py download`")
        ds.close()
        return None

    return ds


def interpolate_era5_to_stations(
    era5_ds,
    station_coords: pd.DataFrame,
) -> pd.DataFrame:
    print("[ERA5] Interpolating to station locations...")
    records = []

    for _, row in station_coords.iterrows():
        lat, lon = row["latitude"], row["longitude"]
        station = row["station"]


        point = era5_ds.interp(latitude=lat, longitude=lon, method="linear")


        point_df = point.to_dataframe().reset_index()
        point_df["station"] = station
        records.append(point_df)

    result = pd.concat(records, ignore_index=True)
    print(f"[ERA5] Interpolated to {station_coords.shape[0]} stations.")
    return result


def merge_datasets(cpcb_df: pd.DataFrame, era5_df: pd.DataFrame) -> pd.DataFrame:
    print("[MERGE] Merging CPCB + ERA5...")


    cpcb_df["timestamp"] = cpcb_df["timestamp"].dt.floor("h")
    if "time" in era5_df.columns:
        era5_df = era5_df.rename(columns={"time": "timestamp"})
    era5_df["timestamp"] = pd.to_datetime(era5_df["timestamp"]).dt.floor("h")


    rename_map = {
        "u10": "u_wind",
        "v10": "v_wind",
        "t2m": "temperature",
        "blh": "boundary_layer_height",
    }
    era5_df = era5_df.rename(columns={
        k: v for k, v in rename_map.items() if k in era5_df.columns
    })


    if "latitude" in era5_df.columns:
        era5_df = era5_df.drop(columns=["latitude", "longitude"])


    merged = pd.merge(
        cpcb_df, era5_df,
        on=["station", "timestamp"],
        how="inner",
    )

    print(f"[MERGE] Result: {len(merged)} rows, {merged['station'].nunique()} stations.")
    return merged


def assert_no_all_nan(df: pd.DataFrame, new_cols: list, source: str = "merge"):
    for c in new_cols:
        if c not in df.columns:
            raise ValueError(f"[ASSERT/{source}] Expected column '{c}' missing after merge.")
        n_nan = df[c].isna().sum()
        n_total = len(df)
        if n_total == 0:
            raise ValueError(f"[ASSERT/{source}] DataFrame is empty.")
        if n_nan == n_total:
            raise ValueError(f"[ASSERT/{source}] Column '{c}' is 100% NaN — merge silently failed.")
        frac = n_nan / n_total
        if frac > 0.5:
            print(f"[ASSERT/{source}] WARN: '{c}' is {frac*100:.1f}% NaN — high but not all.")
        else:
            print(f"[ASSERT/{source}] OK: '{c}' has {(1-frac)*100:.1f}% coverage.")


def add_firms_features(df: pd.DataFrame, firms_dir: Path = None) -> pd.DataFrame:
    if firms_dir is None:
        firms_dir = RAW_DIR / "firms"

    firms_files = sorted(firms_dir.glob("*.csv"))
    if not firms_files:
        print("[FIRMS] No CSV files found — filling fire features with zeros.")
        df["fire_count_daily"] = 0.0
        df["fire_count_7day"] = 0.0
        return df

    print(f"[FIRMS] Loading {len(firms_files)} files...")
    chunks = []
    for f in firms_files:
        try:
            chunks.append(pd.read_csv(f, low_memory=False))
        except Exception as e:
            print(f"[FIRMS] Failed to load {f.name}: {e}")
    if not chunks:
        df["fire_count_daily"] = 0.0
        df["fire_count_7day"] = 0.0
        return df

    firms = pd.concat(chunks, ignore_index=True)
    if "acq_date" not in firms.columns:
        date_col = next((c for c in firms.columns if "date" in c.lower()), None)
        if date_col is None:
            print("[FIRMS] No date column found — skipping.")
            df["fire_count_daily"] = 0.0
            df["fire_count_7day"] = 0.0
            return df
        firms = firms.rename(columns={date_col: "acq_date"})

    firms["acq_date"] = pd.to_datetime(firms["acq_date"], errors="coerce")
    firms = firms.dropna(subset=["acq_date"])
    print(f"[FIRMS] {len(firms):,} fire detections, "
          f"{firms['acq_date'].min().date()} to {firms['acq_date'].max().date()}")

    daily = firms.groupby(firms["acq_date"].dt.normalize()).size().rename("fire_count_daily")
    daily.index = pd.to_datetime(daily.index)

    t_min = df["timestamp"].min().normalize()
    t_max = df["timestamp"].max().normalize()
    full_idx = pd.date_range(t_min, t_max, freq="1D")
    daily_full = daily.reindex(full_idx, fill_value=0)
    rolling_7d = daily_full.rolling(7, min_periods=1).sum().rename("fire_count_7day")

    fire_df = pd.DataFrame({
        "fire_count_daily": daily_full,
        "fire_count_7day": rolling_7d,
    })
    fire_df.index.name = "_date"
    fire_df = fire_df.reset_index()

    df["_date"] = df["timestamp"].dt.normalize()
    df = df.merge(fire_df, on="_date", how="left").drop(columns=["_date"])
    df["fire_count_daily"] = df["fire_count_daily"].fillna(0)
    df["fire_count_7day"] = df["fire_count_7day"].fillna(0)

    winter_mask = df["timestamp"].dt.month.isin([10, 11, 12, 1])
    mean_winter = df.loc[winter_mask, "fire_count_daily"].mean()
    mean_other = df.loc[~winter_mask, "fire_count_daily"].mean()
    print(f"[FIRMS] Daily fire count — winter mean: {mean_winter:.1f} | non-winter mean: {mean_other:.1f}")

    for col in ["fire_count_daily", "fire_count_7day"]:
        cmin = float(df[col].quantile(0.01))
        cmax = float(df[col].quantile(0.99))
        if cmax - cmin < 1e-8:
            df[f"{col}_norm"] = 0.0
        else:
            df[f"{col}_norm"] = ((df[col] - cmin) / (cmax - cmin)).clip(0, 1).fillna(0)
        print(f"[FIRMS] Normalized {col} -> {col}_norm, range [{cmin:.0f}, {cmax:.0f}]")

    assert_no_all_nan(df, ["fire_count_daily_norm", "fire_count_7day_norm"], source="FIRMS")
    return df


def add_modis_aod(df: pd.DataFrame, modis_dir: Path = None, station_coords: pd.DataFrame = None) -> pd.DataFrame:
    if modis_dir is None:
        modis_dir = RAW_DIR / "modis"

    try:
        from pyhdf.SD import SD, SDC
    except ImportError:
        print("[MODIS] pyhdf not installed — skipping AOD merge. (Install with: pip install pyhdf)")
        df["aod_norm"] = 0.0
        return df

    hdf_files = sorted(modis_dir.rglob("*.hdf"))
    if not hdf_files:
        print("[MODIS] No HDF files found — skipping AOD merge.")
        df["aod_norm"] = 0.0
        return df

    print(f"[MODIS] Processing {len(hdf_files)} HDF files...")

    records = []
    failed = 0
    for f in hdf_files:
        try:
            stem = f.stem
            parts = stem.split(".")
            year_doy = next((p for p in parts if p.startswith("A") and len(p) == 8), None)
            if year_doy is None:
                failed += 1
                continue
            year = int(year_doy[1:5])
            doy = int(year_doy[5:8])
            date = datetime(year, 1, 1) + timedelta(days=doy - 1)

            hdf = SD(str(f), SDC.READ)
            aod_var_candidates = ["Optical_Depth_055", "AOD_550_Dark_Target_Deep_Blue_Combined",
                                  "Optical_Depth_Land_And_Ocean"]
            aod_data = None
            for vn in aod_var_candidates:
                try:
                    obj = hdf.select(vn)
                    aod_data = obj[:].astype(np.float64)
                    attrs = obj.attributes()
                    obj.endaccess()
                    scale = attrs.get("scale_factor", 0.001)
                    fill = attrs.get("_FillValue", -28672)
                    aod_data = np.where(aod_data == fill, np.nan, aod_data * scale)
                    break
                except Exception:
                    continue
            hdf.end()

            if aod_data is None:
                failed += 1
                continue

            mean_aod = float(np.nanmean(aod_data))
            if np.isfinite(mean_aod):
                records.append({"date": date, "aod": mean_aod})
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"[MODIS] Failed {f.name}: {e}")

    if failed > 0:
        print(f"[MODIS] {failed}/{len(hdf_files)} files failed to parse.")

    if not records:
        print("[MODIS] No valid AOD values extracted — skipping.")
        df["aod_norm"] = 0.0
        return df

    modis_df = pd.DataFrame(records)
    daily_aod = modis_df.groupby("date")["aod"].mean().reset_index()
    daily_aod["date"] = pd.to_datetime(daily_aod["date"])
    print(f"[MODIS] {len(daily_aod):,} daily AOD values, mean={daily_aod['aod'].mean():.3f}")

    df["_date"] = df["timestamp"].dt.normalize()
    df = df.merge(daily_aod, left_on="_date", right_on="date", how="left").drop(columns=["_date", "date"])
    df = df.sort_values(["station", "timestamp"]).reset_index(drop=True)
    df["aod"] = df.groupby("station")["aod"].transform(lambda s: s.ffill(limit=72).bfill(limit=72))

    aod_min = float(df["aod"].quantile(0.01))
    aod_max = float(df["aod"].quantile(0.99))
    if aod_max - aod_min < 1e-8:
        df["aod_norm"] = 0.0
    else:
        df["aod_norm"] = ((df["aod"] - aod_min) / (aod_max - aod_min)).clip(0, 1).fillna(0)

    coverage = df["aod"].notna().mean() * 100
    print(f"[MODIS] AOD merged. Coverage: {coverage:.1f}%, range [{aod_min:.3f}, {aod_max:.3f}]")

    assert_no_all_nan(df, ["aod_norm"], source="MODIS")
    return df


def add_sentinel5p(df: pd.DataFrame, s5p_dir: Path = None, station_coords: pd.DataFrame = None) -> pd.DataFrame:
    if s5p_dir is None:
        s5p_dir = RAW_DIR / "s5p"

    nc_files = sorted(s5p_dir.rglob("*.nc"))
    if not nc_files:
        print("[S5P] No NetCDF files found — skipping.")
        df["s5p_norm"] = 0.0
        return df

    import xarray as xr

    sample = nc_files[0].name.upper()
    product_map = {
        "AER_AI": (["absorbing_aerosol_index", "aerosol_index_354_388"], "aer_ai"),
        "NO2___": (["nitrogendioxide_tropospheric_column"], "no2"),
        "SO2___": (["sulfurdioxide_total_vertical_column"], "so2"),
        "CO____": (["carbonmonoxide_total_column"], "co"),
        "HCHO__": (["formaldehyde_tropospheric_vertical_column"], "hcho"),
    }
    var_names, out_short = None, None
    for token, (vns, short) in product_map.items():
        if token in sample:
            var_names, out_short = vns, short
            break
    if var_names is None:
        print(f"[S5P] Unknown product type in {sample} — skipping.")
        df["s5p_norm"] = 0.0
        return df

    print(f"[S5P] Detected product: {out_short.upper()} | Processing {len(nc_files)} files...")

    lat_lo, lat_hi = DELHI_LAT_RANGE
    lon_lo, lon_hi = DELHI_LON_RANGE
    records = []
    failed = 0
    for f in nc_files:
        try:
            ds = xr.open_dataset(f, group="PRODUCT")
            var_name = next((vn for vn in var_names if vn in ds.data_vars), None)
            if var_name is None:
                failed += 1
                ds.close()
                continue
            lat = ds["latitude"].values.flatten()
            lon = ds["longitude"].values.flatten()
            vals = ds[var_name].values.flatten()
            ts_arr = ds["time"].values
            time_val = pd.to_datetime(ts_arr[0]) if len(ts_arr) > 0 else None
            ds.close()

            if time_val is None:
                failed += 1
                continue

            mask = (
                (lat >= lat_lo) & (lat <= lat_hi) &
                (lon >= lon_lo) & (lon <= lon_hi) &
                np.isfinite(vals)
            )
            if mask.sum() > 0:
                mean_val = float(np.nanmean(vals[mask]))
                records.append({"timestamp": time_val, "value": mean_val})
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"[S5P] Failed {f.name}: {e}")

    if failed > 0:
        print(f"[S5P] {failed}/{len(nc_files)} files failed to parse.")

    out_col_raw = out_short
    out_col_norm = f"{out_short}_norm"

    if not records:
        print("[S5P] No valid values extracted — filling with zeros.")
        df[out_col_norm] = 0.0
        return df

    s5p_df = pd.DataFrame(records).sort_values("timestamp")
    print(f"[S5P] {len(s5p_df)} overpass values, "
          f"{s5p_df['timestamp'].min()} to {s5p_df['timestamp'].max()}")

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = df["timestamp"].astype("datetime64[us]")
    s5p_df["timestamp"] = s5p_df["timestamp"].astype("datetime64[us]")


    df = pd.merge_asof(
        df, s5p_df.rename(columns={"value": out_col_raw}),
        on="timestamp", direction="nearest",
        tolerance=pd.Timedelta("3D"),
    )

    val_min = float(df[out_col_raw].quantile(0.01))
    val_max = float(df[out_col_raw].quantile(0.99))
    if val_max - val_min < 1e-12:
        df[out_col_norm] = 0.0
    else:
        df[out_col_norm] = ((df[out_col_raw] - val_min) / (val_max - val_min)).clip(0, 1).fillna(0)

    coverage = df[out_col_raw].notna().mean() * 100
    print(f"[S5P] {out_short.upper()} merged. Coverage: {coverage:.1f}%, range [{val_min:.3e}, {val_max:.3e}]")

    assert_no_all_nan(df, [out_col_norm], source="S5P")
    return df


def normalize_and_split(df: pd.DataFrame, station_coords: pd.DataFrame):
    print("[NORM] Normalizing features...")


    norm_cols = ["pm25", "no2", "o3", "so2", "u_wind", "v_wind", "temperature", "boundary_layer_height",
                 "fire_count_daily", "fire_count_7day"]
    norm_cols = [c for c in norm_cols if c in df.columns]


    norm_period_mask = df["timestamp"].dt.year.isin([2015, 2016, 2017, 2018])
    norm_params = {}
    for col in norm_cols:
        col_min = df.loc[norm_period_mask, col].min()
        col_max = df.loc[norm_period_mask, col].max()
        norm_params[col] = {"min": float(col_min), "max": float(col_max)}
        df[f"{col}_norm"] = (df[col] - col_min) / (col_max - col_min + 1e-8)

    if "temperature_norm" in df.columns:
        df["temp_norm"] = df["temperature_norm"]
    if "boundary_layer_height_norm" in df.columns:
        df["blh_norm"] = df["boundary_layer_height_norm"]


    if "latitude" in df.columns:
        lat_min, lat_max = DELHI_LAT_RANGE
        lon_min, lon_max = DELHI_LON_RANGE
        df["x_norm"] = (df["longitude"] - lon_min) / (lon_max - lon_min)
        df["y_norm"] = (df["latitude"] - lat_min) / (lat_max - lat_min)
        norm_params["latitude"] = {"min": lat_min, "max": lat_max}
        norm_params["longitude"] = {"min": lon_min, "max": lon_max}


    t_min = df["timestamp"].min().timestamp()
    t_max = df["timestamp"].max().timestamp()
    df["t_norm"] = (df["timestamp"].apply(lambda x: x.timestamp()) - t_min) / (t_max - t_min)
    norm_params["timestamp"] = {"min": t_min, "max": t_max}

    df["hour_sin"]  = np.sin(2 * np.pi * df["timestamp"].dt.hour / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["timestamp"].dt.hour / 24)
    df["doy_sin"]   = np.sin(2 * np.pi * df["timestamp"].dt.dayofyear / 365)
    df["doy_cos"]   = np.cos(2 * np.pi * df["timestamp"].dt.dayofyear / 365)

    from src.config import INDIAN_HOLIDAYS
    df["is_holiday"] = df["timestamp"].dt.strftime("%Y-%m-%d").isin(INDIAN_HOLIDAYS).astype(float)


    norm_path = PROCESSED_DIR / "normalized_params.json"
    with open(norm_path, "w") as f:
        json.dump(norm_params, f, indent=2)
    print(f"[NORM] Saved normalization params to {norm_path}")


    print("[SPLIT] Creating train/val/test splits...")


    diwali_2018 = (df["timestamp"] >= "2018-11-04") & (df["timestamp"] <= "2018-11-11")
    diwali_2019 = (df["timestamp"] >= "2019-10-24") & (df["timestamp"] <= "2019-10-31")
    diwali_mask = diwali_2018 | diwali_2019


    winter_2018_19 = (df["timestamp"] >= "2018-12-01") & (df["timestamp"] <= "2019-01-31")
    winter_2019_20 = (df["timestamp"] >= "2019-12-01") & (df["timestamp"] <= "2020-01-31")
    winter_mask = winter_2018_19 | winter_2019_20


    recent_pool = (df["timestamp"].dt.year.isin([2019, 2020])) & ~winter_mask & ~diwali_mask
    random_test_idx = df[recent_pool].sample(frac=0.15, random_state=42).index
    random_test_mask = df.index.isin(random_test_idx)


    test_mask = diwali_mask | winter_mask | random_test_mask


    train_pool = ~test_mask
    val_idx = df[train_pool].sample(frac=0.10, random_state=42).index
    val_mask = df.index.isin(val_idx)


    train_split_mask = ~test_mask & ~val_mask


    df[train_split_mask].to_parquet(SPLITS_DIR / "train.parquet", index=False)
    df[val_mask].to_parquet(SPLITS_DIR / "val.parquet", index=False)
    df[test_mask].to_parquet(SPLITS_DIR / "test.parquet", index=False)


    df[diwali_mask].to_parquet(SPLITS_DIR / "test_diwali.parquet", index=False)
    df[winter_mask].to_parquet(SPLITS_DIR / "test_winter.parquet", index=False)
    df[random_test_mask].to_parquet(SPLITS_DIR / "test_random.parquet", index=False)

    print(f"[SPLIT] Train: {train_split_mask.sum():,} | Val: {val_mask.sum():,} | Test: {test_mask.sum():,}")
    print(f"  Test breakdown — Diwali: {diwali_mask.sum():,} | Winter: {winter_mask.sum():,} | Random: {random_test_mask.sum():,}")

    return df


def generate_collocation_points(
    n_points: int = 50_000,
    save: bool = True,
) -> np.ndarray:
    print(f"[COLLOC] Generating {n_points:,} collocation points...")

    rng = np.random.default_rng(seed=42)
    points = rng.uniform(0, 1, size=(n_points, 3))

    if save:
        path = PROCESSED_DIR / "collocation_points.npy"
        np.save(path, points)
        print(f"[COLLOC] Saved to {path}")

    return points


def load_station_hour_data(spatial_dir: Path = None) -> pd.DataFrame:
    if spatial_dir is None:
        spatial_dir = RAW_DIR / "cpcb" / "spatial"

    sh_path = spatial_dir / "station_hour.csv"
    if not sh_path.exists():
        print("[SH] station_hour.csv not found.")
        return None

    print(f"[SH] Loading {sh_path.name} ({sh_path.stat().st_size // (1024*1024):.0f} MB)...")
    sh = pd.read_csv(sh_path, low_memory=False,
                     usecols=["StationId", "Datetime", "PM2.5", "NO2", "O3", "SO2"])
    sh.columns = ["station_id", "timestamp", "pm25", "no2", "o3", "so2"]
    sh["timestamp"] = pd.to_datetime(sh["timestamp"], errors="coerce")
    sh["pm25"] = pd.to_numeric(sh["pm25"], errors="coerce")
    sh["no2"] = pd.to_numeric(sh["no2"], errors="coerce")
    sh["o3"] = pd.to_numeric(sh["o3"], errors="coerce")
    sh["so2"] = pd.to_numeric(sh["so2"], errors="coerce")


    sh = sh[sh["station_id"].str.startswith("DL", na=False)].copy()


    st_path = spatial_dir / "stations.csv"
    if st_path.exists():
        st_meta = pd.read_csv(st_path, usecols=["StationId", "StationName"])
        st_meta.columns = ["station_id", "station_name"]
        sh = sh.merge(st_meta, on="station_id", how="left")

        sh["station"] = sh["station_name"].str.split(",").str[0].str.strip()
    else:
        sh["station"] = sh["station_id"]


    from src.config import YEARS
    sh = sh[sh["timestamp"].dt.year.isin(YEARS)].copy()

    sh = add_station_coordinates(sh)


    from src.config import DELHI_LAT_RANGE, DELHI_LON_RANGE
    lat_min, lat_max = DELHI_LAT_RANGE
    lon_min, lon_max = DELHI_LON_RANGE
    bbox_mask = (
        sh["latitude"].between(lat_min, lat_max) &
        sh["longitude"].between(lon_min, lon_max)
    )
    outside = (~bbox_mask).sum()
    if outside > 0:
        print(f"[SH] Dropped {outside:,} rows outside Delhi NCR bounding box.")
    sh = sh[bbox_mask]

    print(f"[SH] {len(sh):,} rows | {sh['station'].nunique()} stations | "
          f"{sh['timestamp'].min().date()} to {sh['timestamp'].max().date()}")
    print(f"[SH] PM2.5 available: {sh['pm25'].notna().mean()*100:.1f}%")

    return sh[["timestamp", "station", "pm25", "no2", "o3", "so2", "latitude", "longitude"]]


def load_openaq_extension() -> pd.DataFrame:
    """
    Load OpenAQ data for 2020-07-02 onwards — the period not covered by the
    Kaggle station_hour.csv. Merged with Kaggle data in run_pipeline() to
    give 4 Diwali events in training (2018, 2019, 2020, 2021).

    Schema matches load_station_hour_data() output:
    timestamp, station, pm25, no2, o3, so2, latitude, longitude
    """
    openaq_dir = RAW_DIR / "cpcb" / "openaq"
    cutoff = pd.Timestamp("2020-07-02", tz="UTC")

    frames = []
    for year in [2020, 2021, 2022]:
        path = openaq_dir / f"openaq_{year}.parquet"
        if not path.exists():
            print(f"[OPENAQ-EXT] {path.name} not found, skipping.")
            continue
        df = pd.read_parquet(path)
        # Only keep rows beyond the Kaggle cutoff
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        df = df[df["timestamp"] >= cutoff].copy()
        frames.append(df)
        print(f"[OPENAQ-EXT] {year}: {len(df):,} rows after cutoff filter")

    if not frames:
        print("[OPENAQ-EXT] No OpenAQ extension data found.")
        return None

    ext = pd.concat(frames, ignore_index=True)
    # Strip timezone for consistency with Kaggle data
    ext["timestamp"] = ext["timestamp"].dt.tz_localize(None)

    # Apply same bbox filter as Kaggle data
    from src.config import DELHI_LAT_RANGE, DELHI_LON_RANGE
    lat_min, lat_max = DELHI_LAT_RANGE
    lon_min, lon_max = DELHI_LON_RANGE
    ext = ext[
        ext["latitude"].between(lat_min, lat_max) &
        ext["longitude"].between(lon_min, lon_max)
    ].copy()

    print(f"[OPENAQ-EXT] Total: {len(ext):,} rows | "
          f"{ext['station'].nunique()} stations | "
          f"PM2.5 coverage: {ext['pm25'].notna().mean()*100:.1f}%")

    return ext[["timestamp", "station", "pm25", "no2", "o3", "so2",
                "latitude", "longitude"]]


NCR_CITY_COORDS = {
    "Delhi":         (28.6139, 77.2090),
    "Faridabad":     (28.4033, 77.3153),
    "Gurgaon":       (28.4560, 77.0488),
    "Gurugram":      (28.4560, 77.0488),
    "Noida":         (28.5355, 77.3910),
    "Greater Noida": (28.4756, 77.5036),
    "Ghaziabad":     (28.6671, 77.3742),
    "Baghpat":       (28.9443, 77.2153),
    "Meerut":        (28.9845, 77.7064),
    "Sonipat":       (28.9931, 77.0151),
    "Bhiwani":       (28.7975, 76.1330),
}


NCR_BBOX_CITIES = {
    city: coords for city, coords in NCR_CITY_COORDS.items()
    if (27.5 <= coords[0] <= 29.0) and (76.5 <= coords[1] <= 77.8)
}


def load_spatial_cpcb_data(spatial_dir: Path = None) -> pd.DataFrame:
    if spatial_dir is None:
        spatial_dir = RAW_DIR / "cpcb" / "spatial"

    city_day_path = next(spatial_dir.rglob("city_day.csv"), None)
    if city_day_path is None:
        print("[SPATIAL] city_day.csv not found — spatial supplement unavailable.")
        print(f"  Expected in: {spatial_dir}")
        return None

    print(f"[SPATIAL] Loading {city_day_path.name}...")
    df = pd.read_csv(city_day_path, low_memory=False)
    df.columns = df.columns.str.strip()


    ncr_cities = list(NCR_BBOX_CITIES.keys())
    df = df[df["City"].isin(ncr_cities)].copy()
    print(f"[SPATIAL] NCR cities found: {sorted(df['City'].unique())}")


    df["timestamp"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["timestamp"])


    pm25_cols = [c for c in df.columns
                 if "pm2.5" in c.lower() or "pm25" in c.lower() or "pm2_5" in c.lower()]
    if not pm25_cols:
        print("[SPATIAL] PM2.5 column not found in city_day.csv")
        return None
    df["pm25"] = pd.to_numeric(df[pm25_cols[0]], errors="coerce")
    df["station"] = df["City"]


    df["latitude"]  = df["City"].map(lambda c: NCR_BBOX_CITIES.get(c, (np.nan, np.nan))[0])
    df["longitude"] = df["City"].map(lambda c: NCR_BBOX_CITIES.get(c, (np.nan, np.nan))[1])


    print("[SPATIAL] Upsampling daily → hourly (linear interpolation)...")
    hourly_dfs = []
    for city, group in df.groupby("City"):
        group = group.set_index("timestamp").sort_index()

        hourly = group.resample("1h").asfreq()
        hourly["pm25"] = hourly["pm25"].interpolate(method="linear", limit=48)

        for col in ["station", "latitude", "longitude"]:
            hourly[col] = hourly[col].ffill().bfill()
        hourly_dfs.append(hourly.reset_index())

    spatial_hourly = pd.concat(hourly_dfs, ignore_index=True)

    spatial_hourly = spatial_hourly.dropna(subset=["pm25"])

    print(f"[SPATIAL] {len(spatial_hourly):,} hourly rows across "
          f"{spatial_hourly['station'].nunique()} NCR cities.")
    return spatial_hourly


def run_pipeline():

    print("  aeras — Data Preprocessing Pipeline")

    sh_df = load_station_hour_data()
    if sh_df is not None and len(sh_df) > 10000:
        print(f"\n[PIPELINE] Using station_hour.csv as primary source "
              f"({sh_df['station'].nunique()} stations).")
        cpcb_df = sh_df.copy()

        # Extend with OpenAQ data (2020-07-02 to 2022-12-31).
        # Adds Diwali 2020 and 2021 to training data.
        openaq_ext = load_openaq_extension()
        if openaq_ext is not None and len(openaq_ext) > 1000:
            cpcb_df = pd.concat([cpcb_df, openaq_ext], ignore_index=True)
            cpcb_df = cpcb_df.sort_values(["timestamp", "station"]).reset_index(drop=True)
            print(f"[PIPELINE] After OpenAQ merge: {len(cpcb_df):,} rows | "
                  f"{cpcb_df['station'].nunique()} stations | "
                  f"{cpcb_df['timestamp'].min().date()} to "
                  f"{cpcb_df['timestamp'].max().date()}")

        def flag_outliers(grp):
            mean, std = grp.mean(), grp.std()
            if std > 0:
                grp[(grp - mean).abs() > OUTLIER_SIGMA * std] = np.nan
            return grp

        POLLUTANT_BOUNDS = {
            "pm25": (PM25_MIN, PM25_MAX),
            "no2":  (0.0, 500.0),
            "o3":   (0.0, 500.0),
            "so2":  (0.0, 500.0),
        }
        for poll, (lo, hi) in POLLUTANT_BOUNDS.items():
            if poll in cpcb_df.columns:
                cpcb_df[poll] = cpcb_df[poll].clip(lower=lo, upper=hi)
                cpcb_df[poll] = cpcb_df.groupby("station")[poll].transform(flag_outliers)
    else:

        print("\n[PIPELINE] station_hour.csv not available — falling back to delhi_aqi.csv.")
        print("[WARN] Single-station mode: PINN cannot learn spatial gradients.")
        cpcb_df = load_cpcb_data()
        cpcb_df = clean_cpcb_data(cpcb_df)
        cpcb_df = add_station_coordinates(cpcb_df)


    cpcb_df = add_station_coordinates(cpcb_df)


    if all(c in cpcb_df.columns for c in ["station", "latitude", "longitude"]):
        station_coords = (
            cpcb_df.groupby("station")[["latitude", "longitude"]]
            .first().reset_index()
        )
        station_coords.to_csv(PROCESSED_DIR / "station_coords.csv", index=False)
        print(f"[COORDS] Saved {len(station_coords)} station coordinates.")
    else:
        print("[WARN] Lat/lon not found in CPCB data. Station coords must be added manually.")
        station_coords = None


    era5_ds = load_era5_data()
    if era5_ds is not None and station_coords is not None:
        era5_df = interpolate_era5_to_stations(era5_ds, station_coords)


        merged = merge_datasets(cpcb_df, era5_df)
    else:
        print("[WARN] Skipping ERA5 merge — using CPCB data only for now.")
        merged = cpcb_df


    print("\n[PHASE 3] External data merges starting...")
    merged = add_firms_features(merged)
    merged = add_modis_aod(merged, station_coords=station_coords)
    merged = add_sentinel5p(merged, station_coords=station_coords)
    print("[PHASE 3] External data merges complete.\n")


    merged.to_parquet(PROCESSED_DIR / "merged_hourly.parquet", index=False)
    print(f"[SAVE] Merged dataset: {PROCESSED_DIR / 'merged_hourly.parquet'}")


    normalize_and_split(merged, station_coords)


    generate_collocation_points()


    print("  Pipeline complete!")

if __name__ == "__main__":
    run_pipeline()
