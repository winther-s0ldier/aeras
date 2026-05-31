import numpy as np
import pandas as pd
from typing import Tuple, Dict
import json
from pathlib import Path


def create_features(df: pd.DataFrame, target_col: str = "pm25") -> pd.DataFrame:
    features = df.copy()


    for lag in [1, 3, 6, 12, 24, 48]:
        features[f"pm25_lag_{lag}"] = features[target_col].shift(lag)


    for window in [6, 12, 24]:
        features[f"pm25_rolling_mean_{window}"] = (
            features[target_col].rolling(window, min_periods=1).mean()
        )
        features[f"pm25_rolling_std_{window}"] = (
            features[target_col].rolling(window, min_periods=1).std()
        )
        features[f"pm25_rolling_max_{window}"] = (
            features[target_col].rolling(window, min_periods=1).max()
        )


    if "timestamp" in features.columns:
        ts = pd.to_datetime(features["timestamp"])
        features["hour"] = ts.dt.hour
        features["day_of_week"] = ts.dt.dayofweek
        features["month"] = ts.dt.month
        features["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)


        features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
        features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
        features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
        features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)


    if "u_wind" in features.columns and "v_wind" in features.columns:
        features["wind_speed"] = np.sqrt(features["u_wind"]**2 + features["v_wind"]**2)
        features["wind_dir"] = np.arctan2(features["v_wind"], features["u_wind"])


    features = features.dropna()

    return features


def train_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str = "pm25",
    forecast_horizons: list = None,
) -> Tuple[dict, dict]:
    try:
        import xgboost as xgb
    except ImportError:
        print("[ERROR] xgboost not installed. Run: pip install xgboost")
        return {}, {}

    if forecast_horizons is None:
        forecast_horizons = [1, 6, 12, 24]


    train_feat = create_features(train_df, target_col)
    val_feat = create_features(val_df, target_col)


    exclude = [target_col, "timestamp", "station", "latitude", "longitude"]
    feature_cols = [c for c in train_feat.columns if c not in exclude and "_norm" not in c]

    models = {}
    metrics = {}

    for horizon in forecast_horizons:
        print(f"\n[XGB] Training for {horizon}h forecast horizon...")


        train_feat[f"target_{horizon}h"] = train_feat[target_col].shift(-horizon)
        val_feat[f"target_{horizon}h"] = val_feat[target_col].shift(-horizon)


        train_valid = train_feat.dropna(subset=[f"target_{horizon}h"])
        val_valid = val_feat.dropna(subset=[f"target_{horizon}h"])

        X_train = train_valid[feature_cols].values
        y_train = train_valid[f"target_{horizon}h"].values
        X_val = val_valid[feature_cols].values
        y_val = val_valid[f"target_{horizon}h"].values

        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=20,
            eval_metric="mae",
            tree_method="hist",
            verbosity=1,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )


        val_pred = model.predict(X_val)
        mae = np.mean(np.abs(val_pred - y_val))
        rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))

        print(f"[XGB] {horizon}h — MAE: {mae:.2f} | RMSE: {rmse:.2f}")

        models[horizon] = model
        metrics[horizon] = {"mae": float(mae), "rmse": float(rmse)}


        train_feat.drop(columns=[f"target_{horizon}h"], inplace=True)
        val_feat.drop(columns=[f"target_{horizon}h"], inplace=True)

    return models, metrics


def _per_station_features(df: pd.DataFrame, target_col: str = "pm25_norm") -> pd.DataFrame:
    df = df.sort_values(["station", "timestamp"]).copy()
    parts = []
    for _, grp in df.groupby("station", sort=False):
        parts.append(create_features(grp, target_col))
    return pd.concat(parts, ignore_index=True)


def _eval_test_sets(models: dict, norm_params: dict, splits_dir: Path) -> dict:
    results = {}
    pm25_min = norm_params.get("pm25", {}).get("min", 0.0)
    pm25_max = norm_params.get("pm25", {}).get("max", 500.0)

    for split_name, split_file in [
        ("random",  "test_random"),
        ("diwali",  "test_diwali"),
        ("winter",  "test_winter"),
    ]:
        path = splits_dir / f"{split_file}.parquet"
        if not path.exists():
            print(f"[XGB] {split_file}.parquet not found, skipping.")
            continue

        df_test = pd.read_parquet(path)
        test_feat = _per_station_features(df_test, "pm25_norm")

        exclude = ["pm25_norm", "timestamp", "station", "latitude", "longitude"]
        feature_cols = [c for c in test_feat.columns if c not in exclude and "_norm" not in c]

        model_1h = models.get(1)
        if model_1h is None:
            continue

        test_feat["target_1h"] = test_feat["pm25_norm"].shift(-1)
        test_valid = test_feat.dropna(subset=["target_1h"])
        avail_cols = [c for c in feature_cols if c in test_valid.columns]

        pred_norm = model_1h.predict(test_valid[avail_cols].values)
        true_norm = test_valid["target_1h"].values

        pred = pred_norm * (pm25_max - pm25_min) + pm25_min
        true = true_norm * (pm25_max - pm25_min) + pm25_min

        mae  = float(np.mean(np.abs(pred - true)))
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - true.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

        results[split_name] = {"mae": mae, "rmse": rmse, "r2": r2, "n": len(test_valid)}
        print(f"[XGB] {split_name:10s} — MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f} (n={len(test_valid):,})")

    return results


if __name__ == "__main__":
    import json
    from src.config import SPLITS_DIR, PROCESSED_DIR

    print("  XGBoost Baseline Training")

    print("[XGB] Loading data...")
    train_df = pd.read_parquet(SPLITS_DIR / "train.parquet")
    val_df   = pd.read_parquet(SPLITS_DIR / "val.parquet")

    print("[XGB] Building per-station lag features...")
    train_feat = _per_station_features(train_df, "pm25_norm")
    val_feat   = _per_station_features(val_df,   "pm25_norm")

    print("[XGB] Training models for multiple horizons...")
    models, metrics = train_xgboost(train_feat, val_feat, target_col="pm25_norm")

    print("\n[XGB] Val set metrics (normalised):")
    for horizon, m in metrics.items():
        print(f"  {horizon}h: MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}")

    norm_path = PROCESSED_DIR / "normalized_params.json"
    norm_params = json.loads(norm_path.read_text()) if norm_path.exists() else {}

    print("\n[XGB] OOD test set evaluation (denormalised, PM2.5 in ug/m3, 1h horizon):")
    test_results = _eval_test_sets(models, norm_params, SPLITS_DIR)

    all_results = {"val_normalised": metrics, "test_denormalised_1h": test_results}
    out_path = Path("checkpoints/xgboost_metrics.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[XGB] Results saved to {out_path}")
