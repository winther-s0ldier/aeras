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


if __name__ == "__main__":
    from src.config import SPLITS_DIR
    import json


    print("  XGBoost Baseline Training")

    print("[XGB] Loading data...")
    train_df = pd.read_parquet(SPLITS_DIR / "train.parquet")
    val_df = pd.read_parquet(SPLITS_DIR / "val.parquet")

    print("[XGB] Training models for multiple horizons...")
    models, metrics = train_xgboost(train_df, val_df)

    print("\n[XGB] Training complete! Metrics on validation set:")
    for horizon, m in metrics.items():
        print(f"  {horizon}h: MAE={m['mae']:.2f}, RMSE={m['rmse']:.2f}")

    with open("checkpoints/xgboost_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("[XGB] Saved metrics to checkpoints/xgboost_metrics.json")
