import sys
import argparse


def cmd_download():
    from src.data.download_cpcb import main as download_cpcb
    from src.data.download_era5 import main as download_era5
    print("\n>>> Downloading CPCB data...")
    download_cpcb()
    print("\n>>> Downloading ERA5 data...")
    download_era5()


def cmd_preprocess():
    from src.data.preprocess import run_pipeline
    run_pipeline()


def cmd_train(inverse: bool = False):
    import torch
    import pandas as pd
    from src.config import SPLITS_DIR, DEVICE
    from src.training.trainer import AerasTrainer


    print("[MAIN] Loading training data...")
    train_df = pd.read_parquet(SPLITS_DIR / "train.parquet")
    val_df = pd.read_parquet(SPLITS_DIR / "val.parquet")


    input_cols = ["x_norm", "y_norm", "t_norm",
                  "u_wind_norm", "v_wind_norm",
                  "temp_norm", "blh_norm",
                  "hour_sin", "hour_cos",
                  "doy_sin", "doy_cos",
                  "is_holiday"]
    target_col = "pm25_norm"

    available = [c for c in input_cols if c in train_df.columns]
    missing_era5 = [c for c in ["u_wind_norm", "v_wind_norm", "temp_norm", "blh_norm"] if c not in train_df.columns]


    if missing_era5:
        print(f"[WARN] ERA5 columns not found: {missing_era5}")
        print("[WARN] ERA5 not merged — zero-filling physics inputs.")
        print("[WARN] Run `python main.py download` + `preprocess` with ERA5 for full physics.")
        for col in missing_era5:
            train_df[col] = 0.0
            val_df[col] = 0.0

    available = input_cols
    print(f"[MAIN] Using input features: {available}")
    if missing_era5:
        print(f"[MAIN] Note: {missing_era5} are zero-filled (ERA5 absent)")

    def df_to_tensors(df, include_ic=False):
        inputs = torch.tensor(df[available].values, dtype=torch.float32)
        targets = torch.tensor(df[target_col].values, dtype=torch.float32).unsqueeze(1)
        mask = torch.tensor(df[target_col].notna().values, dtype=torch.float32).unsqueeze(1)


        targets = torch.nan_to_num(targets, nan=0.0)
        result = {"inputs": inputs, "targets": targets, "mask": mask}

        if include_ic:


            ic_rows = (
                df.sort_values("t_norm")
                  .groupby("station", observed=True)
                  .first()
                  .reset_index()
            )
            ic_rows = ic_rows.dropna(subset=[target_col])
            if len(ic_rows) > 0:
                ic_inputs = torch.tensor(ic_rows[available].values, dtype=torch.float32)
                ic_targets = torch.tensor(ic_rows[target_col].values, dtype=torch.float32).unsqueeze(1)
                result["ic_inputs"] = ic_inputs
                result["ic_targets"] = ic_targets
                print(f"[MAIN] IC points: {len(ic_rows)} (one per station at t=0)")

        return result

    train_data = df_to_tensors(train_df, include_ic=True)
    val_data = df_to_tensors(val_df)

    print(f"[MAIN] Train: {len(train_df):,} samples | Val: {len(val_df):,} samples")
    print(f"[MAIN] Inverse mode: {inverse}")

    trainer = AerasTrainer(
        train_data=train_data,
        val_data=val_data,
        inverse_mode=inverse,
        use_wandb=True,
    )
    trainer.train()


def cmd_evaluate():
    from src.evaluation.evaluate_forward import evaluate_all
    from src.evaluation.evaluate_inverse import evaluate_inverse
    evaluate_all()
    print("\n")
    evaluate_inverse()


def main():
    parser = argparse.ArgumentParser(description="aeras")
    parser.add_argument(
        "command",
        choices=["download", "preprocess", "train", "train-inverse", "evaluate", "all"],
        help="Pipeline stage to run",
    )
    args = parser.parse_args()

    if args.command == "download":
        cmd_download()
    elif args.command == "preprocess":
        cmd_preprocess()
    elif args.command == "train":
        cmd_train(inverse=False)
    elif args.command == "train-inverse":
        cmd_train(inverse=True)
    elif args.command == "evaluate":
        cmd_evaluate()
    elif args.command == "all":
        cmd_download()
        cmd_preprocess()
        cmd_train(inverse=False)
        cmd_train(inverse=True)
        cmd_evaluate()


if __name__ == "__main__":
    main()
