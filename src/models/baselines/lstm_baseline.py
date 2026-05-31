import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Tuple

from src.config import DEVICE


class AQSequenceDataset(Dataset):

    def __init__(
        self,
        data: pd.DataFrame,
        input_window: int = 24,
        output_window: int = 24,
        feature_cols: list = None,
        target_col: str = "pm25_norm",
    ):
        self.input_window = input_window
        self.output_window = output_window

        if feature_cols is None:
            feature_cols = [c for c in data.columns if c.endswith("_norm")]
        else:
            feature_cols = [c for c in feature_cols if c in data.columns]

        self.features = data[feature_cols].values.astype(np.float32)
        self.targets = data[target_col].values.astype(np.float32)


        total_window = input_window + output_window
        self.valid_indices = []
        for i in range(len(self.features) - total_window + 1):
            feat_window = self.features[i:i + total_window]
            tgt_window  = self.targets[i + input_window:i + total_window]
            if not (np.isnan(feat_window).any() or np.isnan(tgt_window).any()):
                self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start = self.valid_indices[idx]
        x = self.features[start:start + self.input_window]
        y = self.targets[start + self.input_window:start + self.input_window + self.output_window]
        return torch.tensor(x), torch.tensor(y)


class LSTMBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_window: int = 24,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_window),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)

        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)


def train_lstm(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = None,
) -> Tuple[LSTMBaseline, dict]:
    device = device or DEVICE

    train_dataset = AQSequenceDataset(train_data)
    val_dataset = AQSequenceDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    input_dim = train_dataset.features.shape[1]
    model = LSTMBaseline(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):

        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())


        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_losses.append(criterion(pred, y).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses) if val_losses else float("nan")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 10 == 0:
            print(f"[LSTM] Epoch {epoch:3d} | Train: {train_loss:.4e} | Val: {val_loss:.4e}")

    return model, history


def _eval_lstm_test_sets(model: LSTMBaseline, norm_params: dict, splits_dir: Path, device: str) -> dict:
    from src.config import SPLITS_DIR
    pm25_min = norm_params.get("pm25", {}).get("min", 0.0)
    pm25_max = norm_params.get("pm25", {}).get("max", 500.0)

    results = {}
    for split_name, split_file in [
        ("random", "test_random"),
        ("diwali", "test_diwali"),
        ("winter", "test_winter"),
    ]:
        path = splits_dir / f"{split_file}.parquet"
        if not path.exists():
            print(f"[LSTM] {split_file}.parquet not found, skipping.")
            continue

        df_test = pd.read_parquet(path)
        dataset = AQSequenceDataset(df_test, input_window=24, output_window=1)
        if len(dataset) == 0:
            print(f"[LSTM] {split_name}: no valid windows, skipping.")
            continue

        loader = DataLoader(dataset, batch_size=512, shuffle=False)
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                pred = model(x).cpu().numpy()
                preds.append(pred)
                trues.append(y.numpy())

        pred_norm = np.concatenate(preds).flatten()
        true_norm = np.concatenate(trues).flatten()

        pred = pred_norm * (pm25_max - pm25_min) + pm25_min
        true = true_norm * (pm25_max - pm25_min) + pm25_min

        mae  = float(np.mean(np.abs(pred - true)))
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - true.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

        results[split_name] = {"mae": mae, "rmse": rmse, "r2": r2, "n": len(pred)}
        print(f"[LSTM] {split_name:10s} — MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f} (n={len(pred):,})")

    return results


if __name__ == "__main__":
    import json
    from src.config import SPLITS_DIR, PROCESSED_DIR, DEVICE

    print("  LSTM Baseline Training")

    print("[LSTM] Loading data...")
    train_df = pd.read_parquet(SPLITS_DIR / "train.parquet")
    val_df   = pd.read_parquet(SPLITS_DIR / "val.parquet")

    print("[LSTM] Training model...")
    model, history = train_lstm(train_df, val_df, epochs=50, batch_size=256)

    print("\n[LSTM] Training complete!")
    print(f"  Final Train Loss (MSE): {history['train_loss'][-1]:.4e}")
    print(f"  Final Val Loss   (MSE): {history['val_loss'][-1]:.4e}")

    torch.save(model.state_dict(), "checkpoints/lstm_baseline.pt")
    print("[LSTM] Saved model to checkpoints/lstm_baseline.pt")

    norm_path = PROCESSED_DIR / "normalized_params.json"
    norm_params = json.loads(norm_path.read_text()) if norm_path.exists() else {}

    print("\n[LSTM] OOD test set evaluation (denormalised, PM2.5 in ug/m3, 1h horizon):")
    test_results = _eval_lstm_test_sets(model, norm_params, SPLITS_DIR, DEVICE)

    out_path = Path("checkpoints/lstm_metrics.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"val_mse_normalised": history["val_loss"][-1], "test_denormalised_1h": test_results}, f, indent=2)
    print(f"[LSTM] Results saved to {out_path}")
