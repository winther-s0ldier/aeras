# train_v24_satellite.py -- Dual satellite (AER_AI + MODIS) on frozen v19
# Lessons applied:
#   v20/21: binary gates CANNOT be inside Fourier embedding -> hard gate outside
#   v22: frozen base + zero-init corrections + independent gates = works
#   v23: fire distribution shift -> keep sat_nets simple [x,y,t,sat_val] only
#   v24: adds MODIS AOD (r=0.46) as second independent correction on top of v22

import sys, os, time, json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pathlib import Path
from sklearn.metrics import r2_score

ON_KAGGLE   = Path("/kaggle/input").exists()
REPO        = Path("/kaggle/working/aeras") if ON_KAGGLE else Path(__file__).parent
SPLITS_DIR  = Path("/kaggle/input/datasets/b1042rudrakumar/kaggle-dataset-fixed") if ON_KAGGLE else REPO/"data/splits"
V19_CKPT    = Path("/kaggle/working/v19_checkpoints/aeras_v19_da4_final.pt")
CKPT_DIR    = Path("/kaggle/working")
MODIS_CSV   = REPO / "data" / "modis_aod_station_daily.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}  |  Kaggle: {ON_KAGGLE}")

# ── Column layout ───────────────────────────────────────────────────────────
TARGET_COLS  = ["pm25_norm","no2_norm","o3_norm","so2_norm"]
BASE_COLS_FIXED = ["x_norm","y_norm","t_norm","u_wind_norm","v_wind_norm",
                   "temp_norm","blh_norm"]
LAG_COLS_OUT = ["pm25_lag1h_norm","no2_lag1h_norm","o3_lag1h_norm","so2_lag1h_norm"]
# Final base = BASE_COLS_FIXED(7) + LAG_COLS_OUT(4) = 11
# Full tensor: 0-10=base  11=aer_norm  12=aer_avail  13=mod_norm  14=mod_avail

# ── Architecture ────────────────────────────────────────────────────────────
class FourierEmbedding(nn.Module):
    def __init__(self, in_dim, n_freq=128, sigma=5.0):
        super().__init__()
        self.register_buffer("B", torch.randn(in_dim, n_freq) * sigma)
    def forward(self, x):
        p = x @ self.B
        return torch.cat([torch.sin(p), torch.cos(p)], dim=-1)

class AerasPINN(nn.Module):
    # Attribute names match aeras_v19_da4_final.pt checkpoint exactly
    def __init__(self, in_dim=11, out_dim=4, hidden=256, n_layers=8, n_freq=128):
        super().__init__()
        self.fourier = FourierEmbedding(in_dim, n_freq)   # checkpoint: fourier.B
        dims = [n_freq*2] + [hidden]*n_layers + [out_dim]
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)              # checkpoint: network.X.*
        self.log_Dx         = nn.Parameter(torch.full((out_dim,), -2.0))
        self.log_Dy         = nn.Parameter(torch.full((out_dim,), -2.0))
        self.log_lambda_dep = nn.Parameter(torch.log(torch.full((out_dim,), 0.01)))  # checkpoint key
    def forward(self, x):
        return self.network(self.fourier(x))

class SatCorrectionNet(nn.Module):
    # Input: [x, y, t, sat_value]  -> correction for all 4 pollutants
    # Zero-init output: corrections start at 0 -> guaranteed v19 floor
    def __init__(self, in_dim=4, hidden=64, out_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    def forward(self, x):
        return self.net(x)

class DualSatPINN(nn.Module):
    def __init__(self, v19_path):
        super().__init__()
        # Frozen v19 base (527K params, all frozen)
        self.base = AerasPINN(in_dim=11)
        raw = torch.load(v19_path, map_location="cpu", weights_only=False)
        self.base.load_state_dict(raw["model"])
        for p in self.base.parameters():
            p.requires_grad_(False)
        # Two independent satellite correction nets (~13K params each)
        self.aer_net = SatCorrectionNet(in_dim=4, hidden=64, out_dim=4)  # AER_AI
        self.mod_net = SatCorrectionNet(in_dim=4, hidden=64, out_dim=4)  # MODIS AOD

    def forward(self, x):
        # x: [base(11), aer_norm, aer_avail, mod_norm, mod_avail]
        base_pred = self.base(x[:, :11])

        aer_in    = torch.cat([x[:, 0:3], x[:, 11:12]], dim=1)  # [x,y,t,aer]
        mod_in    = torch.cat([x[:, 0:3], x[:, 13:14]], dim=1)  # [x,y,t,mod]

        # Hard gates: correction=0 when satellite unavailable
        aer_corr  = x[:, 12:13] * self.aer_net(aer_in)
        mod_corr  = x[:, 14:15] * self.mod_net(mod_in)

        return base_pred + aer_corr + mod_corr

# ── Data loading ────────────────────────────────────────────────────────────
def load_split_with_modis(split_name, modis_df, modis_mean, modis_std):
    df = pd.read_parquet(SPLITS_DIR / f"{split_name}.parquet")

    # Compute lag features on the fly (Kaggle splits don't have precomputed lags)
    df = df.sort_values(["station","timestamp"]).reset_index(drop=True)
    for pol in ["pm25","no2","o3","so2"]:
        df[f"{pol}_lag1h_norm"] = (df.groupby("station")[f"{pol}_norm"]
                                     .shift(1).fillna(0).astype(np.float32))
    base_cols = BASE_COLS_FIXED + LAG_COLS_OUT

    # Drop old zero-filled modis columns if present
    for c in ["modis_aod","modis_aod_norm","modis_aod_available"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Merge fresh MODIS AOD
    df["_date"] = pd.to_datetime(df["timestamp"]).dt.date
    modis_day = modis_df.copy()
    modis_day["_date"] = pd.to_datetime(modis_day["date"]).dt.date
    df = df.merge(modis_day[["station","_date","aod_047"]], on=["station","_date"], how="left")
    df["modis_aod_available"] = df["aod_047"].notna().astype(np.float32)
    df["modis_aod_norm"] = ((df["aod_047"].fillna(modis_mean) - modis_mean) / modis_std).astype(np.float32)
    df = df.drop(columns=["_date","aod_047"])

    # aer_ai_available: compute from aer_ai_norm if not present
    if "aer_ai_available" not in df.columns:
        AER_MEAN_FILL = 0.427887
        df["aer_ai_available"] = (df["aer_ai_norm"] != AER_MEAN_FILL).astype(np.float32)

    final_cols = base_cols + ["aer_ai_norm","aer_ai_available","modis_aod_norm","modis_aod_available"]
    missing = [c for c in final_cols + TARGET_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing in {split_name}: {missing}")

    X = torch.tensor(df[final_cols].fillna(0).values, dtype=torch.float32)
    y = torch.tensor(df[TARGET_COLS].values, dtype=torch.float32)
    return X, y

print("Loading MODIS CSV...")
modis_df  = pd.read_csv(MODIS_CSV)
modis_mean = float(modis_df["aod_047"].mean())
modis_std  = float(modis_df["aod_047"].std())
print(f"MODIS: {len(modis_df):,} records  mean={modis_mean:.4f}  std={modis_std:.4f}")

print("Loading splits...")
X_tr, y_tr = load_split_with_modis("train",       modis_df, modis_mean, modis_std)
X_va, y_va = load_split_with_modis("val",         modis_df, modis_mean, modis_std)
X_di, y_di = load_split_with_modis("test_diwali", modis_df, modis_mean, modis_std)
X_wi, y_wi = load_split_with_modis("test_winter", modis_df, modis_mean, modis_std)
X_ra, y_ra = load_split_with_modis("test_random", modis_df, modis_mean, modis_std)

# Report MODIS coverage per split
for name, X in [("train",X_tr),("val",X_va),("diwali",X_di),("winter",X_wi),("random",X_ra)]:
    cov = X[:, 14].mean().item() * 100
    print(f"  {name}: {len(X):>7} rows  MODIS coverage={cov:.1f}%")

X_tr, y_tr = X_tr.to(DEVICE), y_tr.to(DEVICE)
X_va, y_va = X_va.to(DEVICE), y_va.to(DEVICE)

# ── NaN guards ──────────────────────────────────────────────────────────────
# Inputs: replace any NaN/Inf with 0 (shouldn't happen but belt-and-braces)
X_tr = torch.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
X_va = torch.nan_to_num(X_va, nan=0.0, posinf=0.0, neginf=0.0)
# Targets: CPCB has real missing readings -> masked loss required
nan_tr = torch.isnan(y_tr).sum().item()
nan_va = torch.isnan(y_va).sum().item()
print(f"NaN in y_tr: {nan_tr:,}  y_va: {nan_va:,}  (masked loss will skip these)")

def masked_mse(pred, target):
    """MSELoss that ignores NaN targets (missing CPCB readings)."""
    mask = ~torch.isnan(target)
    if not mask.any():
        return pred.sum() * 0.0
    return ((pred[mask] - target[mask]) ** 2).mean()

# ── Build model ─────────────────────────────────────────────────────────────
print(f"\nBuilding DualSatPINN from v19: {V19_CKPT}")
model = DualSatPINN(V19_CKPT).to(DEVICE)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} params")

# ── Adam training ───────────────────────────────────────────────────────────
optimizer = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad], lr=1e-3)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5000, T_mult=1, eta_min=1e-5)

EPOCHS      = 20000
PRINT_EVERY = 2000
best_val    = float("inf")
t0          = time.time()

print(f"\nAdam training ({EPOCHS} epochs)...")
for epoch in range(1, EPOCHS+1):
    model.train()
    pred = model(X_tr)
    loss = masked_mse(pred, y_tr)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if epoch % PRINT_EVERY == 0:
        model.eval()
        with torch.no_grad():
            vl = masked_mse(model(X_va), y_va).item()
        if vl < best_val:
            best_val = vl
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "modis_mean": modis_mean, "modis_std": modis_std},
                       CKPT_DIR / "aeras_v24_best.pt")
        print(f"  ep {epoch:>5} | train={loss.item():.5f} | val={vl:.5f} | best={best_val:.5f} | {time.time()-t0:.0f}s")

# ── L-BFGS fine-tune ────────────────────────────────────────────────────────
print("\nL-BFGS fine-tuning...")
lbfgs = torch.optim.LBFGS(
    [p for p in model.parameters() if p.requires_grad],
    lr=0.1, max_iter=20, history_size=50, line_search_fn="strong_wolfe")
for i in range(50):
    def closure():
        lbfgs.zero_grad()
        l = masked_mse(model(X_tr), y_tr)
        l.backward()
        return l
    lbfgs.step(closure)
    if (i+1) % 10 == 0:
        with torch.no_grad():
            vl = masked_mse(model(X_va), y_va).item()
        print(f"  step {(i+1)*20:>4} | val={vl:.5f}")

# Final save
torch.save({"model": model.state_dict(), "epoch": EPOCHS,
            "modis_mean": modis_mean, "modis_std": modis_std},
           CKPT_DIR / "aeras_v24_final.pt")
print("Saved: aeras_v24_final.pt")

# ── Evaluation ───────────────────────────────────────────────────────────────
def evaluate(X, y, label):
    model.eval()
    with torch.no_grad():
        pred = model(X.to(DEVICE)).cpu().numpy()
    true = y.cpu().numpy()
    pols = ["PM2.5","NO2","O3","SO2"]
    results = {}
    print(f"\n{label}:")
    for i, pol in enumerate(pols):
        mask = np.isfinite(true[:,i])
        if mask.sum() < 10: continue
        r2  = r2_score(true[mask,i], pred[mask,i])
        mae = float(np.abs(true[mask,i]-pred[mask,i]).mean())
        results[pol] = {"r2": round(r2,4), "mae": round(mae,4)}
        print(f"  {pol:>5}: R²={r2:+.4f}  MAE={mae:.4f}")
    return results

results = {}
evaluate(X_va, y_va, "Validation")
results["diwali"] = evaluate(X_di.to(DEVICE), y_di, "test_diwali")
results["winter"] = evaluate(X_wi.to(DEVICE), y_wi, "test_winter")
results["random"] = evaluate(X_ra.to(DEVICE), y_ra, "test_random")

with open(CKPT_DIR / "v24_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
print("Done.")
