"""
train_chem_da.py  —  Multi-pollutant Data Assimilation (± chemistry)
─────────────────────────────────────────────────────────────────────
The Phase-7 fallback if v15 confirms simplified Leighton chemistry can't be
closed from this data. Instead of fighting the chemistry, give NO2 and O3 the
SAME data-assimilation anchor that made PM2.5 work in v13 — their own lag-1h
features — and run it BOTH WAYS so we cleanly separate DA from chemistry.

  INPUT_DIM = 10:  x, y, t, u, v, temp, blh, pm25_lag1h, no2_lag1h, o3_lag1h

Two arms, selected by one flag:
  --chemistry off : multi-pollutant DA only (transport PDE, no Leighton)   [DA baseline]
  --chemistry on  : multi-pollutant DA + Leighton NO2↔O3 photochemistry    [DA + chem]

The comparison (off vs on) answers the ONLY honest question left about chemistry:
  "Does photochemistry add anything ON TOP of data assimilation for NO2/O3?"
  - on ≈ off  → chemistry is redundant; DA subsumes it. Clean finding.
  - on  > off → chemistry adds measurable value over DA. Best case.
  - on  < off → chemistry hurts; confirms null-cycle/nighttime story.

⚠ HONESTY GUARD: if the `off` arm already makes NO2/O3 R² positive, that is a
DATA-ASSIMILATION result (the lag feature), NOT a chemistry result — exactly
like PM2.5 in v13. Do NOT report "our chemistry predicts ozone." Report
"multi-pollutant DA predicts ozone; chemistry adds X on top (or nothing)."

Run on GPU 0 (two separate runs):
  CUDA_VISIBLE_DEVICES=0 python train_chem_da.py --chemistry off
  CUDA_VISIBLE_DEVICES=0 python train_chem_da.py --chemistry on
"""

import os, sys, json, argparse
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import src.config as cfg

# ── parse the chemistry toggle BEFORE touching config / imports ──────────────
_parser = argparse.ArgumentParser()
_parser.add_argument("--chemistry", choices=["on", "off"], default="off",
                     help="'on' = DA + Leighton photochemistry; 'off' = DA only")
_args, _ = _parser.parse_known_args()
CHEMISTRY_ON = (_args.chemistry == "on")

# ── must be set BEFORE importing AerasPINN / trainer (import caches INPUT_DIM) ─
cfg.INPUT_DIM          = 10
cfg.CHECKPOINT_PREFIX  = "aeras_v17_chemda_on" if CHEMISTRY_ON else "aeras_v17_da3_off"
cfg.BATCH_SIZE         = 4096
cfg.NUM_COLLOCATION    = 200_000
cfg.CHECKPOINT_EVERY   = 2_500
cfg.LBFGS_MAX_ITER     = 1_000
cfg.LBFGS_DATA_CHUNK   = 20_000
cfg.LBFGS_COLLOC_CHUNK = 15_000

from src.config import CHECKPOINTS_DIR, POLLUTANTS
from src.training.trainer import AerasTrainer
from src.models.pinn import AerasPINN
from src.models.chemistry import ChemistryModule, DELHI_LATITUDE, IST_OFFSET_HOURS

SPLITS_DIR = Path("/kaggle/input/datasets/rudrakumar21/aeras-splits")

# Lag columns appended at the END so wind stays at idx 3,4 and t at idx 2.
INPUT_COLS  = ["x_norm","y_norm","t_norm","u_wind_norm","v_wind_norm",
               "temp_norm","blh_norm",
               "pm25_lag1h_norm","no2_lag1h_norm","o3_lag1h_norm"]
TARGET_COLS = ["pm25_norm","no2_norm","o3_norm","so2_norm"]

_NORM = {
    "pm25": {"min": 0.03,  "max": 509.75},
    "no2":  {"min": 0.01,  "max": 278.21},
    "o3":   {"min": 0.01,  "max": 66.17},
    "so2":  {"min": 0.01,  "max": 500.0},
}

TAG = "CHEMDA" if CHEMISTRY_ON else "DA3"

# v14 chemistry reference (for the chemistry arm) — Diwali R²
V14_REF = {"pm25": 0.752, "no2": -0.167, "o3": -0.334}


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lag-1h per station for ALL three pollutants. First row per station → 0.
    NEVER fall back to the current value (that leaks the target into the input).
    NO2/O3 are often NaN; shift then fillna(0) degrades gracefully.
    """
    df = df.sort_values(["station", "timestamp"]).copy()
    g = df.groupby("station", observed=True)
    df["pm25_lag1h_norm"] = g["pm25_norm"].shift(1).fillna(0.0)
    df["no2_lag1h_norm"]  = g["no2_norm"].shift(1).fillna(0.0)
    df["o3_lag1h_norm"]   = g["o3_norm"].shift(1).fillna(0.0)
    return df


def df_to_tensors(df: pd.DataFrame, include_ic: bool = False) -> dict:
    inputs  = torch.tensor(df[INPUT_COLS].fillna(0).values,  dtype=torch.float32)
    targets = torch.tensor(df[TARGET_COLS].fillna(0).values, dtype=torch.float32)
    mask    = torch.tensor(df[TARGET_COLS].notna().values,   dtype=torch.float32)

    ev = pd.Series(1.0, index=df.index)
    if "is_holiday" in df.columns:
        ev.loc[df["is_holiday"].astype(bool)] *= 3.0
    if "timestamp" in df.columns:
        ev.loc[df["timestamp"].dt.month.isin([12, 1])] *= 2.0
    if "pm25_norm" in df.columns:
        ev.loc[df["pm25_norm"] > 0.4] *= 2.0
    event_weight = torch.tensor(ev.values, dtype=torch.float32).unsqueeze(1)

    result = {"inputs": inputs, "targets": targets,
              "mask": mask, "event_weight": event_weight}

    if include_ic:
        ic_rows = (df.dropna(subset=TARGET_COLS)
                     .sort_values("t_norm")
                     .groupby("station", observed=True).first()
                     .reset_index())
        if len(ic_rows) > 0:
            result["ic_inputs"]  = torch.tensor(
                ic_rows[INPUT_COLS].fillna(0).values, dtype=torch.float32)
            result["ic_targets"] = torch.tensor(
                ic_rows[TARGET_COLS].fillna(0).values, dtype=torch.float32)
            print(f"[{TAG}] IC points: {len(ic_rows)}")

    return result


def evaluate_test(trainer, name: str, device: str):
    try:
        df = pd.read_parquet(SPLITS_DIR / f"{name}.parquet")
    except FileNotFoundError:
        print(f"[{TAG}] {name}.parquet not found, skipping")
        return None

    df = add_lag_features(df)
    inputs  = torch.tensor(df[INPUT_COLS].fillna(0).values, dtype=torch.float32).to(device)
    targets = df[TARGET_COLS].values.astype(float)
    mask    = df[TARGET_COLS].notna().values

    trainer.model.eval()
    with torch.no_grad():
        pred = trainer.model(inputs).cpu().numpy()

    out = {}
    print(f"\n[{name}]")
    print(f"{'Poll':<6} {'R²':>8} {'MAE (μg/m³)':>13} {'n':>8}")
    for i, key in enumerate(["pm25", "no2", "o3", "so2"]):
        v = mask[:, i]
        if v.sum() == 0:
            continue
        p, t = pred[v, i], targets[v, i]
        mae_n  = float(np.mean(np.abs(p - t)))
        ss_r   = float(np.sum((p - t) ** 2))
        ss_t   = float(np.sum((t - t.mean()) ** 2))
        r2     = 1.0 - ss_r / ss_t if ss_t > 0 else float('nan')
        vmin, vmax = _NORM[key]["min"], _NORM[key]["max"]
        mae_phys = float(np.mean(np.abs(
            p * (vmax - vmin) + vmin - (t * (vmax - vmin) + vmin)
        )))
        print(f"{key.upper():<6} {r2:>8.4f} {mae_phys:>13.2f} {int(v.sum()):>8}")
        out[key] = {"mae_norm": mae_n, "r2": r2, "mae_ug_m3": mae_phys, "n": int(v.sum())}
    return out


def main():
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False

    device = cfg.DEVICE
    print(f"[{TAG}] Chemistry : {'ON (DA + Leighton)' if CHEMISTRY_ON else 'OFF (DA only)'}")
    print(f"[{TAG}] Device    : {device}")
    print(f"[{TAG}] INPUT_DIM : {cfg.INPUT_DIM}  (7 base + 3 lags: pm25/no2/o3)")
    print(f"[{TAG}] Prefix    : {cfg.CHECKPOINT_PREFIX}")

    # ── safety assertions ────────────────────────────────────────────────────
    assert POLLUTANTS == ["PM2.5", "NO2", "O3", "SO2"], (
        f"Channel order changed! Chemistry assumes 0=PM2.5,1=NO2,2=O3. Got {POLLUTANTS}"
    )
    _probe = AerasPINN()
    assert _probe.fourier.B.shape[0] == 10, (
        f"AerasPINN built with input_dim={_probe.fourier.B.shape[0]}, expected 10. "
        f"cfg.INPUT_DIM must be set before importing the trainer."
    )
    del _probe

    print(f"\n[{TAG}] Loading training data...")
    train_df = pd.read_parquet(SPLITS_DIR / "train.parquet")
    val_df   = pd.read_parquet(SPLITS_DIR / "val.parquet")

    print(f"[{TAG}] Computing lag-1h features (pm25/no2/o3)...")
    train_df = add_lag_features(train_df)
    val_df   = add_lag_features(val_df)
    for c in ["pm25_lag1h_norm", "no2_lag1h_norm", "o3_lag1h_norm"]:
        nz = (train_df[c] != 0).mean() * 100
        print(f"[{TAG}]   {c}: {nz:.1f}% non-zero")

    train_data = df_to_tensors(train_df, include_ic=True)
    val_data   = df_to_tensors(val_df)

    # ── chemistry module (only for the 'on' arm) ─────────────────────────────
    chem = None
    if CHEMISTRY_ON:
        ts = train_df["timestamp"]
        ts_utc = ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")
        t_min_unix  = ts_utc.min().timestamp()
        t_range_sec = ts_utc.max().timestamp() - t_min_unix

        lat     = getattr(cfg, "DELHI_LATITUDE", DELHI_LATITUDE)
        tz_off  = getattr(cfg, "IST_OFFSET_HOURS", IST_OFFSET_HOURS)
        log_j   = getattr(cfg, "LOG_J_AMP_INIT", -0.5)
        log_eps = getattr(cfg, "LOG_LEIGHTON_EPS_INIT", -3.0)

        chem = ChemistryModule(
            t_min_unix=t_min_unix, t_range_sec=t_range_sec,
            lat_deg=lat, tz_offset_hours=tz_off,
            log_J_amp_init=log_j, log_leighton_eps_init=log_eps,
        )
        print(f"[{TAG}] ChemistryModule: t_min_unix={t_min_unix:.0f} "
              f"range={t_range_sec/86400:.0f}d  J_amp_init={chem.J_amp.item():.4f}  "
              f"eps_init={chem.leighton_eps.item():.4f}")

    print(f"\n[{TAG}] Training (INPUT_DIM=10, chemistry={'ON' if CHEMISTRY_ON else 'OFF'})...")
    trainer = AerasTrainer(
        train_data=train_data,
        val_data=val_data,
        inverse_mode=False,
        use_wandb=False,
        chemistry_module=chem,   # None for the 'off' arm
    )
    trainer.train()

    if CHEMISTRY_ON and chem is not None:
        print(f"\n[{TAG}] Learned J_amp        = {chem.J_amp.item():.6e} "
              f"(init {torch.exp(torch.tensor(getattr(cfg,'LOG_J_AMP_INIT',-0.5))).item():.4f})")
        print(f"[{TAG}] Learned leighton_eps = {chem.leighton_eps.item():.6e}")

    print("\n" + "=" * 64)
    print(f"[{TAG}] FINAL EVALUATION  (chemistry {'ON' if CHEMISTRY_ON else 'OFF'})")
    print("=" * 64)
    if CHEMISTRY_ON:
        print(f"v14 (chem, no NO2/O3 lag) Diwali ref: "
              f"NO2={V14_REF['no2']:.3f}  O3={V14_REF['o3']:.3f}")
    print("Compare the 'on' vs 'off' JSONs to isolate chemistry's marginal value.")

    all_results = {}
    for split in ["test_random", "test_diwali", "test_winter"]:
        r = evaluate_test(trainer, split, device)
        if r:
            all_results[split] = r

    payload = {"chemistry": CHEMISTRY_ON, "results": all_results}
    if CHEMISTRY_ON and chem is not None:
        payload["learned"] = {
            "J_amp": chem.J_amp.item(),
            "leighton_eps": chem.leighton_eps.item(),
        }

    fname = "chemda_on_results.json" if CHEMISTRY_ON else "chemda_off_results.json"
    out_path = CHECKPOINTS_DIR / fname
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[{TAG}] Results saved → {out_path.name}")
    print(f"[{TAG}] Done.")


if __name__ == "__main__":
    main()
