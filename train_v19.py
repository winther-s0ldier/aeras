import os, sys, json
from pathlib import Path
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
import torch
import numpy as np
import pandas as pd
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import src.config as cfg
cfg.INPUT_DIM = 11
cfg.CHECKPOINT_PREFIX = 'aeras_v19_da4'
cfg.BATCH_SIZE = 4096
cfg.NUM_COLLOCATION = 200000
cfg.CHECKPOINT_EVERY = 2500
cfg.LBFGS_MAX_ITER = 1000
cfg.LBFGS_DATA_CHUNK = 20000
cfg.LBFGS_COLLOC_CHUNK = 15000
from src.config import CHECKPOINTS_DIR, POLLUTANTS
from src.training.trainer import AerasTrainer
from src.models.pinn import AerasPINN
SPLITS_DIR = Path('/kaggle/input/datasets/b1042rudrakumar/aeras-data/splits')
INPUT_COLS = ['x_norm', 'y_norm', 't_norm', 'u_wind_norm', 'v_wind_norm', 'temp_norm', 'blh_norm', 'pm25_lag1h_norm', 'no2_lag1h_norm', 'o3_lag1h_norm', 'so2_lag1h_norm']
TARGET_COLS = ['pm25_norm', 'no2_norm', 'o3_norm', 'so2_norm']
_NORM = {'pm25': {'min': 0.03, 'max': 509.75}, 'no2': {'min': 0.01, 'max': 278.21}, 'o3': {'min': 0.01, 'max': 66.17}, 'so2': {'min': 0.01, 'max': 500.0}}
TAG = 'DA4'
V17_REF = {'diwali': {'pm25': 0.7088, 'no2': 0.5012, 'o3': 0.5624, 'so2': -0.002}, 'winter': {'pm25': 0.6093, 'no2': 0.55, 'o3': 0.1939, 'so2': -0.1366}}

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['station', 'timestamp']).copy()
    g = df.groupby('station', observed=True)
    df['pm25_lag1h_norm'] = g['pm25_norm'].shift(1).fillna(0.0)
    df['no2_lag1h_norm'] = g['no2_norm'].shift(1).fillna(0.0)
    df['o3_lag1h_norm'] = g['o3_norm'].shift(1).fillna(0.0)
    df['so2_lag1h_norm'] = g['so2_norm'].shift(1).fillna(0.0)
    return df

def df_to_tensors(df: pd.DataFrame, include_ic: bool=False) -> dict:
    inputs = torch.tensor(df[INPUT_COLS].fillna(0).values, dtype=torch.float32)
    targets = torch.tensor(df[TARGET_COLS].fillna(0).values, dtype=torch.float32)
    mask = torch.tensor(df[TARGET_COLS].notna().values, dtype=torch.float32)
    ev = pd.Series(1.0, index=df.index)
    if 'is_holiday' in df.columns:
        ev.loc[df['is_holiday'].astype(bool)] *= 3.0
    if 'timestamp' in df.columns:
        ev.loc[df['timestamp'].dt.month.isin([12, 1])] *= 2.0
    if 'pm25_norm' in df.columns:
        ev.loc[df['pm25_norm'] > 0.4] *= 2.0
    event_weight = torch.tensor(ev.values, dtype=torch.float32).unsqueeze(1)
    result = {'inputs': inputs, 'targets': targets, 'mask': mask, 'event_weight': event_weight}
    if include_ic:
        ic_rows = df.dropna(subset=TARGET_COLS).sort_values('t_norm').groupby('station', observed=True).first().reset_index()
        if len(ic_rows) > 0:
            result['ic_inputs'] = torch.tensor(ic_rows[INPUT_COLS].fillna(0).values, dtype=torch.float32)
            result['ic_targets'] = torch.tensor(ic_rows[TARGET_COLS].fillna(0).values, dtype=torch.float32)
            print(f'[{TAG}] IC points: {len(ic_rows)}')
    return result

def evaluate_test(trainer, name: str, device: str):
    try:
        df = pd.read_parquet(SPLITS_DIR / f'{name}.parquet')
    except FileNotFoundError:
        print(f'[{TAG}] {name}.parquet not found, skipping')
        return None
    df = add_lag_features(df)
    inputs = torch.tensor(df[INPUT_COLS].fillna(0).values, dtype=torch.float32).to(device)
    targets = df[TARGET_COLS].values.astype(float)
    mask = df[TARGET_COLS].notna().values
    trainer.model.eval()
    with torch.no_grad():
        pred = trainer.model(inputs).cpu().numpy()
    out = {}
    print(f'\n[{name}]')
    print(f"{'Poll':<6} {'R²':>8} {'MAE (μg/m³)':>13} {'n':>8} {'vs v17_DA3_OFF':>15}")
    for i, key in enumerate(['pm25', 'no2', 'o3', 'so2']):
        v = mask[:, i]
        if v.sum() == 0:
            continue
        p, t = (pred[v, i], targets[v, i])
        mae_n = float(np.mean(np.abs(p - t)))
        ss_r = float(np.sum((p - t) ** 2))
        ss_t = float(np.sum((t - t.mean()) ** 2))
        r2 = 1.0 - ss_r / ss_t if ss_t > 0 else float('nan')
        vmin, vmax = (_NORM[key]['min'], _NORM[key]['max'])
        mae_phys = float(np.mean(np.abs(p * (vmax - vmin) + vmin - (t * (vmax - vmin) + vmin))))
        delta_str = ''
        split_key = name.replace('test_', '')
        if split_key in V17_REF and key in V17_REF[split_key]:
            delta = r2 - V17_REF[split_key][key]
            delta_str = f'{delta:+.4f}'
        print(f'{key.upper():<6} {r2:>8.4f} {mae_phys:>13.2f} {int(v.sum()):>8} {delta_str:>15}')
        out[key] = {'mae_norm': mae_n, 'r2': r2, 'mae_ug_m3': mae_phys, 'n': int(v.sum())}
    for col in ['pm25_lag1h_norm', 'no2_lag1h_norm', 'o3_lag1h_norm', 'so2_lag1h_norm']:
        if col in df.columns:
            print(f'  lag {col}: {(df[col] != 0).mean() * 100:.1f}% non-zero')
    return out

def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    device = cfg.DEVICE
    print(f'[{TAG}] Chemistry : OFF (DA only)')
    print(f'[{TAG}] Device    : {device}')
    print(f'[{TAG}] INPUT_DIM : {cfg.INPUT_DIM}  (7 base + 4 lags)')
    print(f'[{TAG}] Prefix    : {cfg.CHECKPOINT_PREFIX}')
    assert POLLUTANTS == ['PM2.5', 'NO2', 'O3', 'SO2'], f'Channel order changed! Assumes 0=PM2.5,1=NO2,2=O3,3=SO2. Got {POLLUTANTS}'
    _probe = AerasPINN()
    assert _probe.fourier.B.shape[0] == 11, f'AerasPINN built with input_dim={_probe.fourier.B.shape[0]}, expected 11.'
    del _probe
    print(f'\n[{TAG}] Loading training data...')
    train_df = pd.read_parquet(SPLITS_DIR / 'train.parquet')
    val_df = pd.read_parquet(SPLITS_DIR / 'val.parquet')
    print(f'[{TAG}] Computing lag-1h features (all 4 pollutants)...')
    train_df = add_lag_features(train_df)
    val_df = add_lag_features(val_df)
    for c in ['pm25_lag1h_norm', 'no2_lag1h_norm', 'o3_lag1h_norm', 'so2_lag1h_norm']:
        nz = (train_df[c] != 0).mean() * 100
        print(f'[{TAG}]   {c}: {nz:.1f}% non-zero')
    train_data = df_to_tensors(train_df, include_ic=True)
    val_data = df_to_tensors(val_df)
    print(f'\n[{TAG}] Training (INPUT_DIM=11, chemistry=OFF)...')
    trainer = AerasTrainer(train_data=train_data, val_data=val_data, inverse_mode=False, use_wandb=False, chemistry_module=None)
    trainer.train()
    print('\n' + '=' * 64)
    print(f'[{TAG}] FINAL EVALUATION  (chemistry OFF)')
    print('=' * 64)
    all_results = {}
    for split in ['test_random', 'test_diwali', 'test_winter']:
        r = evaluate_test(trainer, split, device)
        if r:
            all_results[split] = r
    payload = {'chemistry': False, 'results': all_results}
    out_path = CHECKPOINTS_DIR / 'v19_da4_results.json'
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'\n[{TAG}] Results saved → {out_path.name}')
    print(f'[{TAG}] Done.')
if __name__ == '__main__':
    main()