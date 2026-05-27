import torch, pandas as pd
from src.config import SPLITS_DIR
from src.training.trainer import VayuPINNTrainer

train_df = pd.read_parquet(SPLITS_DIR / 'train.parquet')
val_df   = pd.read_parquet(SPLITS_DIR / 'val.parquet')

cols = ['x_norm','y_norm','t_norm','u_wind_norm','v_wind_norm']
def to_t(df):
    inp  = torch.tensor(df[cols].values, dtype=torch.float32)
    tgt  = torch.nan_to_num(torch.tensor(df['pm25_norm'].values, dtype=torch.float32).unsqueeze(1), nan=0.0)
    mask = torch.tensor(df['pm25_norm'].notna().values, dtype=torch.float32).unsqueeze(1)
    ic   = df.sort_values('t_norm').groupby('station', observed=True).first().reset_index().dropna(subset=['pm25_norm'])
    return {'inputs':inp,'targets':tgt,'mask':mask,
            'ic_inputs': torch.tensor(ic[cols].values,dtype=torch.float32),
            'ic_targets':torch.tensor(ic['pm25_norm'].values,dtype=torch.float32).unsqueeze(1)}

trainer = VayuPINNTrainer(to_t(train_df), to_t(val_df), use_wandb=False)
trainer.train(epochs=100)
print('SMOKE TEST PASSED')
