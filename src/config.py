import os
from pathlib import Path
from dotenv import load_dotenv

# ── Project Identity ───────────────────────────────────────────────
PROJECT_NAME = "aeras"
PROJECT_DESCRIPTION = "Physics-Informed Neural Network for Delhi NCR Air Quality"
WANDB_PROJECT = "aeras"
CHECKPOINT_PREFIX = "aeras_v9_perpoll_params"

# ── Environment detection ──────────────────────────────────────────
# Kaggle mounts datasets at /kaggle/input/<dataset-slug>/
# Output (checkpoints, logs) goes to /kaggle/working/
_ON_KAGGLE = Path("/kaggle/input").exists()

ROOT_DIR       = Path(__file__).resolve().parent.parent
_KAGGLE_DATA   = Path("/kaggle/input/aeras-splits")

DATA_DIR       = _KAGGLE_DATA       if _ON_KAGGLE else ROOT_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = _KAGGLE_DATA / "processed"  if _ON_KAGGLE else DATA_DIR / "processed"
SPLITS_DIR     = _KAGGLE_DATA / "splits"      if _ON_KAGGLE else DATA_DIR / "splits"
CHECKPOINTS_DIR = Path("/kaggle/working/checkpoints") if _ON_KAGGLE else ROOT_DIR / "checkpoints"


for d in [RAW_DIR / "cpcb", RAW_DIR / "era5", RAW_DIR / "edgar",
          PROCESSED_DIR, SPLITS_DIR, CHECKPOINTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


load_dotenv(ROOT_DIR / ".env")
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")
KAGGLE_KEY = os.getenv("KAGGLE_KEY", "")
CDS_URL = os.getenv("CDS_URL", "https://cds.climate.copernicus.eu/api")
CDS_KEY = os.getenv("CDS_KEY", "")


DELHI_BBOX = [29.0, 76.5, 27.5, 77.8]
DELHI_LAT_RANGE = (27.5, 29.0)
DELHI_LON_RANGE = (76.5, 77.8)


# ── Holiday Flag — Indian Public Holidays + Diwali ─────────────────
# Used as binary indicator feature. Diwali is the most important
# (largest PM2.5 spike of the year). Format: 'YYYY-MM-DD'.
INDIAN_HOLIDAYS = {
    # Diwali (varies each year)
    "2015-11-11", "2016-10-30", "2017-10-19", "2018-11-07", "2019-10-27", "2020-11-14",
    # Holi
    "2015-03-06", "2016-03-24", "2017-03-13", "2018-03-02", "2019-03-21", "2020-03-10",
    # Republic Day
    "2015-01-26", "2016-01-26", "2017-01-26", "2018-01-26", "2019-01-26", "2020-01-26",
    # Independence Day
    "2015-08-15", "2016-08-15", "2017-08-15", "2018-08-15", "2019-08-15", "2020-08-15",
    # Gandhi Jayanti
    "2015-10-02", "2016-10-02", "2017-10-02", "2018-10-02", "2019-10-02", "2020-10-02",
    # Christmas (used for fireworks in some areas)
    "2015-12-25", "2016-12-25", "2017-12-25", "2018-12-25", "2019-12-25", "2020-12-25",
    # New Year (fireworks)
    "2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01", "2020-01-01",
    # Diwali ± 2 days (preparations + cleanup) — broader window
    "2015-11-09", "2015-11-10", "2015-11-12", "2015-11-13",
    "2016-10-28", "2016-10-29", "2016-10-31", "2016-11-01",
    "2017-10-17", "2017-10-18", "2017-10-20", "2017-10-21",
    "2018-11-05", "2018-11-06", "2018-11-08", "2018-11-09",
    "2019-10-25", "2019-10-26", "2019-10-28", "2019-10-29",
    "2020-11-12", "2020-11-13", "2020-11-15", "2020-11-16",
}

YEARS = [2018, 2019, 2020]
POLLUTANTS = ["PM2.5", "NO2", "O3", "SO2"]
TARGET_POLLUTANT = "PM2.5"


MAX_GAP_HOURS_INTERPOLATE = 3
MIN_MONTHLY_COVERAGE = 0.70
OUTLIER_SIGMA = 3.0
PM25_MIN = 0.0
PM25_MAX = 1000.0


HIDDEN_LAYERS = 8
HIDDEN_DIM = 256                   # Run D: 128 → 256 (MLP bottleneck was too tight for 16-dim Fourier input)
ACTIVATION = "tanh"
FOURIER_FEATURES = 128             # Run D: 64 → 128 (more freq resolution for wider input space)
FOURIER_SIGMA = 5.0                # Run D: 10 → 5 (σ√d ≈ 20; keeps projections out of sin/cos aliasing regime)


INPUT_DIM = 7
OUTPUT_DIM = 4


LEARNING_RATE = 1e-3
BATCH_SIZE = 1024
NUM_COLLOCATION = 100_000
EPOCHS = 50_000
CHECKPOINT_EVERY = 5_000


W_DATA = 1.0
W_PDE = 0.1
W_BC = 0.01
W_IC = 0.1


LAMBDA_DEP_INIT = 0.01
LOG_DX_INIT = -2.0
LOG_DY_INIT = -2.0


W_SOURCE_SPARSITY = 0.001
SOURCE_NET_HIDDEN = 64
SOURCE_NET_LAYERS = 4


CURRICULUM_DATA_ONLY_EPOCHS = 5_000
CURRICULUM_PDE_RAMPUP_EPOCHS = 10_000


import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = True
