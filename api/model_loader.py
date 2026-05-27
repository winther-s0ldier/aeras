import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.config import CHECKPOINTS_DIR, PROCESSED_DIR, DEVICE
from src.models.pinn import VayuPINN
from src.models.source_net import SourceNet


class ModelLoader:
    _pinn: VayuPINN = None
    _source: SourceNet = None
    _norm_params: dict = None
    device: str = DEVICE

    @classmethod
    def load(cls, checkpoint_name: str = "final"):
        ckpt_path = CHECKPOINTS_DIR / f"vayupinn_{checkpoint_name}.pt"
        if not ckpt_path.exists():
            print(f"[ModelLoader] Checkpoint not found: {ckpt_path}")
            print("[ModelLoader] API will run without a trained model (predictions unavailable).")
            return

        state = torch.load(ckpt_path, map_location=cls.device, weights_only=False)

        cls._pinn = VayuPINN().to(cls.device)
        cls._pinn.load_state_dict(state["model"])
        cls._pinn.eval()

        if "source_model" in state:
            cls._source = SourceNet().to(cls.device)
            cls._source.load_state_dict(state["source_model"])
            cls._source.eval()


        norm_path = PROCESSED_DIR / "normalized_params.json"
        if norm_path.exists():
            with open(norm_path) as f:
                cls._norm_params = json.load(f)

        print(f"[ModelLoader] Loaded {ckpt_path.name} on {cls.device}")

    @classmethod
    def get_pinn(cls) -> VayuPINN:
        return cls._pinn

    @classmethod
    def get_source(cls) -> SourceNet:
        return cls._source

    @classmethod
    def get_norm_params(cls) -> dict:
        return cls._norm_params

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._pinn is not None

    @classmethod
    def normalize_coords(cls, lat: float, lon: float, t_unix: float) -> tuple:
        p = cls._norm_params
        x = (lon - p["longitude"]["min"]) / (p["longitude"]["max"] - p["longitude"]["min"])
        y = (lat - p["latitude"]["min"]) / (p["latitude"]["max"] - p["latitude"]["min"])
        t = (t_unix - p["timestamp"]["min"]) / (p["timestamp"]["max"] - p["timestamp"]["min"])
        return float(x), float(y), float(t)

    @classmethod
    def denormalize_pm25(cls, pm25_norm: float) -> float:
        p = cls._norm_params["pm25"]
        return float(pm25_norm * (p["max"] - p["min"]) + p["min"])
