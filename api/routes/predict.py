from datetime import datetime
from typing import Optional

import torch
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.model_loader import ModelLoader
from src.config import DELHI_LAT_RANGE, DELHI_LON_RANGE

router = APIRouter()


class PredictRequest(BaseModel):
    latitude: float  = Field(..., ge=DELHI_LAT_RANGE[0], le=DELHI_LAT_RANGE[1],
                             example=28.6139, description="Latitude (27.5–29.0°N)")
    longitude: float = Field(..., ge=DELHI_LON_RANGE[0], le=DELHI_LON_RANGE[1],
                             example=77.2090, description="Longitude (76.5–77.8°E)")
    timestamp: str   = Field(..., example="2019-11-01T12:00:00",
                             description="ISO 8601 datetime (2018–2020)")
    u_wind: float    = Field(0.0, description="Eastward wind (m/s), 0 if unknown")
    v_wind: float    = Field(0.0, description="Northward wind (m/s), 0 if unknown")


class PredictResponse(BaseModel):
    pm25_ugm3: float
    pm25_norm: float
    aqi_category: str
    latitude: float
    longitude: float
    timestamp: str


def aqi_category(pm25: float) -> str:
    if pm25 <= 12:   return "Good"
    if pm25 <= 35.4: return "Satisfactory"
    if pm25 <= 55.4: return "Moderate"
    if pm25 <= 150:  return "Poor"
    if pm25 <= 250:  return "Very Poor"
    return "Severe"


@router.post("/", response_model=PredictResponse)
def predict_point(req: PredictRequest):
    model = ModelLoader.get_pinn()
    if model is None:
        raise HTTPException(503, "Model not loaded. Run training first.")

    norm = ModelLoader.get_norm_params()
    if norm is None:
        raise HTTPException(503, "Normalization params not found.")


    try:
        dt = datetime.fromisoformat(req.timestamp)
        t_unix = dt.timestamp()
    except ValueError:
        raise HTTPException(400, f"Invalid timestamp: {req.timestamp}")

    x, y, t = ModelLoader.normalize_coords(req.latitude, req.longitude, t_unix)


    u_norm = req.u_wind / 10.0
    v_norm = req.v_wind / 10.0

    inputs = torch.tensor([[x, y, t, u_norm, v_norm]],
                          dtype=torch.float32, device=ModelLoader.device)

    with torch.no_grad():
        pm25_norm = model(inputs).item()

    pm25_ugm3 = ModelLoader.denormalize_pm25(pm25_norm)
    pm25_ugm3 = max(0.0, pm25_ugm3)

    return PredictResponse(
        pm25_ugm3=round(pm25_ugm3, 2),
        pm25_norm=round(pm25_norm, 4),
        aqi_category=aqi_category(pm25_ugm3),
        latitude=req.latitude,
        longitude=req.longitude,
        timestamp=req.timestamp,
    )


@router.get("/grid")
def predict_grid(
    timestamp: str,
    resolution: int = 30,
    u_wind: float = 0.0,
    v_wind: float = 0.0,
):
    model = ModelLoader.get_pinn()
    if model is None:
        raise HTTPException(503, "Model not loaded.")

    try:
        dt = datetime.fromisoformat(timestamp)
        t_unix = dt.timestamp()
    except ValueError:
        raise HTTPException(400, f"Invalid timestamp: {timestamp}")

    norm = ModelLoader.get_norm_params()
    lat_min, lat_max = DELHI_LAT_RANGE
    lon_min, lon_max = DELHI_LON_RANGE

    lats = np.linspace(lat_min, lat_max, resolution)
    lons = np.linspace(lon_min, lon_max, resolution)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")


    x_norm = (lon_grid - norm["longitude"]["min"]) / (norm["longitude"]["max"] - norm["longitude"]["min"])
    y_norm = (lat_grid - norm["latitude"]["min"]) / (norm["latitude"]["max"] - norm["latitude"]["min"])
    t_norm_val = (t_unix - norm["timestamp"]["min"]) / (norm["timestamp"]["max"] - norm["timestamp"]["min"])
    u_norm = u_wind / 10.0
    v_norm = v_wind / 10.0

    x_flat = x_norm.flatten()
    y_flat = y_norm.flatten()
    t_flat = np.full_like(x_flat, t_norm_val)
    u_flat = np.full_like(x_flat, u_norm)
    v_flat = np.full_like(x_flat, v_norm)

    inputs = torch.tensor(
        np.stack([x_flat, y_flat, t_flat, u_flat, v_flat], axis=1),
        dtype=torch.float32, device=ModelLoader.device
    )

    with torch.no_grad():
        pm25_norm_flat = model(inputs).cpu().numpy().flatten()

    pm25_flat = np.clip(
        pm25_norm_flat * (norm["pm25"]["max"] - norm["pm25"]["min"]) + norm["pm25"]["min"],
        0, None
    )
    pm25_grid = pm25_flat.reshape(resolution, resolution)

    return {
        "timestamp": timestamp,
        "resolution": resolution,
        "lat_range": [float(lat_min), float(lat_max)],
        "lon_range": [float(lon_min), float(lon_max)],
        "lats": lats.tolist(),
        "lons": lons.tolist(),
        "pm25_grid": pm25_grid.tolist(),
        "pm25_min": float(pm25_grid.min()),
        "pm25_max": float(pm25_grid.max()),
    }
