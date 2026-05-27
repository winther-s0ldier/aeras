import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from scipy.ndimage import label

from api.model_loader import ModelLoader
from src.config import DELHI_LAT_RANGE, DELHI_LON_RANGE

router = APIRouter()


@router.get("/map")
def get_source_map(t_norm: float = 0.5, resolution: int = 50):
    source = ModelLoader.get_source()
    if source is None:
        raise HTTPException(503, "Source model not loaded. Run inverse training first.")

    source_map = source.get_source_map(
        t_value=t_norm, resolution=resolution, device=ModelLoader.device
    )

    lat_min, lat_max = DELHI_LAT_RANGE
    lon_min, lon_max = DELHI_LON_RANGE
    lats = np.linspace(lat_min, lat_max, resolution).tolist()
    lons = np.linspace(lon_min, lon_max, resolution).tolist()

    return {
        "t_norm": t_norm,
        "resolution": resolution,
        "lats": lats,
        "lons": lons,
        "source_grid": source_map.tolist(),
        "source_max": float(source_map.max()),
        "source_total": float(source_map.sum()),
    }


@router.get("/hotspots")
def get_hotspots(t_norm: float = 0.5, resolution: int = 50, top_n: int = 5):
    source = ModelLoader.get_source()
    if source is None:
        raise HTTPException(503, "Source model not loaded.")

    source_map = source.get_source_map(
        t_value=t_norm, resolution=resolution, device=ModelLoader.device
    )

    lat_min, lat_max = DELHI_LAT_RANGE
    lon_min, lon_max = DELHI_LON_RANGE


    threshold = source_map.max() * 0.4
    binary = source_map > threshold
    labeled, n_features = label(binary)

    hotspots = []
    for i in range(1, min(n_features + 1, top_n + 1)):
        region = np.where(labeled == i)
        cy = float(np.mean(region[0])) / resolution
        cx = float(np.mean(region[1])) / resolution
        intensity = float(source_map[region].mean())

        lat = lat_min + cy * (lat_max - lat_min)
        lon = lon_min + cx * (lon_max - lon_min)
        hotspots.append({
            "rank": i,
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "intensity": round(intensity, 4),
        })

    hotspots.sort(key=lambda h: h["intensity"], reverse=True)
    return {"t_norm": t_norm, "hotspots": hotspots}
