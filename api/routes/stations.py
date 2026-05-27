import pandas as pd
from fastapi import APIRouter, HTTPException
from api.model_loader import ModelLoader
from src.config import PROCESSED_DIR

router = APIRouter()

_stations_cache = None


def _load_stations():
    global _stations_cache
    if _stations_cache is not None:
        return _stations_cache

    path = PROCESSED_DIR / "station_coords.csv"
    if not path.exists():
        return []

    df = pd.read_csv(path)
    _stations_cache = df.to_dict(orient="records")
    return _stations_cache


@router.get("/")
def list_stations():
    stations = _load_stations()
    if not stations:
        raise HTTPException(404, "station_coords.csv not found. Run preprocessing first.")
    return {"count": len(stations), "stations": stations}


@router.get("/{station_name}")
def get_station(station_name: str):
    stations = _load_stations()
    matches = [s for s in stations if station_name.lower() in str(s.get("station", "")).lower()]
    if not matches:
        raise HTTPException(404, f"Station '{station_name}' not found.")
    return matches[0]
