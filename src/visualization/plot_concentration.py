import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import DELHI_LAT_RANGE, DELHI_LON_RANGE, DEVICE


def generate_spatial_grid(resolution: int = 100) -> np.ndarray:
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return xx, yy


def predict_concentration_field(
    model,
    t_norm: float,
    u_wind: float = 0.5,
    v_wind: float = 0.5,
    resolution: int = 100,
    device: str = DEVICE,
    norm_params: dict = None,
) -> np.ndarray:
    xx, yy = generate_spatial_grid(resolution)

    inputs = torch.tensor(
        np.column_stack([
            xx.flatten(),
            yy.flatten(),
            np.full(resolution ** 2, t_norm),
            np.full(resolution ** 2, u_wind),
            np.full(resolution ** 2, v_wind),
        ]),
        dtype=torch.float32,
    ).to(device)

    model.eval()
    with torch.no_grad():
        C = model(inputs).cpu().numpy().reshape(resolution, resolution)


    if norm_params and "pm25" in norm_params:
        pm_min = norm_params["pm25"]["min"]
        pm_max = norm_params["pm25"]["max"]
        C = C * (pm_max - pm_min) + pm_min

    return C


def plot_concentration_heatmap(
    C: np.ndarray,
    title: str = "PM2.5 Concentration (μg/m³)",
    station_lats: np.ndarray = None,
    station_lons: np.ndarray = None,
    station_values: np.ndarray = None,
    save_path: Optional[str] = None,
):
    lat_range = DELHI_LAT_RANGE
    lon_range = DELHI_LON_RANGE

    fig = go.Figure()


    fig.add_trace(go.Heatmap(
        z=C.T,
        x=np.linspace(lon_range[0], lon_range[1], C.shape[0]),
        y=np.linspace(lat_range[0], lat_range[1], C.shape[1]),
        colorscale=[
            [0.0, "#00e400"],
            [0.1, "#ffff00"],
            [0.2, "#ff7e00"],
            [0.4, "#ff0000"],
            [0.6, "#8f3f97"],
            [1.0, "#7e0023"],
        ],
        colorbar=dict(title="PM2.5<br>(μg/m³)"),
        zmin=0,
        zmax=500,
    ))


    if station_lons is not None and station_lats is not None:
        marker_text = None
        if station_values is not None:
            marker_text = [f"{v:.0f} μg/m³" for v in station_values]

        fig.add_trace(go.Scatter(
            x=station_lons,
            y=station_lats,
            mode="markers+text",
            marker=dict(size=8, color="white", line=dict(width=2, color="black")),
            text=marker_text,
            textposition="top center",
            textfont=dict(size=9),
            name="CPCB Stations",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        width=800,
        height=700,
        template="plotly_dark",
    )

    if save_path:
        fig.write_html(save_path)
        print(f"[VIZ] Saved: {save_path}")

    return fig


def plot_time_series_comparison(
    times: np.ndarray,
    true_values: np.ndarray,
    pinn_pred: np.ndarray,
    lstm_pred: np.ndarray = None,
    station_name: str = "Station",
    save_path: Optional[str] = None,
):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=times, y=true_values,
        mode="lines", name="Observed (CPCB)",
        line=dict(color="#ffffff", width=2),
    ))

    fig.add_trace(go.Scatter(
        x=times, y=pinn_pred,
        mode="lines", name="aeras",
        line=dict(color="#00d4ff", width=2),
    ))

    if lstm_pred is not None:
        fig.add_trace(go.Scatter(
            x=times, y=lstm_pred,
            mode="lines", name="LSTM Baseline",
            line=dict(color="#ff6b6b", width=2, dash="dash"),
        ))

    fig.update_layout(
        title=f"{station_name} — PM2.5 Prediction",
        xaxis_title="Time",
        yaxis_title="PM2.5 (μg/m³)",
        template="plotly_dark",
        width=1000,
        height=400,
    )

    if save_path:
        fig.write_html(save_path)

    return fig
