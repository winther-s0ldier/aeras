import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import DELHI_LAT_RANGE, DELHI_LON_RANGE


def plot_source_map(
    source_field: np.ndarray,
    title: str = "Learned Emission Sources",
    edgar_field: np.ndarray = None,
    save_path: Optional[str] = None,
):
    lat = np.linspace(DELHI_LAT_RANGE[0], DELHI_LAT_RANGE[1], source_field.shape[0])
    lon = np.linspace(DELHI_LON_RANGE[0], DELHI_LON_RANGE[1], source_field.shape[1])

    if edgar_field is not None:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("aeras (Learned)", "EDGAR Inventory (Reference)"),
        )

        fig.add_trace(go.Heatmap(
            z=source_field.T, x=lon, y=lat,
            colorscale="Hot", reversescale=True,
            showscale=True,
        ), row=1, col=1)

        fig.add_trace(go.Heatmap(
            z=edgar_field.T, x=lon, y=lat,
            colorscale="Hot", reversescale=True,
            showscale=True,
        ), row=1, col=2)

        fig.update_layout(width=1400, height=600, template="plotly_dark", title=title)
    else:
        fig = go.Figure(go.Heatmap(
            z=source_field.T, x=lon, y=lat,
            colorscale="Hot", reversescale=True,
            colorbar=dict(title="Emission<br>Rate"),
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            width=800, height=700,
            template="plotly_dark",
        )

    if save_path:
        fig.write_html(save_path)

    return fig


def plot_source_evolution(
    source_maps: dict,
    save_path: Optional[str] = None,
):
    n_times = len(source_maps)
    fig = make_subplots(
        rows=1, cols=n_times,
        subplot_titles=[f"t = {t:.2f}" for t in source_maps.keys()],
    )

    lat = np.linspace(DELHI_LAT_RANGE[0], DELHI_LAT_RANGE[1], 50)
    lon = np.linspace(DELHI_LON_RANGE[0], DELHI_LON_RANGE[1], 50)

    for i, (t, field) in enumerate(source_maps.items(), 1):
        fig.add_trace(go.Heatmap(
            z=field.T, x=lon, y=lat,
            colorscale="Hot", reversescale=True,
            showscale=(i == n_times),
        ), row=1, col=i)

    fig.update_layout(
        title="Source Term Evolution S(x, y, t)",
        width=400 * n_times, height=500,
        template="plotly_dark",
    )

    if save_path:
        fig.write_html(save_path)

    return fig
