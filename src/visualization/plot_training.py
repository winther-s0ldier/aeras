import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import CHECKPOINTS_DIR


def plot_training_curves(
    history_path: str = None,
    save_path: Optional[str] = None,
):
    if history_path is None:
        history_path = CHECKPOINTS_DIR / "training_history.json"

    with open(history_path) as f:
        history = json.load(f)

    epochs = history["epoch"]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Total Loss", "Validation Loss",
            "PDE Residual", "Loss Components"
        ),
    )


    fig.add_trace(go.Scatter(
        x=epochs, y=history["train_loss"],
        mode="lines", name="Train Loss", line=dict(color="#00d4ff"),
    ), row=1, col=1)


    fig.add_trace(go.Scatter(
        x=epochs, y=history["val_loss"],
        mode="lines", name="Val Loss", line=dict(color="#ff6b6b"),
    ), row=1, col=2)


    fig.add_trace(go.Scatter(
        x=epochs, y=history["pde_residual"],
        mode="lines", name="PDE Residual", line=dict(color="#50fa7b"),
    ), row=2, col=1)

    fig.update_layout(
        title="aeras Training Progress",
        template="plotly_dark",
        width=1200, height=800,
        showlegend=False,
    )
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=2)
    fig.update_yaxes(type="log", row=2, col=1)

    if save_path:
        fig.write_html(save_path)

    return fig
