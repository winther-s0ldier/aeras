import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional


def plot_model_comparison_bar(
    results: Dict[str, Dict[str, float]],
    metric: str = "mae",
    title: str = "Model Comparison — MAE (μg/m³)",
    save_path: Optional[str] = None,
):
    models = list(results.keys())
    test_sets = list(next(iter(results.values())).keys())

    fig = go.Figure()
    colors = ["#00d4ff", "#ff6b6b", "#50fa7b", "#ffb86c", "#bd93f9"]

    for i, test_set in enumerate(test_sets):
        values = [results[m].get(test_set, {}).get(metric, 0) for m in models]
        fig.add_trace(go.Bar(
            name=test_set,
            x=models,
            y=values,
            marker_color=colors[i % len(colors)],
        ))

    fig.update_layout(
        title=title,
        barmode="group",
        template="plotly_dark",
        width=1000, height=500,
        yaxis_title=metric.upper(),
    )

    if save_path:
        fig.write_html(save_path)

    return fig


def plot_sparse_degradation(
    pinn_results: Dict[float, Dict[str, float]],
    lstm_results: Dict[float, Dict[str, float]],
    metric: str = "mae",
    save_path: Optional[str] = None,
):
    dropout_rates = sorted(pinn_results.keys())
    pinn_values = [pinn_results[r][metric] for r in dropout_rates]
    lstm_values = [lstm_results[r][metric] for r in dropout_rates]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[f"{r:.0%}" for r in dropout_rates],
        y=pinn_values,
        mode="lines+markers",
        name="aeras",
        line=dict(color="#00d4ff", width=3),
        marker=dict(size=10),
    ))

    fig.add_trace(go.Scatter(
        x=[f"{r:.0%}" for r in dropout_rates],
        y=lstm_values,
        mode="lines+markers",
        name="LSTM Baseline",
        line=dict(color="#ff6b6b", width=3, dash="dash"),
        marker=dict(size=10),
    ))

    fig.update_layout(
        title="Sparse Sensor Degradation — PINN vs LSTM",
        xaxis_title="Sensor Dropout Rate",
        yaxis_title=f"{metric.upper()} (μg/m³)",
        template="plotly_dark",
        width=800, height=500,
    )

    if save_path:
        fig.write_html(save_path)

    return fig
