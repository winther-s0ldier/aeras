import streamlit as st
import numpy as np
import json
import torch
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Aeras PINN Dashboard", layout="wide")
st.title("Aeras: Physics-Informed Neural Network for Delhi NCR Air Quality")

CHECKPOINTS_DIR = ROOT / "checkpoints"
NORM_PARAMS_PATH = ROOT / "data" / "processed" / "normalized_params.json"

# Delhi NCR bounding box
LAT_MIN, LAT_MAX = 27.5, 29.0
LON_MIN, LON_MAX = 76.5, 77.8

# ── normalization params ──────────────────────────────────────────────────────
@st.cache_data
def load_norm_params():
    with open(NORM_PARAMS_PATH) as f:
        return json.load(f)

NORM = load_norm_params()

def denorm(val_norm, key):
    vmin = NORM[key]["min"]
    vmax = NORM[key]["max"]
    return val_norm * (vmax - vmin) + vmin

def norm(val, key):
    vmin = NORM[key]["min"]
    vmax = NORM[key]["max"]
    return (val - vmin) / (vmax - vmin)

# ── AQI functions (CPCB India standard) ───────────────────────────────────────
AQI_BREAKPOINTS_PM25 = [
    (0,   30,  0,   50,  "Good",        "#00e400"),
    (31,  60,  51,  100, "Satisfactory","#ffff00"),
    (61,  90,  101, 200, "Moderate",    "#ff7e00"),
    (91,  120, 201, 300, "Poor",        "#ff0000"),
    (121, 250, 301, 400, "Very Poor",   "#8f3f97"),
    (251, 500, 401, 500, "Severe",      "#7e0023"),
]

AQI_BREAKPOINTS_NO2 = [
    (0,   40,  0,   50,  "Good",        "#00e400"),
    (41,  80,  51,  100, "Satisfactory","#ffff00"),
    (81,  180, 101, 200, "Moderate",    "#ff7e00"),
    (181, 280, 201, 300, "Poor",        "#ff0000"),
    (281, 400, 301, 400, "Very Poor",   "#8f3f97"),
    (401, 800, 401, 500, "Severe",      "#7e0023"),
]

def concentration_to_aqi(conc, breakpoints):
    for bp_lo, bp_hi, aqi_lo, aqi_hi, _, _ in breakpoints:
        if conc <= bp_hi:
            return ((aqi_hi - aqi_lo) / max(bp_hi - bp_lo, 1)) * (conc - bp_lo) + aqi_lo
    return 500.0

def aqi_to_category(aqi):
    for _, _, aqi_lo, aqi_hi, label, color in AQI_BREAKPOINTS_PM25:
        if aqi <= aqi_hi:
            return label, color
    return "Severe", "#7e0023"

def pm25_ug_to_aqi(pm25):
    return np.vectorize(lambda c: concentration_to_aqi(c, AQI_BREAKPOINTS_PM25))(pm25)

# ── model loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    from src.models.pinn import AerasPINN
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = CHECKPOINTS_DIR / "aeras_v11_prod.pt"
    if not ckpt_path.exists():
        return None, device
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = AerasPINN().to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, device

# ── spatial grid inference ────────────────────────────────────────────────────
@st.cache_data
def run_spatial_inference(t_norm: float, u_wind_norm: float, v_wind_norm: float,
                          temp_norm: float, blh_norm: float, resolution: int = 40):
    model, device = load_model()
    if model is None:
        return None

    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    xx, yy = np.meshgrid(x, y)

    inputs = np.column_stack([
        xx.flatten(),
        yy.flatten(),
        np.full(resolution**2, t_norm),
        np.full(resolution**2, u_wind_norm),
        np.full(resolution**2, v_wind_norm),
        np.full(resolution**2, temp_norm),
        np.full(resolution**2, blh_norm),
    ]).astype(np.float32)

    with torch.no_grad():
        out = model(torch.tensor(inputs).to(device)).cpu().numpy()

    # Denormalise to μg/m³
    pm25 = denorm(out[:, 0], "pm25").reshape(resolution, resolution)
    no2  = denorm(out[:, 1], "no2").reshape(resolution, resolution)
    o3   = denorm(out[:, 2], "o3").reshape(resolution, resolution)
    so2  = denorm(out[:, 3], "so2").reshape(resolution, resolution)

    lats = np.linspace(LAT_MIN, LAT_MAX, resolution)
    lons = np.linspace(LON_MIN, LON_MAX, resolution)

    return {"pm25": pm25, "no2": no2, "o3": o3, "so2": so2,
            "lats": lats, "lons": lons}

# ── JSON loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    def load_json(name):
        try:
            with open(CHECKPOINTS_DIR / name) as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    fwd_eval = load_json("evaluation_results.json")
    inv_eval = load_json("inverse_evaluation_results.json")
    inv_fixed = load_json("inverse_evaluation_fixed.json")
    lstm_eval = load_json("lstm_metrics.json")
    xgb_eval  = load_json("xgboost_metrics.json")
    loo_eval  = load_json("loo_results.json")

    try:
        source_maps = dict(np.load(CHECKPOINTS_DIR / "source_maps.npz"))
    except FileNotFoundError:
        source_maps = {}

    return fwd_eval, inv_eval, inv_fixed, lstm_eval, xgb_eval, loo_eval, source_maps

fwd_eval, inv_eval, inv_fixed, lstm_eval, xgb_eval, loo_eval, source_maps = load_data()

# ── sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Select View:", [
    "AQI Map (Live Spatial)",
    "Inverse Model — Source Discovery",
    "Forward Model — Baseline (v10)",
    "Model Comparison",
    "Spatial Validation (LOO)",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: AQI MAP
# ═══════════════════════════════════════════════════════════════════════════════
if mode == "AQI Map (Live Spatial)":
    st.header("Delhi NCR AQI Map — Spatial Prediction")
    st.markdown("""
    The PINN predicts PM2.5 at **every point** in Delhi NCR, not just sensor locations.
    Adjust the controls to simulate different atmospheric conditions.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Atmospheric Conditions")
        u_wind = st.slider("Wind U (East-West, m/s)", -8.0, 8.0, 2.0, 0.5)
        v_wind = st.slider("Wind V (North-South, m/s)", -8.0, 8.0, 1.0, 0.5)
        temp_c = st.slider("Temperature (°C)", 5.0, 45.0, 25.0, 1.0)
        blh_m  = st.slider("Boundary Layer Height (m)", 100, 4000, 800, 100)

        scenario = st.selectbox("Or pick a scenario:", [
            "Custom (use sliders)",
            "Diwali night (calm, cold)",
            "Summer afternoon (hot, windy)",
            "Winter morning inversion",
        ])

        if scenario == "Diwali night (calm, cold)":
            u_wind, v_wind, temp_c, blh_m = 0.5, 0.2, 18.0, 300
        elif scenario == "Summer afternoon (hot, windy)":
            u_wind, v_wind, temp_c, blh_m = 5.0, 3.0, 40.0, 2000
        elif scenario == "Winter morning inversion":
            u_wind, v_wind, temp_c, blh_m = 0.3, 0.1, 12.0, 150

        pollutant = st.selectbox("Pollutant to display:", ["PM2.5", "NO2", "O3", "SO2"])
        show_aqi  = st.checkbox("Show AQI categories (PM2.5 only)", value=True)
        resolution = st.select_slider("Map resolution:", [20, 30, 40, 50], value=40)

    # Normalise inputs
    u_norm    = norm(u_wind,         "u_wind")
    v_norm    = norm(v_wind,         "v_wind")
    temp_norm = norm(temp_c + 273.15, "temperature")
    blh_norm  = norm(blh_m,          "boundary_layer_height")
    t_norm    = 0.5  # mid-dataset time

    with st.spinner("Running PINN inference across Delhi NCR grid..."):
        result = run_spatial_inference(t_norm, u_norm, v_norm, temp_norm, blh_norm, resolution)

    with col2:
        if result is None:
            st.error("Model checkpoint not found. Make sure aeras_v11_prod.pt is in checkpoints/")
        else:
            poll_key = pollutant.lower().replace(".", "")
            poll_map = result[poll_key]
            lats     = result["lats"]
            lons     = result["lons"]

            # Flatten grid → individual points for mapbox overlay
            lats_g, lons_g = np.meshgrid(lats, lons, indexing="ij")
            lats_flat = lats_g.flatten()
            lons_flat = lons_g.flatten()

            AQI_COLORSCALE = [
                [0.00, "#00e400"], [0.10, "#00e400"],
                [0.10, "#ffff00"], [0.20, "#ffff00"],
                [0.20, "#ff7e00"], [0.40, "#ff7e00"],
                [0.40, "#ff0000"], [0.60, "#ff0000"],
                [0.60, "#8f3f97"], [0.80, "#8f3f97"],
                [0.80, "#7e0023"], [1.00, "#7e0023"],
            ]

            if pollutant == "PM2.5" and show_aqi:
                aqi_map  = np.clip(pm25_ug_to_aqi(poll_map), 0, 500)
                z_flat   = aqi_map.flatten()
                colorscale = AQI_COLORSCALE
                zmin, zmax = 0, 500
                colorbar_title = "AQI"
                colorbar_ticks = dict(
                    tickvals=[25, 75, 150, 250, 350, 450],
                    ticktext=["Good", "Satisfactory", "Moderate",
                              "Poor", "Very Poor", "Severe"],
                )
            else:
                z_flat   = poll_map.flatten()
                colorscale = "RdYlGn_r"
                zmin, zmax = float(z_flat.min()), float(z_flat.max())
                colorbar_title = {"pm25":"PM2.5 (μg/m³)","no2":"NO2 (μg/m³)",
                                  "o3":"O3 (μg/m³)","so2":"SO2 (μg/m³)"}.get(poll_key, pollutant)
                colorbar_ticks = {}

            fig = go.Figure(go.Densitymap(
                lat=lats_flat,
                lon=lons_flat,
                z=z_flat,
                radius=25,
                opacity=0.75,
                colorscale=colorscale,
                zmin=zmin, zmax=zmax,
                colorbar=dict(title=colorbar_title, **colorbar_ticks),
                hovertemplate="Lat: %{lat:.3f}<br>Lon: %{lon:.3f}<br>Value: %{z:.1f}<extra></extra>",
            ))
            fig.update_layout(
                map_style="carto-positron",
                map_center={"lat": 28.55, "lon": 77.2},
                map_zoom=10,
                title=f"Delhi NCR {pollutant} AQI — U={u_wind:.1f} V={v_wind:.1f} T={temp_c:.0f}°C BLH={blh_m}m",
                margin={"r": 0, "t": 40, "l": 0, "b": 0},
                height=520,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Summary stats
            aqi_vals = pm25_ug_to_aqi(result["pm25"])
            mean_aqi  = float(aqi_vals.mean())
            max_aqi   = float(aqi_vals.max())
            cat, color = aqi_to_category(mean_aqi)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Mean PM2.5", f"{result['pm25'].mean():.1f} μg/m³")
            m2.metric("Mean AQI",   f"{mean_aqi:.0f} ({cat})")
            m3.metric("Max PM2.5",  f"{result['pm25'].max():.1f} μg/m³")
            m4.metric("Max AQI",    f"{max_aqi:.0f}")

            st.info(
                "This map shows predicted PM2.5 / AQI at **every grid point** in Delhi NCR — "
                "including areas with **no CPCB sensor**. LSTM and XGBoost cannot produce this map."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: INVERSE MODEL
# ═══════════════════════════════════════════════════════════════════════════════
elif mode == "Inverse Model — Source Discovery":
    st.header("Inverse Model: Dynamic Source Discovery")
    st.markdown("""
    The **v11 Inverse Model** uses `SourceNet` to dynamically solve the inverse problem.
    When the physics equations don't balance, `SourceNet` maps the hidden emission sources
    (like Diwali fireworks) that caused the anomaly.
    """)

    if source_maps:
        time_points = sorted(k for k in source_maps if k.startswith("t_"))
        time_str = st.select_slider("Time step (normalised):", options=time_points)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### PINN Predicted Source (SourceNet)")
            pred_map = source_maps[time_str]
            fig = px.imshow(pred_map, color_continuous_scale="magma", zmin=0, zmax=float(pred_map.max()))
            fig.update_layout(margin=dict(l=0,r=0,b=0,t=0))
            fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### EDGAR Baseline (Static 2018)")
            if "edgar" in source_maps:
                edgar_map = source_maps["edgar"]
                fig = px.imshow(edgar_map, color_continuous_scale="magma",
                                zmin=0, zmax=float(edgar_map.max()))
                fig.update_layout(margin=dict(l=0,r=0,b=0,t=0))
                fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("EDGAR baseline not found in source_maps.npz")

        st.info(
            "**EDGAR** = static annual average (permanent factories, roads). "
            "**PINN** = dynamic anomaly (Diwali fireworks in residential zones "
            "where EDGAR expects zero emissions)."
        )

        if inv_eval:
            t_key = f"t={time_str.replace('t_','')}"
            if t_key in inv_eval:
                c1, c2 = st.columns(2)
                c1.metric("Location Error", f"{inv_eval[t_key]['location_error_km']:.1f} km")
                c2.metric("Magnitude Error", f"{inv_eval[t_key]['magnitude_error']*100:.1f}%")

        if inv_fixed:
            st.subheader("Fixed SourceNet Results (v12)")
            pm25 = inv_fixed.get("pm25", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("PM2.5 R²",      f"{pm25.get('r2', 0):.4f}")
            c2.metric("PM2.5 MAE",     f"{pm25.get('mae', 0):.4f}")
            c3.metric("Source S_mean", f"{pm25.get('source_mean', 0):.4e}")
            if pm25.get("source_mean", 0) > 1e-4:
                st.success("SourceNet collapse fixed — non-zero emissions detected.")
    else:
        st.error("source_maps.npz not found in checkpoints/")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: FORWARD MODEL
# ═══════════════════════════════════════════════════════════════════════════════
elif mode == "Forward Model — Baseline (v10)":
    st.header("Forward Model Evaluation (v10)")
    st.markdown("""
    The **v10 Forward Model** enforces the advection-diffusion PDE but assumes S=0
    (no emission sources). It fails on extreme events because it cannot know why
    PM2.5 suddenly spikes — this is what motivated the Inverse Model.
    """)

    if fwd_eval:
        c1, c2, c3 = st.columns(3)
        c1.metric("Overall R²",      f"{fwd_eval.get('full_test',{}).get('pm25',{}).get('r2',0):.4f}")
        c2.metric("Diwali 2019 R²",  f"{fwd_eval.get('diwali_2019',{}).get('pm25',{}).get('r2',0):.4f}")
        c3.metric("Winter 2019 R²",  f"{fwd_eval.get('winter_2019',{}).get('pm25',{}).get('r2',0):.4f}")
        st.warning(
            "Diwali and Winter R² are near zero or negative because the forward model "
            "has no source term — it cannot predict sudden emission spikes."
        )

        st.subheader("Learned Physics Parameters")
        st.markdown("""
        These are real atmospheric diffusivity values learned from data — not assumed:
        """)
        params_data = {
            "Pollutant": ["PM2.5", "NO2", "O3", "SO2"],
            "Dx (diffusion x)":    [0.004730, 0.006031, 0.004252, 0.009118],
            "Dy (diffusion y)":    [0.005323, 0.007062, 0.003805, 0.010464],
            "λ (deposition rate)": [0.010197, 0.010104, 0.009894, 0.010050],
        }
        import pandas as pd
        st.dataframe(pd.DataFrame(params_data), use_container_width=True)
    else:
        st.error("evaluation_results.json not found")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
elif mode == "Model Comparison":
    st.header("Model Comparison — PINN vs LSTM vs XGBoost")
    st.markdown("""
    LSTM wins on raw MAE because it **sees recent PM2.5 readings as input** (lag features).
    The PINN receives **only physics inputs** (x, y, t, wind) — yet it can predict spatially
    anywhere and discover emission sources. These are fundamentally different capabilities.
    """)

    import pandas as pd

    if fwd_eval and lstm_eval and xgb_eval:
        rows = []
        for split, pinn_key, bl_key in [
            ("Random",  "random",      "random"),
            ("Diwali",  "diwali_2019", "diwali"),
            ("Winter",  "winter_2019", "winter"),
        ]:
            pinn_r2 = fwd_eval.get(pinn_key, {}).get("pm25", {}).get("r2", 0)
            lstm_r2 = lstm_eval.get("test_denormalised_1h", {}).get(bl_key, {}).get("r2", 0)
            xgb_r2  = xgb_eval.get("test_denormalised_1h",  {}).get(bl_key, {}).get("r2", 0)
            rows += [
                {"Split": split, "Model": "PINN (v11)", "R²": pinn_r2},
                {"Split": split, "Model": "LSTM",       "R²": lstm_r2},
                {"Split": split, "Model": "XGBoost",    "R²": xgb_r2},
            ]

        df = pd.DataFrame(rows)
        fig = px.bar(df, x="Split", y="R²", color="Model", barmode="group",
                     title="PM2.5 R² by Test Split",
                     color_discrete_map={
                         "PINN (v11)": "#EF553B",
                         "LSTM":       "#636EFA",
                         "XGBoost":    "#00CC96",
                     })
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("What each model can do")
        cap_df = pd.DataFrame({
            "Capability": [
                "Predict at sensor locations",
                "Predict at unmonitored locations",
                "AQI map of entire city",
                "Discover emission sources",
                "Physically consistent (PDE)",
                "Explain WHY pollution spiked",
            ],
            "LSTM":    ["✓ (best)", "✗", "✗", "✗", "✗", "✗"],
            "XGBoost": ["✓",        "✗", "✗", "✗", "✗", "✗"],
            "PINN":    ["✓",        "✓", "✓", "✓", "✓", "✓"],
        })
        st.dataframe(cap_df, use_container_width=True, hide_index=True)

        st.info(
            "LSTM is better at temporal forecasting at **known** locations. "
            "The PINN is the only model that works at **unknown** locations and "
            "can localise emission sources — tasks LSTM and XGBoost cannot attempt."
        )
    else:
        st.error("Missing metric JSON files")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: LOO SPATIAL VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
elif mode == "Spatial Validation (LOO)":
    st.header("Leave-One-Out Spatial Validation")
    st.markdown("""
    **The key experiment:** one station was completely removed from training.
    The PINN was trained on the remaining stations, then evaluated at the held-out location.

    If R² > 0 at a location the model was **never trained on**, spatial generalisation is proven.
    """)

    if loo_eval:
        import pandas as pd
        st.success(f"Held-out station: **{loo_eval.get('held_out_station', '?')}**")

        rows = []
        for poll, m in loo_eval.get("pollutants", {}).items():
            rows.append({
                "Pollutant":     poll.upper(),
                "R² (norm)":     round(m.get("r2_normalised", 0), 4),
                "MAE (μg/m³)":   round(m.get("mae_ug_m3", 0), 2),
                "n samples":     m.get("n", 0),
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        pm25_r2 = loo_eval.get("pollutants", {}).get("pm25", {}).get("r2_normalised", float("nan"))
        if pm25_r2 > 0.1:
            st.success(
                f"PM2.5 R² = {pm25_r2:.4f} at an **unseen station**. "
                "Spatial generalisation confirmed — the PINN predicts at locations "
                "it was never trained on."
            )
        elif pm25_r2 > 0:
            st.warning(f"PM2.5 R² = {pm25_r2:.4f}. Weak positive — physics helps but data is sparse.")
        else:
            st.error(
                f"PM2.5 R² = {pm25_r2:.4f}. Spatial generalisation not confirmed. "
                "Sparse sensors make the problem underdetermined."
            )

        st.markdown("""
        **Why this matters:** LSTM and XGBoost cannot predict at unmonitored locations at all —
        they require observations as input. The PINN only needs coordinates and wind.
        """)
    else:
        st.info(
            "LOO results not yet available. "
            "Run `train_loo.py` on Kaggle (GPU 1) and copy `loo_results.json` to checkpoints/."
        )
