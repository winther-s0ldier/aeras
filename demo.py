import streamlit as st
import numpy as np
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Configuration
st.set_page_config(page_title="Aeras PINN Dashboard", layout="wide")
st.title("Aeras: Physics-Informed Neural Network (PINN) for Air Quality")

CHECKPOINTS_DIR = Path("checkpoints")

# Load Data
@st.cache_data
def load_data():
    def load_json(name):
        try:
            with open(CHECKPOINTS_DIR / name, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    fwd_eval = load_json("evaluation_results.json")
    inv_eval = load_json("inverse_evaluation_results.json")
    lstm_eval = load_json("lstm_metrics.json")
    xgb_eval = load_json("xgboost_metrics.json")

    # Load Source Maps
    try:
        source_maps = np.load(CHECKPOINTS_DIR / "source_maps.npz")
        maps = {k: source_maps[k] for k in source_maps.files}
    except FileNotFoundError:
        maps = {}

    return fwd_eval, inv_eval, lstm_eval, xgb_eval, maps

fwd_eval, inv_eval, lstm_eval, xgb_eval, source_maps = load_data()

# Sidebar Navigation
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Select Model Mode:", [
    "Inverse Model (v11) - Source Discovery", 
    "Forward Model (v10) - Baseline",
    "Model Comparison (PINN vs LSTM/XGBoost)"
])

if mode == "Forward Model (v10) - Baseline":
    st.header("Forward Model Evaluation")
    st.markdown("""
    The **v10 Forward Model** learned to propagate pollution strictly using atmospheric physics (diffusion and wind) but was forced to assume that **no new emission sources ($S=0$)** existed.
    This caused it to severely underpredict massive anomalous events like Diwali fireworks.
    """)
    
    if fwd_eval:
        st.subheader("Model Accuracy (PM2.5)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Overall R²", f"{fwd_eval.get('full_test', {}).get('pm25', {}).get('r2', 0):.4f}")
        col2.metric("Diwali 2019 R²", f"{fwd_eval.get('diwali_2019', {}).get('pm25', {}).get('r2', 0):.4f}")
        col3.metric("Winter 2019 R²", f"{fwd_eval.get('winter_2019', {}).get('pm25', {}).get('r2', 0):.4f}")
        
        st.warning("Notice how the Diwali and Winter R² scores are very low (or negative). The model fails to predict these spikes because it doesn't know the sources exist!")
    else:
        st.error("evaluation_results.json not found in checkpoints directory.")

elif mode == "Inverse Model (v11) - Source Discovery":
    st.header("Inverse Model: Dynamic Source Discovery")
    st.markdown("""
    The **v11 Inverse Model** uses a secondary neural network (`SourceNet`) to dynamically solve the inverse problem.
    When the physics equations don't add up, `SourceNet` draws a map of the hidden emission sources (like Diwali fireworks) that caused the anomaly.
    """)

    if source_maps:
        st.subheader("SourceNet Emission Heatmaps")
        st.markdown("Compare the dynamic emissions predicted by the neural network against the static European Commission (EDGAR) baseline.")
        
        time_points = [k for k in source_maps.keys() if k.startswith("t_")]
        if not time_points:
            st.error("No temporal maps found in source_maps.npz")
        else:
            time_str = st.select_slider("Select Time Step (Normalized):", options=sorted(time_points))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### PINN Predicted Source (SourceNet)")
                pred_map = source_maps[time_str]
                fig_pred = px.imshow(pred_map, color_continuous_scale="magma", zmin=0, zmax=1)
                fig_pred.update_layout(margin=dict(l=0, r=0, b=0, t=0))
                fig_pred.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
                st.plotly_chart(fig_pred, use_container_width=True)
                
            with col2:
                st.markdown("### EDGAR Baseline (Static 2018)")
                if "edgar" in source_maps:
                    edgar_map = source_maps["edgar"]
                    fig_edgar = px.imshow(edgar_map, color_continuous_scale="magma", zmin=0, zmax=1)
                    fig_edgar.update_layout(margin=dict(l=0, r=0, b=0, t=0))
                    fig_edgar.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
                    st.plotly_chart(fig_edgar, use_container_width=True)
                else:
                    st.warning("EDGAR baseline map not found in source_maps.npz")
                    
        st.info("**Why don't they match perfectly?** EDGAR represents a static annual average (e.g. permanent factories). The PINN is dynamically discovering short-lived anomalies like Diwali fireworks which occur in residential zones where EDGAR expects zero emissions.")
        
        if inv_eval:
            st.subheader("Inverse Evaluation Metrics")
            t_key = f"t={time_str.replace('t_', '')}"
            if t_key in inv_eval:
                m_col1, m_col2 = st.columns(2)
                m_col1.metric("Location Error (km)", f"{inv_eval[t_key].get('location_error_km', 0):.1f}")
                m_col2.metric("Magnitude Error", f"{inv_eval[t_key].get('magnitude_error', 0)*100:.1f}%")

    else:
        st.error("source_maps.npz not found in checkpoints directory.")

elif mode == "Model Comparison (PINN vs LSTM/XGBoost)":
    st.header("Baseline Comparison (R² Scores)")
    st.markdown("""
    How does the Physics-Informed Neural Network compare to purely data-driven baselines like XGBoost and LSTM?
    Because purely data-driven models can 'memorize' spikes from recent time steps, they often score higher on standard metrics during anomalies. 
    However, they learn **no actual physics**, meaning they cannot explain *why* the spike happened. The PINN forces physical consistency, which allows it to discover the hidden sources!
    """)
    
    if lstm_eval and xgb_eval and fwd_eval:
        import pandas as pd
        
        splits = ["random", "diwali", "winter"]
        data = []
        
        for split in splits:
            # Safely get R2 scores
            pinn_r2 = fwd_eval.get(split + "_2019", {}).get("pm25", {}).get("r2", 0) if split != "random" else fwd_eval.get("random", {}).get("pm25", {}).get("r2", 0)
            xgb_r2 = xgb_eval.get("test_denormalised_1h", {}).get(split, {}).get("r2", 0)
            lstm_r2 = lstm_eval.get("test_denormalised_1h", {}).get(split, {}).get("r2", 0)
            
            data.append({"Split": split.capitalize(), "Model": "PINN (Physics)", "R² Score": pinn_r2})
            data.append({"Split": split.capitalize(), "Model": "XGBoost", "R² Score": xgb_r2})
            data.append({"Split": split.capitalize(), "Model": "LSTM", "R² Score": lstm_r2})
            
        df = pd.DataFrame(data)
        
        fig = px.bar(df, x="Split", y="R² Score", color="Model", barmode="group",
                     title="Model Comparison across Test Splits (PM2.5)",
                     color_discrete_map={"PINN (Physics)": "#EF553B", "XGBoost": "#00CC96", "LSTM": "#636EFA"})
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("Notice how the PINN struggles heavily during Diwali and Winter (negative R²). This spectacular failure of the forward model is exactly what motivated the Inverse Model (v11) to discover the hidden sources!")
    else:
        st.error("Missing metric JSON files to build the comparison chart.")
