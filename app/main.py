import os
import sys
import streamlit as st

# Configure wide layout and page metadata
st.set_page_config(
    page_title="ERCOT Operational Grid Forecast Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.data_loader import load_operational_forecast_data
from app.views.executive import render_executive_view
from app.views.analytics import render_analytics_view
from app.views.monitoring import render_monitoring_view

# Load Custom CSS Styling
css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------- Sidebar Configuration ----------------
with st.sidebar:
    st.markdown("### ⚡ Grid Dispatch Control")
    st.markdown("Operational electricity demand forecasting & capacity risk analytics.")
    st.divider()

    selected_region = st.selectbox(
        "Balancing Authority",
        ["ERCO (Electric Reliability Council of Texas)"],
        index=0
    )

    peak_threshold_mw = st.slider(
        "Peak Alert Threshold (MW)",
        min_value=60000,
        max_value=90000,
        value=75000,
        step=1000,
        help="Triggers critical dispatch warnings when forecasted load exceeds this value."
    )

    forecast_horizon = st.selectbox(
        "Forecast Horizon",
        ["Day-Ahead (48-Hour Horizon)", "Intraday (6-Hour Horizon)", "Week-Ahead (168-Hour Horizon)"],
        index=0
    )

    st.divider()

    # Load Data
    with st.spinner("Connecting to Gold Forecast Mart..."):
        df_forecast, data_source = load_operational_forecast_data()

    st.markdown("### 📡 Data Pipeline Status")
    if "Databricks" in data_source:
        st.markdown(f'<span class="badge-success">LIVE: {data_source}</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="badge-warning">OFFLINE: {data_source}</span>', unsafe_allow_html=True)
        st.caption("Using high-fidelity local ERCOT simulation. Set Databricks credentials in .env to stream live Unity Catalog tables.")

    st.divider()
    st.markdown("""
    **Model Champion:** `LightGBM Hybrid`  
    **Target:** Day-Ahead Load ($H=24..48$)  
    **EIA Lift:** **+3.58% Global**, **+7.37% Peak**
    """)

# ---------------- Main Dashboard Layout ----------------
st.title("ERCOT Regional Demand Forecasting Platform")
st.caption("Day-Ahead Operational Analytics & Institutional Benchmark Comparison Engine")

# Top Navigation Tabs
tab_exec, tab_analytics, tab_monitoring = st.tabs([
    "🚀 Executive Dispatch",
    "📊 Operational Analytics",
    "🛡️ Model Governance & Monitoring"
])

with tab_exec:
    render_executive_view(df_forecast, peak_threshold_mw)

with tab_analytics:
    render_analytics_view(df_forecast)

with tab_monitoring:
    render_monitoring_view(df_forecast)
