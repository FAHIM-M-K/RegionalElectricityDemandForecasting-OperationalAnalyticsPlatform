import os
import sys
import streamlit as st

# Configure wide layout and page metadata
st.set_page_config(
    page_title="ERCOT Operational Grid What-If Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.data_loader import load_operational_forecast_data
from app.views.scenario_simulator import render_scenario_simulator_view

# Load Custom CSS Styling
css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------- Sidebar Configuration ----------------
with st.sidebar:
    st.markdown("#### Grid Dispatch Controls")
    st.caption("Operational What-If Stress Testing Simulator.")
    st.divider()

    selected_region = st.selectbox(
        "Balancing Authority",
        ["ERCO (Electric Reliability Council of Texas)"],
        index=0
    )

    installed_cap_gw = st.number_input(
        "Total Available Grid Capacity (GW)",
        min_value=70.0,
        max_value=110.0,
        value=86.5,
        step=0.5,
        help="Effective thermal, nuclear, wind, and solar dispatch capacity."
    )

    st.divider()

    # Load Baseline Data
    with st.spinner("Connecting to Gold Forecast Mart..."):
        df_forecast, data_source = load_operational_forecast_data()

    st.markdown("##### Connection Status")
    if "Databricks" in data_source:
        st.markdown(f'<span class="status-pill-success">{data_source}</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="status-pill-neutral">{data_source}</span>', unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    **Production Dashboards:** Power BI  
    - Page 1: Executive Dispatch  
    - Page 2: Grid Stress & Growth  
    - Page 3: Thermodynamic Response  
    - Page 4: ERCOT Zonal Map  

    **Streamlit Role:** Real-Time Simulation Engine  
    *(Zero triggers; runs dynamic in-memory what-if perturbations).*
    """)

# ---------------- Main Layout ----------------
st.title("ERCOT Operational Stress & What-If Simulation Engine")
st.caption("Dynamic thermodynamic perturbator and reserve margin estimator (Zero-trigger interactive simulator).")

# Render Simulator
render_scenario_simulator_view(df_forecast)
