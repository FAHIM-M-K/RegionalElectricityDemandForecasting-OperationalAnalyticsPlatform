"""
ERCOT Operational Grid What-If Scenario Stress Testing & Simulation Engine
Interactive thermodynamic perturbation analysis (exclusively in Streamlit; reporting dashboards live in Power BI).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def render_scenario_simulator_view(df_forecast):
    """
    Renders the dynamic What-If Grid Stress Simulator.
    Simulates operational reserve margins and peak demand under severe weather anomalies.
    """
    st.markdown("### Operational What-If Grid Stress Testing & Scenario Simulator")
    st.caption("Live mathematical stress simulation of ERCOT dispatch reserves under extreme weather perturbations.")

    if df_forecast is None or len(df_forecast) == 0:
        st.warning("No baseline forecast data available for simulation.")
        return

    # Select baseline day from forecast
    df = df_forecast.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    if "predicted_demand_mw" not in df.columns:
        df["predicted_demand_mw"] = df.get("day_ahead_forecast_mw", df.get("actual_demand_mw", 50000.0))

    # Take most recent 48 hours as simulation baseline
    df_sim = df.tail(48).reset_index(drop=True)
    baseline_temp = df_sim.get("weighted_temperature_f", pd.Series([78.0] * len(df_sim)))

    # 1. Preset Scenarios and Interactive Controls
    st.markdown("##### 1. Define Severe Stress Perturbations")
    
    sc_col1, sc_col2 = st.columns([1, 2])
    
    with sc_col1:
        scenario_preset = st.selectbox(
            "Load Disaster Preset:",
            [
                "Custom Perturbation",
                "Severe Summer Heatwave (+8°F, -10% Solar)",
                "Catastrophic Summer Super-Peak (+14°F, -25% Solar)",
                "Deep Winter Freeze (-20°F Polar Vortex)",
                "Solar Cloud Intermittency (-45% Solar Drop)"
            ],
            index=0
        )

    # Set default slider values based on preset
    if scenario_preset == "Severe Summer Heatwave (+8°F, -10% Solar)":
        def_temp_delta = 8.0
        def_solar_drop = 10.0
        def_base_growth = 2.0
    elif scenario_preset == "Catastrophic Summer Super-Peak (+14°F, -25% Solar)":
        def_temp_delta = 14.0
        def_solar_drop = 25.0
        def_base_growth = 4.0
    elif scenario_preset == "Deep Winter Freeze (-20°F Polar Vortex)":
        def_temp_delta = -20.0
        def_solar_drop = 15.0
        def_base_growth = 3.0
    elif scenario_preset == "Solar Cloud Intermittency (-45% Solar Drop)":
        def_temp_delta = 0.0
        def_solar_drop = 45.0
        def_base_growth = 0.0
    else:
        def_temp_delta = 0.0
        def_solar_drop = 0.0
        def_base_growth = 0.0

    with sc_col2:
        c_slider1, c_slider2, c_slider3 = st.columns(3)
        with c_slider1:
            temp_delta = st.slider(
                "Temperature Anomaly (°F)",
                min_value=-30.0,
                max_value=20.0,
                value=float(def_temp_delta),
                step=1.0,
                help="Temperature shift applied to all Texas population centers."
            )
        with c_slider2:
            solar_loss_pct = st.slider(
                "Solar Generation Loss (%)",
                min_value=0.0,
                max_value=60.0,
                value=float(def_solar_drop),
                step=5.0,
                help="Grid-scale solar photovoltaic loss due to cloud cover / storms."
            )
        with c_slider3:
            load_growth_pct = st.slider(
                "Industrial/Data Center Ramp (%)",
                min_value=-10.0,
                max_value=15.0,
                value=float(def_base_growth),
                step=1.0,
                help="Large industrial / crypto / data center load fluctuation."
            )

    # 2. Dynamic Simulation Engine
    # Non-linear thermal response physics derived from ERCOT empirical modeling:
    # Baseline balance point = 55.2 deg F
    # Cooling response: ~1,680 MW per deg F above 55.2 deg F with quadratic acceleration in 95-105 deg F range
    # Heating response: ~640 MW per deg F below 55.2 deg F
    sim_rows = []
    installed_capacity_mw = 86500.0  # ERCOT peak available thermal + renewable capacity

    for idx, row in df_sim.iterrows():
        base_mw = float(row["predicted_demand_mw"])
        base_t = float(baseline_temp.iloc[idx]) if idx < len(baseline_temp) else 78.0
        new_t = base_t + temp_delta

        # Calculate thermal MW shift
        delta_mw = 0.0
        if new_t > 55.2:
            cooling_deg_orig = max(0.0, base_t - 55.2)
            cooling_deg_new = max(0.0, new_t - 55.2)
            # Quadratic ramp for extreme heat (>85 deg F)
            heat_factor = 1.0 + 0.03 * max(0.0, new_t - 85.0)
            delta_mw += (cooling_deg_new - cooling_deg_orig) * 1680.0 * heat_factor
        else:
            heating_deg_orig = max(0.0, 55.2 - base_t)
            heating_deg_new = max(0.0, 55.2 - new_t)
            # Electric resistive heating surge below 32 deg F
            cold_factor = 1.0 + 0.04 * max(0.0, 32.0 - new_t)
            delta_mw += (heating_deg_new - heating_deg_orig) * 640.0 * cold_factor

        # Solar deficit load transfer to grid
        hour = int(row.get("hour_local", idx % 24))
        is_daylight = 7 <= hour <= 19
        solar_deficit_mw = (solar_loss_pct / 100.0) * 12500.0 * np.sin((hour - 7) * np.pi / 12) if is_daylight else 0.0

        # Industrial baseline shift
        ind_mw = base_mw * (load_growth_pct / 100.0)

        simulated_mw = max(15000.0, base_mw + delta_mw + max(0.0, solar_deficit_mw) + ind_mw)
        sim_rows.append({
            "timestamp_utc": row["timestamp_utc"],
            "timestamp_local": row.get("timestamp_local", row["timestamp_utc"]),
            "hour_local": hour,
            "baseline_forecast_mw": base_mw,
            "simulated_demand_mw": simulated_mw,
            "net_delta_mw": simulated_mw - base_mw,
            "reserve_margin_mw": installed_capacity_mw - simulated_mw,
        })

    df_result = pd.DataFrame(sim_rows)

    # 3. Operational Reserves & Risk KPI Strip
    peak_baseline = df_result["baseline_forecast_mw"].max()
    peak_simulated = df_result["simulated_demand_mw"].max()
    net_peak_surge = peak_simulated - peak_baseline
    min_reserve = df_result["reserve_margin_mw"].min()

    # Determine ERCOT Energy Emergency Alert (EEA) Level
    if min_reserve < 1000.0:
        alert_state = "EEA LEVEL 3 (ROLLING OUTAGE RISK)"
        alert_color = "#dc2626"
        banner_class = "alert-banner-danger"
    elif min_reserve < 2300.0:
        alert_state = "EEA LEVEL 2 (WATCH / VOLUNTARY CURTAILMENT)"
        alert_color = "#ea580c"
        banner_class = "alert-banner-danger"
    elif min_reserve < 3000.0:
        alert_state = "EEA LEVEL 1 (DISPATCH CONSERVATION ADVISORY)"
        alert_color = "#d97706"
        banner_class = "alert-banner-danger"
    else:
        alert_state = "NORMAL (ADEQUATE OPERATING RESERVES)"
        alert_color = "#16a34a"
        banner_class = "alert-banner-normal"

    st.markdown(f"""
    <div class="{banner_class}">
        <strong>Simulated Grid Alert:</strong> {alert_state} — Minimum Operating Reserve: <strong>{min_reserve:,.0f} MW</strong>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Simulated Peak Demand</div>
            <div class="metric-value">{peak_simulated:,.0f} <span style="font-size:0.9rem;font-weight:500;">MW</span></div>
            <div class="metric-sub">Baseline: {peak_baseline:,.0f} MW</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Peak Demand Surge</div>
            <div class="metric-value" style="color:{'#dc2626' if net_peak_surge > 0 else '#16a34a'};">
                {net_peak_surge:+,.0f} <span style="font-size:0.9rem;font-weight:500;">MW</span>
            </div>
            <div class="metric-sub">{((peak_simulated - peak_baseline)/peak_baseline)*100:+.1f}% vs Day-Ahead plan</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Minimum Reserve Cushion</div>
            <div class="metric-value" style="color:{alert_color};">
                {min_reserve:,.0f} <span style="font-size:0.9rem;font-weight:500;">MW</span>
            </div>
            <div class="metric-sub">Capacity headroom: {(min_reserve/installed_capacity_mw)*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        # Estimated Ancillary peaker dispatch cost surge ($9,000/MWh cap during scarcity)
        scarcity_mwh = df_result.loc[df_result["reserve_margin_mw"] < 2500.0, "simulated_demand_mw"].sum()
        est_cost_impact = (scarcity_mwh * 150.0) / 1e6
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Est. Peaker Dispatch Cost</div>
            <div class="metric-value">${est_cost_impact:.1f}M</div>
            <div class="metric-sub">Thermal peaking premium</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Comparative Dynamic Curve Chart
    st.markdown("##### 2. Baseline Day-Ahead Forecast vs Simulated Stress Horizon")

    fig_sim = go.Figure()

    # Installed Grid Capacity Ceiling Line
    fig_sim.add_hline(
        y=installed_capacity_mw,
        line_dash="dot",
        line_color="#dc2626",
        line_width=2,
        annotation_text=f"Total Available Grid Capacity ({installed_capacity_mw:,.0f} MW)",
        annotation_position="bottom right",
        annotation_font=dict(color="#dc2626", size=10)
    )

    # Baseline Forecast Line (Slate Grey)
    fig_sim.add_trace(go.Scatter(
        x=df_result["timestamp_local"],
        y=df_result["baseline_forecast_mw"],
        mode="lines",
        name="Baseline Day-Ahead Forecast",
        line=dict(color="#94a3b8", width=2, dash="dash"),
    ))

    # Simulated Stress Curve (Cobalt / Red)
    curve_color = "#dc2626" if peak_simulated >= 78000.0 else "#2563eb"
    fig_sim.add_trace(go.Scatter(
        x=df_result["timestamp_local"],
        y=df_result["simulated_demand_mw"],
        mode="lines",
        name="Simulated Stress Load",
        line=dict(color=curve_color, width=3),
        fill='tonexty',
        fillcolor='rgba(220, 38, 38, 0.08)' if net_peak_surge > 0 else 'rgba(37, 99, 235, 0.08)'
    ))

    fig_sim.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin=dict(l=40, r=20, t=20, b=30),
        height=420,
        font=dict(family="Inter, sans-serif", size=11, color="#334155"),
        xaxis=dict(title="", gridcolor="#f1f5f9", linecolor="#cbd5e1"),
        yaxis=dict(title="Demand (MW)", gridcolor="#f1f5f9", linecolor="#cbd5e1", tickformat=","),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig_sim, use_container_width=True)

    # Note
    st.info("Institutional Power BI Reporting Suite hosts the official Production Dashboards (Executive Dispatch, Grid Stress & Growth, Thermodynamic Non-linear Response, and Texas 8-Zone Spatial Map).")
