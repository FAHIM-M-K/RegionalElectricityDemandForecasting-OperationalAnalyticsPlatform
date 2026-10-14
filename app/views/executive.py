import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

def render_executive_view(df, peak_threshold_mw):
    st.subheader("⚡ Operational Grid Dispatch & Day-Ahead Forecast")
    st.caption("Real-time load tracking, 48-hour forward capacity forecasting, and operational peak risk alerts.")

    # Current time cutoff
    now = datetime.now()
    df_actuals = df[df["actual_demand_mw"].notna()].sort_values("timestamp_local")
    df_forecast = df[df["actual_demand_mw"].isna()].sort_values("timestamp_local")

    if len(df_forecast) == 0:
        # Fallback if all hours have actuals: treat last 48h as forecast
        cutoff = df["timestamp_local"].max() - timedelta(hours=48)
        df_actuals = df[df["timestamp_local"] <= cutoff]
        df_forecast = df[df["timestamp_local"] > cutoff]

    latest_actual = df_actuals.iloc[-1] if len(df_actuals) > 0 else df.iloc[0]
    next_48h_peak = df_forecast["predicted_demand_mw"].max() if len(df_forecast) > 0 else 0
    peak_row = df_forecast.loc[df_forecast["predicted_demand_mw"].idxmax()] if len(df_forecast) > 0 else None

    is_peak_alert = next_48h_peak >= peak_threshold_mw

    # 1. Operational Peak Risk Banner
    if is_peak_alert and peak_row is not None:
        st.markdown(f"""
        <div class="alert-banner-danger">
            <div>
                <span class="badge-danger" style="margin-right: 8px;">CRITICAL PEAK ALERT</span>
                <strong>High Demand Exceedance Warning:</strong> Forecasted load will peak at 
                <span style="font-size: 1.15rem; font-weight: 700; color: #fca5a5;">{next_48h_peak:,.0f} MW</span>
                on <strong>{peak_row['timestamp_local'].strftime('%A, %b %d at %H:00')}</strong> 
                (exceeding operational safety threshold of {peak_threshold_mw:,.0f} MW).
            </div>
            <div>
                <span class="badge-danger">Level 2 Reserve Alert</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-banner-normal">
            <div>
                <span class="badge-success" style="margin-right: 8px;">GRID NORMAL</span>
                <strong>Grid Operating Margin Adequate:</strong> Maximum projected 48h demand is 
                <span style="font-size: 1.15rem; font-weight: 700; color: #a7f3d0;">{next_48h_peak:,.0f} MW</span>, 
                staying safely below the {peak_threshold_mw:,.0f} MW peak alert threshold.
            </div>
            <div>
                <span class="badge-success">Normal Dispatch</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Executive KPI Cards
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Current Grid Load</div>
            <div class="metric-value">{latest_actual['actual_demand_mw']:,.0f} <span style="font-size: 1rem; color: #94a3b8;">MW</span></div>
            <div class="metric-sub">Recorded at {latest_actual['timestamp_local'].strftime('%H:00 Local')}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">48H Expected Peak</div>
            <div class="metric-value" style="color: {'#f87171' if is_peak_alert else '#38bdf8'};">{next_48h_peak:,.0f} <span style="font-size: 1rem; color: #94a3b8;">MW</span></div>
            <div class="metric-sub">Target: {peak_row['timestamp_local'].strftime('%b %d, %H:00') if peak_row is not None else 'N/A'}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col3:
        current_temp = latest_actual.get("weighted_temperature_f", 82.5)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">ERCOT Weighted Temp</div>
            <div class="metric-value">{current_temp:.1f} <span style="font-size: 1rem; color: #94a3b8;">°F</span></div>
            <div class="metric-sub">Cooling demand regime (>55.2°F)</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col4:
        # Comparison vs EIA at peak hour
        eia_peak_val = peak_row["day_ahead_forecast_mw"] if peak_row is not None else 0
        diff_eia = next_48h_peak - eia_peak_val
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Peak vs EIA Benchmark</div>
            <div class="metric-value" style="color: #a78bfa;">{diff_eia:+,.0f} <span style="font-size: 1rem; color: #94a3b8;">MW</span></div>
            <div class="metric-sub">EIA institutional: {eia_peak_val:,.0f} MW</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # 3. 48-Hour Day-Ahead Interactive Dispatch Forecast Chart
    st.markdown("#### 48-Hour Horizon Day-Ahead Load Forecast")
    
    # Slice last 48 hours of actuals + next 48 hours of forecast
    plot_start = df_actuals["timestamp_local"].max() - timedelta(hours=48)
    df_plot_actuals = df_actuals[df_actuals["timestamp_local"] >= plot_start]

    fig = go.Figure()

    # Confidence Interval Shaded Band (95% CI)
    if len(df_forecast) > 0 and "prediction_lower_95_mw" in df_forecast.columns:
        fig.add_trace(go.Scatter(
            x=list(df_forecast["timestamp_local"]) + list(df_forecast["timestamp_local"])[::-1],
            y=list(df_forecast["prediction_upper_95_mw"]) + list(df_forecast["prediction_lower_95_mw"])[::-1],
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.12)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval (±3,111 MW)',
            hoverinfo="skip",
            showlegend=True
        ))

    # Observed Actual Demand
    if len(df_plot_actuals) > 0:
        fig.add_trace(go.Scatter(
            x=df_plot_actuals["timestamp_local"],
            y=df_plot_actuals["actual_demand_mw"],
            name="Actual Demand (Observed)",
            line=dict(color="#10b981", width=3),
            mode="lines"
        ))

    # LightGBM Champion Hybrid Point Forecast
    if len(df_forecast) > 0:
        # Bridge last actual point to first forecast point for visual continuity
        bridge_x = [df_plot_actuals["timestamp_local"].iloc[-1]] + list(df_forecast["timestamp_local"])
        bridge_y = [df_plot_actuals["actual_demand_mw"].iloc[-1]] + list(df_forecast["predicted_demand_mw"])

        fig.add_trace(go.Scatter(
            x=bridge_x,
            y=bridge_y,
            name="LightGBM Champion (Hybrid Day-Ahead)",
            line=dict(color="#3b82f6", width=3, dash="solid"),
            mode="lines"
        ))

        # EIA Benchmark Day-Ahead Forecast
        fig.add_trace(go.Scatter(
            x=df_forecast["timestamp_local"],
            y=df_forecast["day_ahead_forecast_mw"],
            name="EIA Institutional Benchmark",
            line=dict(color="#ef4444", width=2, dash="dash"),
            mode="lines"
        ))

    # Operational Peak Alert Threshold Line
    fig.add_hline(
        y=peak_threshold_mw,
        line_dash="dot",
        line_color="#f59e0b",
        annotation_text=f"Operational Peak Alert ({peak_threshold_mw:,.0f} MW)",
        annotation_position="top left",
        annotation_font=dict(color="#fbbf24", size=11)
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        margin=dict(l=20, r=20, t=30, b=20),
        height=480,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(15, 23, 42, 0.7)"
        ),
        xaxis=dict(
            title="Local Timestamp (Central Time)",
            gridcolor="rgba(51, 65, 85, 0.4)",
            showgrid=True
        ),
        yaxis=dict(
            title="Electricity Demand (MW)",
            gridcolor="rgba(51, 65, 85, 0.4)",
            showgrid=True,
            tickformat=","
        ),
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    # 4. Critical Hours Detail Table
    exceedance_hours = df_forecast[df_forecast["predicted_demand_mw"] >= peak_threshold_mw]
    if len(exceedance_hours) > 0:
        st.markdown("#### ⚠️ High-Risk Exceedance Windows")
        table_df = exceedance_hours[[
            "timestamp_local", "predicted_demand_mw", "prediction_lower_95_mw", 
            "prediction_upper_95_mw", "day_ahead_forecast_mw"
        ]].copy()
        table_df.columns = ["Timestamp", "Model Forecast (MW)", "Lower 95% (MW)", "Upper 95% (MW)", "EIA Benchmark (MW)"]
        table_df["Reserve Exceedance (MW)"] = table_df["Model Forecast (MW)"] - peak_threshold_mw
        st.dataframe(
            table_df.style.format({
                "Model Forecast (MW)": "{:,.1f}",
                "Lower 95% (MW)": "{:,.1f}",
                "Upper 95% (MW)": "{:,.1f}",
                "EIA Benchmark (MW)": "{:,.1f}",
                "Reserve Exceedance (MW)": "+{:,.1f}"
            }),
            use_container_width=True
        )
