import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

def render_executive_view(df, peak_threshold_mw):
    st.markdown("### Operational Grid Dispatch and Forward Forecast")
    st.caption("Central Balancing Authority load tracking, 48-hour forward capacity forecast, and reserve margin assessment.")

    # Current time cutoff
    now = datetime.now()
    df_actuals = df[df["actual_demand_mw"].notna()].sort_values("timestamp_local")
    df_forecast = df[df["actual_demand_mw"].isna()].sort_values("timestamp_local")

    if len(df_forecast) == 0:
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
            <strong>CRITICAL PEAK ALERT:</strong> Forecasted load reaches 
            <strong>{next_48h_peak:,.0f} MW</strong> on 
            <strong>{peak_row['timestamp_local'].strftime('%A, %b %d at %H:00')}</strong>, 
            exceeding the operational threshold of {peak_threshold_mw:,.0f} MW.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-banner-normal">
            <strong>NORMAL DISPATCH:</strong> Peak projected demand is 
            <strong>{next_48h_peak:,.0f} MW</strong>, staying within operational reserve limits.
        </div>
        """, unsafe_allow_html=True)

    # 2. Executive KPI Cards
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Current Grid Load</div>
            <div class="metric-value">{latest_actual['actual_demand_mw']:,.0f} <span style="font-size: 0.9rem; font-weight: 500; color: #64748b;">MW</span></div>
            <div class="metric-sub">Observed at {latest_actual['timestamp_local'].strftime('%H:00 Local')}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">48H Expected Peak</div>
            <div class="metric-value" style="color: {'#dc2626' if is_peak_alert else '#2563eb'};">{next_48h_peak:,.0f} <span style="font-size: 0.9rem; font-weight: 500; color: #64748b;">MW</span></div>
            <div class="metric-sub">Expected at {peak_row['timestamp_local'].strftime('%b %d, %H:00') if peak_row is not None else 'N/A'}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col3:
        current_temp = latest_actual.get("weighted_temperature_f", 82.5)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Regional Mean Temperature</div>
            <div class="metric-value">{current_temp:.1f} <span style="font-size: 0.9rem; font-weight: 500; color: #64748b;">deg F</span></div>
            <div class="metric-sub">Cooling regime (Balance: 55.2 deg F)</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col4:
        eia_peak_val = peak_row["day_ahead_forecast_mw"] if peak_row is not None else 0
        diff_eia = next_48h_peak - eia_peak_val
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Peak Variance vs EIA</div>
            <div class="metric-value" style="color: {'#0f172a' if diff_eia == 0 else ('#dc2626' if diff_eia > 0 else '#2563eb')};">{diff_eia:+,.0f} <span style="font-size: 0.9rem; font-weight: 500; color: #64748b;">MW</span></div>
            <div class="metric-sub">EIA baseline: {eia_peak_val:,.0f} MW</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # 3. 48-Hour Day-Ahead Interactive Dispatch Forecast Chart
    st.markdown("##### 48-Hour Day-Ahead Horizon Forecast")
    
    plot_start = df_actuals["timestamp_local"].max() - timedelta(hours=48)
    df_plot_actuals = df_actuals[df_actuals["timestamp_local"] >= plot_start]

    fig = go.Figure()

    # 95% Confidence Interval Band (clean subtle tint)
    if len(df_forecast) > 0 and "prediction_lower_95_mw" in df_forecast.columns:
        fig.add_trace(go.Scatter(
            x=list(df_forecast["timestamp_local"]) + list(df_forecast["timestamp_local"])[::-1],
            y=list(df_forecast["prediction_upper_95_mw"]) + list(df_forecast["prediction_lower_95_mw"])[::-1],
            fill='toself',
            fillcolor='rgba(37, 99, 235, 0.08)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval',
            hoverinfo="skip",
            showlegend=True
        ))

    # Observed Actual Demand (Primary solid black/slate)
    if len(df_plot_actuals) > 0:
        fig.add_trace(go.Scatter(
            x=df_plot_actuals["timestamp_local"],
            y=df_plot_actuals["actual_demand_mw"],
            name="Observed Actual Demand",
            line=dict(color="#0f172a", width=2.5),
            mode="lines"
        ))

    # LightGBM Champion Hybrid Point Forecast (Cobalt Blue)
    if len(df_forecast) > 0:
        bridge_x = [df_plot_actuals["timestamp_local"].iloc[-1]] + list(df_forecast["timestamp_local"])
        bridge_y = [df_plot_actuals["actual_demand_mw"].iloc[-1]] + list(df_forecast["predicted_demand_mw"])

        fig.add_trace(go.Scatter(
            x=bridge_x,
            y=bridge_y,
            name="Champion Model Forecast",
            line=dict(color="#2563eb", width=2.5),
            mode="lines"
        ))

        # EIA Benchmark Day-Ahead Forecast (Muted dashed slate)
        fig.add_trace(go.Scatter(
            x=df_forecast["timestamp_local"],
            y=df_forecast["day_ahead_forecast_mw"],
            name="EIA Benchmark Forecast",
            line=dict(color="#94a3b8", width=1.75, dash="dash"),
            mode="lines"
        ))

    # Operational Peak Alert Threshold Line
    fig.add_hline(
        y=peak_threshold_mw,
        line_dash="dot",
        line_color="#dc2626",
        line_width=1.5,
        annotation_text=f"Operational Threshold ({peak_threshold_mw:,.0f} MW)",
        annotation_position="top left",
        annotation_font=dict(color="#dc2626", size=10)
    )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin=dict(l=40, r=20, t=30, b=30),
        height=420,
        font=dict(family="Inter, sans-serif", size=11, color="#334155"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.9)"
        ),
        xaxis=dict(
            title="",
            gridcolor="#f1f5f9",
            linecolor="#cbd5e1",
            showgrid=True
        ),
        yaxis=dict(
            title="Electricity Demand (MW)",
            gridcolor="#f1f5f9",
            linecolor="#cbd5e1",
            showgrid=True,
            tickformat=","
        ),
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    # 4. Critical Hours Detail Table
    exceedance_hours = df_forecast[df_forecast["predicted_demand_mw"] >= peak_threshold_mw]
    if len(exceedance_hours) > 0:
        st.markdown("##### Threshold Exceedance Schedule")
        table_df = exceedance_hours[[
            "timestamp_local", "predicted_demand_mw", "prediction_lower_95_mw", 
            "prediction_upper_95_mw", "day_ahead_forecast_mw"
        ]].copy()
        table_df.columns = ["Timestamp", "Model Forecast (MW)", "Lower 95% (MW)", "Upper 95% (MW)", "EIA Benchmark (MW)"]
        table_df["Exceedance (MW)"] = table_df["Model Forecast (MW)"] - peak_threshold_mw
        st.dataframe(
            table_df.style.format({
                "Model Forecast (MW)": "{:,.1f}",
                "Lower 95% (MW)": "{:,.1f}",
                "Upper 95% (MW)": "{:,.1f}",
                "EIA Benchmark (MW)": "{:,.1f}",
                "Exceedance (MW)": "+{:,.1f}"
            }),
            use_container_width=True
        )
