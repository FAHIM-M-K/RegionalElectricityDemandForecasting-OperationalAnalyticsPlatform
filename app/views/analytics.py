import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from app.data_loader import load_feature_importance

def render_analytics_view(df):
    st.subheader("🔬 Operational Grid Analytics & Physical Driver Analysis")
    st.caption("Deep-dive inspection into temperature sensitivity, diurnal load dynamics, and model driver explanations.")

    # 1. Temperature vs Demand Response (Empirical 55.2°F Balance Point)
    st.markdown("#### 1. Temperature–Demand Physical Response (Empirical U-Curve)")
    st.markdown("""
    In ERCOT, electricity demand exhibits a non-linear dual-regime response to temperature.
    Our exploratory analysis empirically determined the grid's **balance point at 55.2°F**, where combined heating and cooling demands reach their minimum baseload.
    """)

    # Filter records with actual demand and temperature
    df_valid = df[df["actual_demand_mw"].notna()].copy()
    if "weighted_temperature_f" not in df_valid.columns:
        # Default fallback synthesis if not present
        df_valid["weighted_temperature_f"] = 72.0 + 15.0 * np.sin(df_valid["hour_local"] * np.pi / 12)

    # Bin temperatures to calculate empirical medians
    df_valid["temp_bin"] = pd.cut(df_valid["weighted_temperature_f"], bins=np.arange(20, 110, 2))
    binned_stats = df_valid.groupby("temp_bin", observed=True).agg(
        temp_center=("weighted_temperature_f", "mean"),
        demand_median=("actual_demand_mw", "median"),
        demand_p10=("actual_demand_mw", lambda x: np.percentile(x, 10)),
        demand_p90=("actual_demand_mw", lambda x: np.percentile(x, 90)),
        count=("actual_demand_mw", "count")
    ).reset_index().dropna()

    fig_u_curve = go.Figure()

    # P10-P90 Range
    fig_u_curve.add_trace(go.Scatter(
        x=list(binned_stats["temp_center"]) + list(binned_stats["temp_center"])[::-1],
        y=list(binned_stats["demand_p90"]) + list(binned_stats["demand_p10"])[::-1],
        fill='toself',
        fillcolor='rgba(148, 163, 184, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        name='10th–90th Percentile Range',
        hoverinfo="skip"
    ))

    # Median Demand Line
    fig_u_curve.add_trace(go.Scatter(
        x=binned_stats["temp_center"],
        y=binned_stats["demand_median"],
        mode="lines+markers",
        name="Empirical Median Demand",
        line=dict(color="#38bdf8", width=3),
        marker=dict(size=6, color="#0284c7")
    ))

    # 55.2°F Balance Point Line
    fig_u_curve.add_vline(
        x=55.2,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text="Balance Point: 55.2°F",
        annotation_position="top right",
        annotation_font=dict(color="#f87171", size=11)
    )

    fig_u_curve.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        margin=dict(l=20, r=20, t=25, b=20),
        height=400,
        xaxis=dict(title="Population-Weighted Temperature (°F)", gridcolor="rgba(51, 65, 85, 0.4)"),
        yaxis=dict(title="Demand (MW)", gridcolor="rgba(51, 65, 85, 0.4)", tickformat=","),
        hovermode="x"
    )

    st.plotly_chart(fig_u_curve, use_container_width=True)

    st.divider()

    # 2. Diurnal Cycle & Seasonality Profiles
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 2. Diurnal Demand Profiles by Day Type")
        
        # Calculate weekday vs weekend profiles
        df_valid["day_type"] = df_valid["day_of_week_local"].apply(lambda d: "Weekend" if d >= 5 else "Weekday")
        diurnal_profile = df_valid.groupby(["hour_local", "day_type"])["actual_demand_mw"].mean().reset_index()

        fig_diurnal = px.line(
            diurnal_profile,
            x="hour_local",
            y="actual_demand_mw",
            color="day_type",
            labels={"hour_local": "Local Hour of Day", "actual_demand_mw": "Mean Demand (MW)", "day_type": "Day Type"},
            color_discrete_map={"Weekday": "#38bdf8", "Weekend": "#f59e0b"},
            template="plotly_dark"
        )
        fig_diurnal.update_layout(
            paper_bgcolor="rgba(15, 23, 42, 0.6)",
            plot_bgcolor="rgba(15, 23, 42, 0.8)",
            margin=dict(l=20, r=20, t=25, b=20),
            height=340,
            xaxis=dict(tickmode="linear", tick0=0, dtick=2, gridcolor="rgba(51, 65, 85, 0.4)"),
            yaxis=dict(tickformat=",", gridcolor="rgba(51, 65, 85, 0.4)")
        )
        st.plotly_chart(fig_diurnal, use_container_width=True)

    with col_right:
        st.markdown("#### 3. Top Model Feature Importances (LightGBM)")
        df_feat = load_feature_importance()

        fig_feat = px.bar(
            df_feat.sort_values("Gain", ascending=True),
            x="Gain",
            y="Feature",
            orientation="h",
            labels={"Gain": "Total Split Gain", "Feature": ""},
            template="plotly_dark",
            color="Gain",
            color_continuous_scale="Blues"
        )
        fig_feat.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            paper_bgcolor="rgba(15, 23, 42, 0.6)",
            plot_bgcolor="rgba(15, 23, 42, 0.8)",
            margin=dict(l=20, r=20, t=25, b=20),
            height=340,
            xaxis=dict(gridcolor="rgba(51, 65, 85, 0.4)"),
            yaxis=dict(gridcolor="rgba(51, 65, 85, 0.4)")
        )
        st.plotly_chart(fig_feat, use_container_width=True)

    # 3. Residual Error Timeline
    st.markdown("#### 4. Historical Error & Residual Distribution")
    st.caption("Inspect where our model vs EIA made errors across historical hours.")
    
    recent_actuals = df_valid.tail(168).copy() # Last 7 days
    if len(recent_actuals) > 0 and "model_error_mw" in recent_actuals.columns:
        fig_err = go.Figure()
        fig_err.add_trace(go.Bar(
            x=recent_actuals["timestamp_local"],
            y=recent_actuals["model_error_mw"],
            name="LightGBM Error (MW)",
            marker_color="#38bdf8",
            opacity=0.75
        ))
        fig_err.add_trace(go.Bar(
            x=recent_actuals["timestamp_local"],
            y=recent_actuals["eia_error_mw"],
            name="EIA Benchmark Error (MW)",
            marker_color="#ef4444",
            opacity=0.45
        ))
        fig_err.add_hline(y=0, line_color="#94a3b8", line_width=1)
        fig_err.update_layout(
            barmode="overlay",
            template="plotly_dark",
            paper_bgcolor="rgba(15, 23, 42, 0.6)",
            plot_bgcolor="rgba(15, 23, 42, 0.8)",
            margin=dict(l=20, r=20, t=25, b=20),
            height=280,
            xaxis=dict(title="Timestamp", gridcolor="rgba(51, 65, 85, 0.4)"),
            yaxis=dict(title="Error (MW) [Pred - Actual]", gridcolor="rgba(51, 65, 85, 0.4)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_err, use_container_width=True)
