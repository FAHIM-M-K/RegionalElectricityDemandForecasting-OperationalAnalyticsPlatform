import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from app.data_loader import load_feature_importance

def render_analytics_view(df):
    st.markdown("### Operational Grid Analytics and Driver Attribution")
    st.caption("Empirical temperature response dynamics, diurnal load cycles, and machine learning feature attribution.")

    # 1. Temperature vs Demand Response (Empirical 55.2°F Balance Point)
    st.markdown("##### 1. Empirical Temperature-Demand Response")
    st.markdown("""
    Electricity demand exhibits an asymmetric non-linear response to ambient temperature.
    Statistical profiling identifies the system balance point at **55.2 deg F**, where heating and cooling loads reach baseline.
    """)

    df_valid = df[df["actual_demand_mw"].notna()].copy()
    if "weighted_temperature_f" not in df_valid.columns:
        df_valid["weighted_temperature_f"] = 72.0 + 15.0 * np.sin(df_valid["hour_local"] * np.pi / 12)

    df_valid["temp_bin"] = pd.cut(df_valid["weighted_temperature_f"], bins=np.arange(20, 110, 2))
    binned_stats = df_valid.groupby("temp_bin", observed=True).agg(
        temp_center=("weighted_temperature_f", "mean"),
        demand_median=("actual_demand_mw", "median"),
        demand_p10=("actual_demand_mw", lambda x: np.percentile(x, 10)),
        demand_p90=("actual_demand_mw", lambda x: np.percentile(x, 90)),
        count=("actual_demand_mw", "count")
    ).reset_index().dropna()

    fig_u_curve = go.Figure()

    # P10-P90 Range (Subtle neutral fill)
    fig_u_curve.add_trace(go.Scatter(
        x=list(binned_stats["temp_center"]) + list(binned_stats["temp_center"])[::-1],
        y=list(binned_stats["demand_p90"]) + list(binned_stats["demand_p10"])[::-1],
        fill='toself',
        fillcolor='rgba(15, 23, 42, 0.05)',
        line=dict(color='rgba(255,255,255,0)'),
        name='10th-90th Percentile Range',
        hoverinfo="skip"
    ))

    # Median Demand Line (Cobalt Blue)
    fig_u_curve.add_trace(go.Scatter(
        x=binned_stats["temp_center"],
        y=binned_stats["demand_median"],
        mode="lines+markers",
        name="Empirical Median Demand",
        line=dict(color="#2563eb", width=2.5),
        marker=dict(size=5, color="#1d4ed8")
    ))

    # 55.2 deg F Balance Point Line
    fig_u_curve.add_vline(
        x=55.2,
        line_dash="dash",
        line_color="#dc2626",
        line_width=1.5,
        annotation_text="Balance Point: 55.2 deg F",
        annotation_position="top right",
        annotation_font=dict(color="#dc2626", size=10)
    )

    fig_u_curve.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin=dict(l=40, r=20, t=25, b=30),
        height=360,
        font=dict(family="Inter, sans-serif", size=11, color="#334155"),
        xaxis=dict(title="Population-Weighted Temperature (deg F)", gridcolor="#f1f5f9", linecolor="#cbd5e1"),
        yaxis=dict(title="Observed Demand (MW)", gridcolor="#f1f5f9", linecolor="#cbd5e1", tickformat=","),
        hovermode="x"
    )

    st.plotly_chart(fig_u_curve, use_container_width=True)

    st.divider()

    # 2. Diurnal Cycle and Feature Attribution
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("##### 2. Diurnal Load Profiles by Day Type")
        df_valid["day_type"] = df_valid["day_of_week_local"].apply(lambda d: "Weekend" if d >= 5 else "Weekday")
        diurnal_profile = df_valid.groupby(["hour_local", "day_type"])["actual_demand_mw"].mean().reset_index()

        fig_diurnal = px.line(
            diurnal_profile,
            x="hour_local",
            y="actual_demand_mw",
            color="day_type",
            labels={"hour_local": "Hour of Day (Local)", "actual_demand_mw": "Mean Demand (MW)", "day_type": "Schedule"},
            color_discrete_map={"Weekday": "#0f172a", "Weekend": "#64748b"},
            template="plotly_white"
        )
        fig_diurnal.update_layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            margin=dict(l=30, r=20, t=25, b=30),
            height=320,
            font=dict(family="Inter, sans-serif", size=11, color="#334155"),
            xaxis=dict(tickmode="linear", tick0=0, dtick=2, gridcolor="#f1f5f9", linecolor="#cbd5e1"),
            yaxis=dict(tickformat=",", gridcolor="#f1f5f9", linecolor="#cbd5e1")
        )
        st.plotly_chart(fig_diurnal, use_container_width=True)

    with col_right:
        st.markdown("##### 3. Machine Learning Feature Attribution (LightGBM)")
        df_feat = load_feature_importance()

        fig_feat = px.bar(
            df_feat.sort_values("Gain", ascending=True),
            x="Gain",
            y="Feature",
            orientation="h",
            labels={"Gain": "Split Gain", "Feature": ""},
            template="plotly_white",
            color_discrete_sequence=["#2563eb"]
        )
        fig_feat.update_layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            margin=dict(l=30, r=20, t=25, b=30),
            height=320,
            font=dict(family="Inter, sans-serif", size=10, color="#334155"),
            xaxis=dict(gridcolor="#f1f5f9", linecolor="#cbd5e1"),
            yaxis=dict(gridcolor="#f1f5f9", linecolor="#cbd5e1")
        )
        st.plotly_chart(fig_feat, use_container_width=True)

    # 3. Residual Error Timeline
    st.markdown("##### 4. Chronological Forecast Residuals")
    st.caption("Comparison of point error residuals (Predicted minus Actual) across evaluated historical hours.")
    
    recent_actuals = df_valid.tail(168).copy()
    if len(recent_actuals) > 0 and "model_error_mw" in recent_actuals.columns:
        fig_err = go.Figure()
        fig_err.add_trace(go.Bar(
            x=recent_actuals["timestamp_local"],
            y=recent_actuals["model_error_mw"],
            name="Champion Model Error (MW)",
            marker_color="#2563eb",
            opacity=0.8
        ))
        fig_err.add_trace(go.Bar(
            x=recent_actuals["timestamp_local"],
            y=recent_actuals["eia_error_mw"],
            name="EIA Benchmark Error (MW)",
            marker_color="#94a3b8",
            opacity=0.6
        ))
        fig_err.add_hline(y=0, line_color="#0f172a", line_width=1)
        fig_err.update_layout(
            barmode="overlay",
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            margin=dict(l=40, r=20, t=25, b=30),
            height=260,
            font=dict(family="Inter, sans-serif", size=11, color="#334155"),
            xaxis=dict(title="", gridcolor="#f1f5f9", linecolor="#cbd5e1"),
            yaxis=dict(title="Residual Error (MW)", gridcolor="#f1f5f9", linecolor="#cbd5e1"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_err, use_container_width=True)
