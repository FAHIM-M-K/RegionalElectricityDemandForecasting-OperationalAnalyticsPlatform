"""Small operational viewer for the persisted H24 and H168 forecast products."""

import plotly.express as px
import pandas as pd
import streamlit as st

from app.data_loader import load_operational_forecast_data

st.set_page_config(page_title="ERCOT Demand Forecast", layout="wide")
st.title("ERCOT Demand Forecast")
st.caption("Direct demand-only LightGBM forecasts. H24 supports day-ahead operations; H168 supports week-ahead planning.")

forecast, status, monitoring = load_operational_forecast_data()
if status.startswith("Stale"):
    st.warning(status)
else:
    st.caption(status)
if forecast.empty:
    st.info("No persisted forecast is available. Run the scoring workflow after a production H24 and H168 model has been promoted.")
    st.stop()

horizon = st.radio("Forecast product", [24, 168], horizontal=True,
                   format_func=lambda h: "Day ahead (H24)" if h == 24 else "Week ahead (H168)")
view = forecast[forecast.horizon_hours == horizon].copy()
peak = view.loc[view.predicted_demand_mw.idxmax()]
first, second, third = st.columns(3)
first.metric("Expected peak", f"{peak.predicted_demand_mw:,.0f} MW")
second.metric("Peak time (Central)", str(peak.timestamp_local))
third.metric("Hours above high-demand threshold", int(view.is_predicted_high_demand.sum()))

figure = px.line(view, x="timestamp_local", y="predicted_demand_mw", markers=True,
                 labels={"timestamp_local": "Central time", "predicted_demand_mw": "Forecast demand (MW)"})
figure.add_hline(y=float(view.high_demand_threshold_mw.iloc[0]), line_dash="dash", line_color="#b91c1c",
                 annotation_text="High-demand threshold")
st.plotly_chart(figure, use_container_width=True)
st.caption(f"Model: {view.model_version.iloc[0]} | Run: {view.model_run_id.iloc[0]} | Forecast origin: {view.forecast_origin_utc.iloc[0]}")

if not monitoring.empty:
    st.subheader("Observed forecast reliability")
    recent = monitoring[monitoring.horizon_hours == horizon].copy()
    if not recent.empty:
        recent["target_date_utc"] = pd.to_datetime(recent["target_date_utc"])
        reliability = recent.groupby("target_date_utc", as_index=False).agg(mae_mw=("mae_mw", "mean"), rmse_mw=("rmse_mw", "mean"))
        reliability_figure = px.line(reliability, x="target_date_utc", y=["mae_mw", "rmse_mw"], markers=True,
                                     labels={"value": "Error (MW)", "target_date_utc": "Target date UTC", "variable": "Metric"})
        st.plotly_chart(reliability_figure, use_container_width=True)
        st.caption("Errors are grouped by forecast horizon and lead hour. Early lead hours may have more evaluated observations than later leads.")
    else:
        st.info("No matured forecast errors are available for this horizon yet.")
