"""Small operational viewer for the persisted rolling H24 forecast."""

import plotly.express as px
import pandas as pd
import streamlit as st

from data_loader import load_operational_forecast_data

MIN_MATURED_ERRORS = 7 * 24

st.set_page_config(page_title="ERCOT Demand Forecast", layout="wide")
st.title("ERCOT Demand Forecast")
st.caption("Hourly refreshed system demand forecast for the next 24 hours, using weather available at issue time.")

if st.button("Refresh forecast data"):
    load_operational_forecast_data.clear()
    st.rerun()

with st.spinner("Connecting to Databricks and loading the latest forecast…"):
    forecast, status, monitoring, monitoring_status = load_operational_forecast_data(
        cache_contract_version="h24-only-v4"
    )
if status.startswith("Stale"):
    st.warning(status)
else:
    st.caption(status)
if forecast.empty:
    if status.startswith("Forecast unavailable") or status == "Databricks connection is not configured":
        st.error(status)
    else:
        st.info(status or "No persisted forecast is available yet.")
    st.stop()

view = forecast[forecast.horizon_hours == 24].copy()
if view.empty:
    st.error("No rolling H24 forecast rows are available.")
    st.stop()
peak = view.loc[view.predicted_demand_mw.idxmax()]
first, second, third = st.columns(3)
first.metric("Expected peak", f"{peak.predicted_demand_mw:,.0f} MW")
peak_time_central = pd.Timestamp(peak.target_timestamp_utc).tz_convert("America/Chicago")
second.metric("Peak time (Central)", peak_time_central.strftime("%b %d, %Y %I:%M %p %Z"))
latest_actual_mw = view.latest_actual_demand_mw.iloc[0]
latest_actual_time = view.latest_actual_timestamp_utc.iloc[0]
if pd.notna(latest_actual_mw) and pd.notna(latest_actual_time):
    latest_actual_time_central = pd.Timestamp(latest_actual_time).tz_convert("America/Chicago")
    third.metric(
        "Latest actual demand",
        f"{latest_actual_mw:,.0f} MW",
        help=f"Most recent observed demand, recorded {latest_actual_time_central.strftime('%b %d, %Y %I:%M %p %Z')}.",
    )
else:
    third.metric("Latest actual demand", "Unavailable", help="No non-null actual demand is currently available.")

figure = px.line(
    view, x="timestamp_local", y="predicted_demand_mw", markers=True,
    labels={"timestamp_local": "Central time", "predicted_demand_mw": "Forecast demand (MW)"},
)
figure.update_layout(height=480, font={"size": 13}, margin={"l": 30, "r": 30, "t": 45, "b": 30})
st.plotly_chart(figure, width="stretch")
forecast_source = str(view.forecast_source.iloc[0])
st.caption(f"Source: {forecast_source} | Model: {view.model_version.iloc[0]} | Run: {view.model_run_id.iloc[0]} | Forecast origin: {view.forecast_origin_utc.iloc[0]}")

st.subheader("Observed forecast reliability")
if monitoring.empty:
    if monitoring_status.startswith("Reliability data could not be loaded") or monitoring_status.startswith("Monitoring data is unavailable"):
        st.warning(monitoring_status)
    else:
        st.info(monitoring_status)
else:
    selected_run_id = view.model_run_id.iloc[0]
    recent = monitoring[monitoring.model_run_id == selected_run_id].copy()
    # Older monitor output can include timestamp matches without actual values.
    # Drop incomplete KPI rows so pandas does not turn all-null sums into 0 MW.
    recent = recent.dropna(subset=["mae_mw", "rmse_mw", "bias_mw", "evaluated_rows"])
    recent = recent[recent["evaluated_rows"] > 0].copy()
    first_target_utc = view["target_timestamp_utc"].min()
    latest_actual_utc = view["latest_actual_timestamp_utc"].max()
    current_issue_has_no_matured_targets = (
        pd.notna(latest_actual_utc) and latest_actual_utc < first_target_utc
    )
    if current_issue_has_no_matured_targets:
        actual_label = latest_actual_utc.strftime("%Y-%m-%d %H:%M UTC")
        target_label = first_target_utc.strftime("%Y-%m-%d %H:%M UTC")
        st.info(
            "Current H24 forecast issue has no matured errors yet: actual demand is "
            f"available through {actual_label}, and this issue starts at {target_label}."
        )
    if not recent.empty:
        st.caption(
            "The reliability figures below summarize previously matured forecasts for this "
            "production model; they are not error metrics for the current forecast issue."
        )
        recent["target_date_utc"] = pd.to_datetime(recent["target_date_utc"])
        recent["absolute_error_sum"] = recent["mae_mw"] * recent["evaluated_rows"]
        recent["squared_error_sum"] = recent["rmse_mw"].pow(2) * recent["evaluated_rows"]
        recent["signed_error_sum"] = recent["bias_mw"] * recent["evaluated_rows"]
        reliability = recent.groupby("target_date_utc", as_index=False).agg(
            evaluated_rows=("evaluated_rows", "sum"),
            absolute_error_sum=("absolute_error_sum", "sum"),
            squared_error_sum=("squared_error_sum", "sum"),
            signed_error_sum=("signed_error_sum", "sum"),
        )
        reliability["mae_mw"] = reliability["absolute_error_sum"] / reliability["evaluated_rows"]
        reliability["rmse_mw"] = (reliability["squared_error_sum"] / reliability["evaluated_rows"]).pow(0.5)
        reliability["bias_mw"] = reliability["signed_error_sum"] / reliability["evaluated_rows"]
        sample_rows = int(reliability["evaluated_rows"].sum())
        sample_days = int(reliability["target_date_utc"].nunique())
        enough_monitoring_history = sample_rows >= MIN_MATURED_ERRORS and sample_days >= 7
        if not enough_monitoring_history:
            st.warning(
                f"Preliminary reliability only: {sample_rows} matured forecast errors across "
                f"{sample_days} target date(s) for this model. Wait for at least "
                f"{MIN_MATURED_ERRORS} hourly errors across 7 dates before treating the trend as useful."
            )
        else:
            st.caption(
                f"Monitoring coverage: {sample_rows:,} matured forecast errors across "
                f"{sample_days} target dates for the selected production model."
            )
        if enough_monitoring_history:
            reliability_figure = px.line(
                reliability, x="target_date_utc", y=["mae_mw", "rmse_mw"], markers=True,
                labels={"value": "Error (MW)", "target_date_utc": "Target date UTC", "variable": "Metric"},
            )
            reliability_figure.update_layout(
                height=360, font={"size": 13},
                legend={"orientation": "h", "y": 1.08, "x": 1, "xanchor": "right"},
            )
            st.plotly_chart(reliability_figure, width="stretch")
        st.caption("Daily error metrics summarize the matured hourly H24 forecasts for the selected production model.")
    else:
        if current_issue_has_no_matured_targets:
            st.caption("Historical reliability will appear after this model has matured forecast errors.")
        else:
            st.info(
                "No matured forecast errors are available for this H24 model yet. "
                f"This issue contains {len(view):,} forecast hours. Check that the monitoring task "
                "has processed actual demand for the forecast targets."
            )
