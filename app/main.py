"""Small operational viewer for the persisted H24 and H168 forecast products."""

import plotly.express as px
import pandas as pd
import streamlit as st

from data_loader import load_operational_forecast_data

st.set_page_config(page_title="ERCOT Demand Forecast", layout="wide")
st.title("ERCOT Demand Forecast")
st.caption("H24 is the day-ahead forecast. H168 is a separate model forecast for all 168 hours of the week-ahead horizon.")

if st.button("Refresh forecast data"):
    load_operational_forecast_data.clear()
    st.rerun()

with st.spinner("Connecting to Databricks and loading the latest forecast…"):
    forecast, status, monitoring, monitoring_status = load_operational_forecast_data()
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

horizon = st.radio("Forecast product", [24, 168], horizontal=True,
                   format_func=lambda h: "Day ahead (H24)" if h == 24 else "Week ahead (H168)")
view = forecast[forecast.horizon_hours == horizon].copy()
if view.empty:
    st.error(f"No H{horizon} forecast rows are available.")
    st.stop()
peak = view.loc[view.predicted_demand_mw.idxmax()]
first, second, third = st.columns(3)
first.metric("Expected peak", f"{peak.predicted_demand_mw:,.0f} MW")
peak_time_central = pd.Timestamp(peak.target_timestamp_utc).tz_convert("America/Chicago")
second.metric("Peak time (Central)", peak_time_central.strftime("%b %d, %Y %I:%M %p %Z"))
third.metric("Hours above high-demand threshold", int(view.is_predicted_high_demand.sum()))

figure = px.line(
    view, x="timestamp_local", y="predicted_demand_mw", markers=True,
    labels={"timestamp_local": "Central time", "predicted_demand_mw": "Forecast demand (MW)"},
)
figure.add_hline(y=float(view.high_demand_threshold_mw.iloc[0]), line_dash="dash", line_color="#b91c1c",
                 annotation_text="Configured demand advisory threshold")
figure.update_layout(height=480, font={"size": 13}, margin={"l": 30, "r": 30, "t": 45, "b": 30})
st.plotly_chart(figure, width="stretch")
st.caption("Above-threshold flags identify high forecast demand only; they do not estimate grid-stress probability or available reserve capacity.")
forecast_source = str(view.forecast_source.iloc[0])
if forecast_source == "weekly_seasonal_naive_baseline":
    st.warning("No H168 ML model has passed validation yet. This week-ahead view is the weekly seasonal-naive benchmark, not an ML forecast.")
    st.caption(f"Source: {forecast_source} | Model: {view.model_version.iloc[0]} | Run: {view.model_run_id.iloc[0]} | Forecast origin: {view.forecast_origin_utc.iloc[0]}")
else:
    st.caption(f"Source: {forecast_source} | Model: {view.model_version.iloc[0]} | Run: {view.model_run_id.iloc[0]} | Forecast origin: {view.forecast_origin_utc.iloc[0]}")

st.subheader("Observed forecast reliability")
if monitoring.empty:
    if monitoring_status.startswith("Reliability data could not be loaded") or monitoring_status.startswith("Monitoring data is unavailable"):
        st.warning(monitoring_status)
    else:
        st.info(monitoring_status)
else:
    selected_run_id = view.model_run_id.iloc[0]
    recent = monitoring[
        (monitoring.horizon_hours == horizon)
        & (monitoring.model_run_id == selected_run_id)
    ].copy()
    # Older monitor output can include timestamp matches without actual values.
    # Drop incomplete KPI rows so pandas does not turn all-null sums into 0 MW.
    recent = recent.dropna(subset=["mae_mw", "rmse_mw", "bias_mw", "evaluated_rows"])
    recent = recent[recent["evaluated_rows"] > 0].copy()
    if not recent.empty:
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
        enough_monitoring_history = sample_rows >= 168 and sample_days >= 7
        if not enough_monitoring_history:
            st.warning(
                f"Preliminary reliability only: {sample_rows} matured forecast errors across "
                f"{sample_days} target date(s) for this model. Wait for at least 168 errors "
                "across 7 dates before treating trend charts as useful monitoring signals."
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
        st.caption("Daily error metrics are weighted by the number of matured forecast rows in each lead-hour bucket.")

        lead_rows = recent.copy()
        if horizon == 24:
            lead_rows["lead_bucket"] = "1–24 hours"
        else:
            lead_rows["lead_bucket"] = pd.cut(
                lead_rows["lead_hours"], bins=[0, 24, 48, 168],
                labels=["1–24 hours", "25–48 hours", "49–168 hours"],
            )
        lead_rows = lead_rows.dropna(
            subset=["lead_bucket", "mae_mw", "rmse_mw", "bias_mw", "evaluated_rows"]
        )
        lead_rows["absolute_error_sum"] = lead_rows["mae_mw"] * lead_rows["evaluated_rows"]
        lead_rows["squared_error_sum"] = lead_rows["rmse_mw"].pow(2) * lead_rows["evaluated_rows"]
        lead_rows["signed_error_sum"] = lead_rows["bias_mw"] * lead_rows["evaluated_rows"]
        by_lead = lead_rows.groupby("lead_bucket", observed=False, as_index=False).agg(
            evaluated_rows=("evaluated_rows", "sum"),
            absolute_error_sum=("absolute_error_sum", "sum"),
            squared_error_sum=("squared_error_sum", "sum"),
            signed_error_sum=("signed_error_sum", "sum"),
        )
        by_lead = by_lead[by_lead["evaluated_rows"] > 0].copy()
        by_lead["mae_mw"] = by_lead["absolute_error_sum"] / by_lead["evaluated_rows"]
        by_lead["rmse_mw"] = (by_lead["squared_error_sum"] / by_lead["evaluated_rows"]).pow(0.5)
        by_lead["bias_mw"] = by_lead["signed_error_sum"] / by_lead["evaluated_rows"]
        st.subheader("Observed error by forecast lead")
        if horizon == 168:
            st.caption("Week-ahead errors are separated into day one, day two, and days three through seven.")
        if enough_monitoring_history:
            lead_figure = px.bar(
                by_lead, x="lead_bucket", y=["mae_mw", "rmse_mw"], barmode="group",
                labels={"lead_bucket": "Forecast lead", "value": "Error (MW)", "variable": "Metric"},
            )
            lead_figure.update_layout(
                height=360, font={"size": 13},
                legend={"orientation": "h", "y": 1.08, "x": 1, "xanchor": "right"},
            )
            st.plotly_chart(lead_figure, width="stretch")
        else:
            st.caption("Lead-bucket trend chart will appear after at least 168 matured errors across 7 target dates.")
        st.dataframe(by_lead[["lead_bucket", "evaluated_rows", "mae_mw", "rmse_mw", "bias_mw"]],
                     hide_index=True, width="stretch")
    else:
        st.info(
            f"No matured forecast errors are available for the selected H{horizon} model run yet. "
            f"This run has {len(view):,} issued forecast hours. Reliability metrics will appear "
            "after matching actual demand is available and the monitoring task processes it."
        )
