"""Small operational viewer for the persisted H24 and H168 forecast products."""

import plotly.express as px
import pandas as pd
import streamlit as st

from app.data_loader import load_operational_forecast_data

st.set_page_config(page_title="ERCOT Demand Forecast", layout="wide")
st.title("ERCOT Demand Forecast")
st.caption("H24 uses the promoted demand model. H168 uses its promoted model when available, otherwise the labeled weekly seasonal-naive benchmark.")

forecast, status, monitoring = load_operational_forecast_data()
if status.startswith("Stale"):
    st.warning(status)
else:
    st.caption(status)
if forecast.empty:
    st.info("No persisted forecast is available. Run the scoring workflow after an H24 production model has been promoted.")
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
forecast_source = str(view.forecast_source.iloc[0])
if forecast_source == "weekly_seasonal_naive_baseline":
    st.warning("No H168 ML model has passed validation yet. This week-ahead view is the weekly seasonal-naive benchmark, not an ML forecast.")
st.caption(f"Source: {forecast_source} | Model: {view.model_version.iloc[0]} | Run: {view.model_run_id.iloc[0]} | Forecast origin: {view.forecast_origin_utc.iloc[0]}")

if not monitoring.empty:
    st.subheader("Observed forecast reliability")
    selected_run_id = view.model_run_id.iloc[0]
    recent = monitoring[
        (monitoring.horizon_hours == horizon)
        & (monitoring.model_run_id == selected_run_id)
    ].copy()
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
        if sample_rows < 168 or sample_days < 7:
            st.warning(
                f"Preliminary reliability only: {sample_rows} matured forecast errors across "
                f"{sample_days} target date(s) for this model. Wait for at least 168 errors "
                "across 7 dates before treating the trend as a useful weekly monitoring signal."
            )
        else:
            st.caption(
                f"Monitoring coverage: {sample_rows:,} matured forecast errors across "
                f"{sample_days} target dates for the selected production model."
            )
        reliability_figure = px.line(reliability, x="target_date_utc", y=["mae_mw", "rmse_mw"], markers=True,
                                     labels={"value": "Error (MW)", "target_date_utc": "Target date UTC", "variable": "Metric"})
        st.plotly_chart(reliability_figure, use_container_width=True)
        st.caption("Daily error metrics are weighted by the number of matured forecast rows in each lead-hour bucket.")

        lead_rows = recent.copy()
        if horizon == 24:
            lead_rows["lead_bucket"] = "1–24 hours"
        else:
            lead_rows["lead_bucket"] = pd.cut(
                lead_rows["lead_hours"], bins=[0, 24, 48, 168],
                labels=["1–24 hours", "25–48 hours", "49–168 hours"],
            )
        lead_rows = lead_rows.dropna(subset=["lead_bucket"])
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
        lead_figure = px.bar(
            by_lead, x="lead_bucket", y=["mae_mw", "rmse_mw"], barmode="group",
            labels={"lead_bucket": "Forecast lead", "value": "Error (MW)", "variable": "Metric"},
        )
        st.plotly_chart(lead_figure, use_container_width=True)
        st.dataframe(by_lead[["lead_bucket", "evaluated_rows", "mae_mw", "rmse_mw", "bias_mw"]],
                     hide_index=True, use_container_width=True)
    else:
        st.info("No matured forecast errors are available for this horizon yet.")
