"""Read the current real forecast issue; never fabricate an offline forecast."""

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def _query_frame(connection, query):
    """Run a Databricks SQL query without pandas' unsupported-DBAPI warning."""
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        columns = [description[0] for description in cursor.description]
        return pd.DataFrame.from_records(cursor.fetchall(), columns=columns)
    finally:
        cursor.close()


def get_databricks_connection():
    host, path, token = (os.environ.get(key, "").strip() for key in ("DATABRICKS_HOST", "DATABRICKS_HTTP_PATH", "DATABRICKS_TOKEN"))
    if not all((host, path, token)):
        return None
    try:
        from databricks import sql
        return sql.connect(server_hostname=host.removeprefix("https://").removeprefix("http://").rstrip("/"),
                           http_path=path, access_token=token)
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_operational_forecast_data():
    """Return the latest persisted forecast and an honest connection status."""
    connection = get_databricks_connection()
    if connection is None:
        return pd.DataFrame(), "Databricks connection is not configured", pd.DataFrame()
    try:
        forecast = _query_frame(connection, """
            SELECT respondent, forecast_origin_utc, issued_at_utc, target_timestamp_utc, timestamp_local,
                   horizon_hours, lead_hours, hour_local, predicted_demand_mw,
                   is_predicted_high_demand, high_demand_threshold_mw, forecast_source, model_version, model_run_id
            FROM main.gold.forecast_latest
            WHERE respondent = 'ERCO'
            ORDER BY horizon_hours, target_timestamp_utc
        """, connection)
        if forecast.empty:
            return forecast, "No forecast has been issued yet", pd.DataFrame()
        forecast["target_timestamp_utc"] = pd.to_datetime(forecast["target_timestamp_utc"], utc=True)
        forecast["issued_at_utc"] = pd.to_datetime(forecast["issued_at_utc"], utc=True)
        # Derive display time from the canonical UTC instant. The persisted
        # timestamp_local column may be returned by the SQL connector as UTC.
        forecast["timestamp_local"] = forecast["target_timestamp_utc"].dt.tz_convert("America/Chicago")
        issue_age = (pd.Timestamp.now(tz="UTC") - forecast["issued_at_utc"].max()).total_seconds() / 3600
        status = f"Latest forecast issued {issue_age:.1f} hours ago"
        if issue_age > 26:
            status = f"Stale persisted forecast — last issued {issue_age:.1f} hours ago"
        try:
            monitoring = _query_frame(connection, """
                SELECT model_run_id, horizon_hours, lead_hours, target_date_utc,
                       evaluated_rows, mae_mw, rmse_mw, bias_mw
                FROM main.gold.forecast_monitoring_daily
                WHERE target_date_utc >= date_sub(current_date(), 30)
                ORDER BY target_date_utc
            """, connection)
        except Exception:
            monitoring = pd.DataFrame()
        return forecast, status, monitoring
    except Exception as error:
        return pd.DataFrame(), "Forecast unavailable. Check the Databricks connection and forecast table.", pd.DataFrame()
    finally:
        connection.close()
