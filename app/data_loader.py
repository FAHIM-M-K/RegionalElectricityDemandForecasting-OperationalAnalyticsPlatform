"""Read the current real forecast issue; never fabricate an offline forecast."""

import os
import logging
from time import perf_counter

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def _query_frame(connection, query):
    """Run a Databricks SQL query without pandas' unsupported-DBAPI warning."""
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        columns = [description[0] for description in cursor.description]
        return pd.DataFrame.from_records(cursor.fetchall(), columns=columns)
    except KeyboardInterrupt:
        # Ask Databricks to cancel the server-side statement when the operator
        # interrupts Streamlit, so shutdown is not waiting on the SQL request.
        try:
            cursor.cancel()
        except Exception:
            logger.exception("Could not cancel the interrupted Databricks query.")
        raise
    finally:
        cursor.close()


def get_databricks_connection():
    host, path, token = (os.environ.get(key, "").strip() for key in ("DATABRICKS_HOST", "DATABRICKS_HTTP_PATH", "DATABRICKS_TOKEN"))
    if not all((host, path, token)):
        return None
    from databricks import sql
    return sql.connect(
        server_hostname=host.removeprefix("https://").removeprefix("http://").rstrip("/"),
        http_path=path,
        access_token=token,
    )


@st.cache_data(ttl=300)
def load_operational_forecast_data(cache_contract_version="h24-only-v3"):
    """Return the latest persisted forecast and an honest connection status.

    The version argument is part of Streamlit's cache key. Bump it whenever
    the returned tuple contract changes so older cached shapes are discarded.
    """
    started = perf_counter()
    logger.info("Connecting to Databricks for the forecast dashboard (cache %s).", cache_contract_version)
    connection = None
    try:
        connection = get_databricks_connection()
        if connection is None:
            return (
                pd.DataFrame(),
                "Databricks connection is not configured",
                pd.DataFrame(),
                "Monitoring data is unavailable because Databricks is not configured.",
            )
        logger.info("Databricks connected; querying latest forecast rows.")
        forecast = _query_frame(connection, """
            WITH latest_actual AS (
                SELECT timestamp_utc, actual_demand_mw
                FROM main.gold.fact_hourly_electricity_demand
                WHERE respondent = 'ERCO' AND actual_demand_mw IS NOT NULL
                ORDER BY timestamp_utc DESC
                LIMIT 1
            )
            SELECT f.respondent, f.forecast_origin_utc, f.issued_at_utc, f.target_timestamp_utc, f.timestamp_local,
                   f.horizon_hours, f.lead_hours, f.hour_local, f.predicted_demand_mw,
                   f.forecast_source, f.model_version, f.model_run_id,
                   a.timestamp_utc AS latest_actual_timestamp_utc,
                   a.actual_demand_mw AS latest_actual_demand_mw
            FROM main.gold.forecast_latest f
            LEFT JOIN latest_actual a ON TRUE
            WHERE f.respondent = 'ERCO' AND f.horizon_hours = 24
            ORDER BY f.target_timestamp_utc
        """)
        logger.info("Latest forecast query returned %s rows.", len(forecast))
        if forecast.empty:
            return forecast, "No forecast has been issued yet", pd.DataFrame(), "Monitoring was not queried because no forecast is available."
        forecast["target_timestamp_utc"] = pd.to_datetime(forecast["target_timestamp_utc"], utc=True)
        forecast["issued_at_utc"] = pd.to_datetime(forecast["issued_at_utc"], utc=True)
        forecast["latest_actual_timestamp_utc"] = pd.to_datetime(
            forecast["latest_actual_timestamp_utc"], utc=True
        )
        # Derive display time from the canonical UTC instant. The persisted
        # timestamp_local column may be returned by the SQL connector as UTC.
        forecast["timestamp_local"] = forecast["target_timestamp_utc"].dt.tz_convert("America/Chicago")
        issue_age = (pd.Timestamp.now(tz="UTC") - forecast["issued_at_utc"].max()).total_seconds() / 3600
        status = f"Latest forecast issued {issue_age:.1f} hours ago"
        if issue_age > 26:
            status = f"Stale persisted forecast — last issued {issue_age:.1f} hours ago"
        try:
            logger.info("Querying forecast monitoring metrics.")
            monitoring = _query_frame(connection, """
                SELECT model_run_id, horizon_hours, lead_hours, target_date_utc,
                       evaluated_rows, mae_mw, rmse_mw, bias_mw
                FROM main.gold.forecast_monitoring_daily
                WHERE horizon_hours = 24 AND target_date_utc >= date_sub(current_date(), 30)
                ORDER BY target_date_utc
            """)
            logger.info("Forecast monitoring query returned %s rows.", len(monitoring))
            monitoring_status = (
                "Monitoring query succeeded but returned no rows yet."
                if monitoring.empty
                else "Monitoring data loaded."
            )
        except Exception:
            logger.exception("Forecast loaded, but monitoring query failed.")
            monitoring = pd.DataFrame()
            monitoring_status = "Reliability data could not be loaded. Check the Streamlit terminal logs."
        logger.info("Dashboard data loaded in %.1f seconds.", perf_counter() - started)
        return forecast, status, monitoring, monitoring_status
    except Exception as error:
        logger.exception("Could not load forecast dashboard data.")
        return (
            pd.DataFrame(),
            f"Forecast unavailable ({type(error).__name__}). Check the Streamlit terminal for details.",
            pd.DataFrame(),
            "Monitoring data is unavailable because the Databricks connection failed.",
        )
    finally:
        if connection is not None:
            connection.close()
