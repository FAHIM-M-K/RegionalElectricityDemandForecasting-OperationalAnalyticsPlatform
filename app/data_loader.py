import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv()

def get_databricks_connection():
    """
    Attempts to establish a live connection to Databricks SQL Warehouse.
    Returns connection object or None.
    """
    server_hostname = os.environ.get("DATABRICKS_HOST", "").strip()
    http_path = os.environ.get("DATABRICKS_HTTP_PATH", "").strip()
    access_token = os.environ.get("DATABRICKS_TOKEN", "").strip()

    if not (server_hostname and http_path and access_token):
        return None

    # Strip protocol and trailing slashes if present
    if server_hostname.startswith("https://"):
        server_hostname = server_hostname[len("https://"):]
    if server_hostname.startswith("http://"):
        server_hostname = server_hostname[len("http://"):]
    server_hostname = server_hostname.rstrip("/")

    try:
        from databricks import sql
        connection = sql.connect(
            server_hostname=server_hostname,
            http_path=http_path,
            access_token=access_token
        )
        return connection
    except Exception as e:
        print(f"Warning: Databricks SQL connection failed: {e}")
        return None

import streamlit as st

@st.cache_data(ttl=300)
def load_operational_forecast_data():
    """
    Loads hourly forecast and demand data.
    First checks Databricks SQL; falls back to local cache or synthetic grid generator.
    """
    conn = get_databricks_connection()
    if conn:
        try:
            query = """
            SELECT 
                f.respondent,
                f.timestamp_utc,
                f.timestamp_local,
                f.date_local,
                f.hour_local,
                f.day_of_week_local,
                f.actual_demand_mw,
                f.day_ahead_forecast_mw,
                f.predicted_demand_mw,
                f.prediction_lower_95_mw,
                f.prediction_upper_95_mw,
                f.is_predicted_peak_risk,
                f.model_error_mw,
                f.model_abs_error_mw,
                f.eia_error_mw,
                f.eia_abs_error_mw,
                f.is_model_better_than_eia,
                f.model_version,
                feat.weighted_temperature_f
            FROM main.gold.forecast_hourly f
            LEFT JOIN main.gold.features_hourly feat
              ON f.respondent = feat.respondent AND f.timestamp_utc = feat.timestamp_utc
            WHERE f.respondent = 'ERCO'
            ORDER BY f.timestamp_utc DESC
            LIMIT 8760
            """
            df = pd.read_sql(query, conn)
            conn.close()
            if len(df) > 0:
                df = df.sort_values("timestamp_utc").reset_index(drop=True)
                df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
                df["timestamp_local"] = df["timestamp_utc"].dt.tz_convert("America/Chicago")
                return df, "Databricks Unity Catalog (Live)"
        except Exception as e:
            err_msg = str(e)
            print(f"Databricks SQL query error: {err_msg}")
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
            if "TABLE_OR_VIEW_NOT_FOUND" in err_msg or "forecast_hourly" in err_msg:
                # Connected to Databricks successfully, but scoring notebook hasn't been executed yet
                df_sim = generate_simulation_data()
                return df_sim, "Databricks Connected (Run score_forecast_hourly.py in Databricks to materialize live table)"

    # Fallback if connection not configured
    df_sim = generate_simulation_data()
    return df_sim, "Local Simulation Mode"

def generate_simulation_data():
    """
    Generates high-fidelity ERCOT operational simulation data.
    Covers past 14 days of actuals + next 48 hours of forward forecasts.
    """
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    start_time = now - timedelta(days=14)
    end_time = now + timedelta(hours=48)
    
    date_range = pd.date_range(start=start_time, end=end_time, freq="h")
    np.random.seed(42)
    
    rows = []
    for ts in date_range:
        hour = ts.hour
        dow = ts.dayofweek
        is_weekend = 1 if dow >= 5 else 0
        
        # Diurnal base profile (MW)
        diurnal_factor = np.sin((hour - 5) * np.pi / 12)  # Peak around 17:00
        base_load = 52000.0 + (16000.0 * diurnal_factor) - (4000.0 * is_weekend)
        
        # Seasonal summer heatwave simulation
        temp_f = 75.0 + 22.0 * np.sin((hour - 9) * np.pi / 12) + np.random.normal(0, 2.0)
        cooling_mw = max(0.0, temp_f - 55.2) * 850.0
        
        true_demand = base_load + cooling_mw + np.random.normal(0, 450.0)
        
        # Peak stress hours (ERCOT heatwave spike up to 82,000 MW)
        if 15 <= hour <= 18 and not is_weekend:
            true_demand += 4500.0
            
        # Is this in the future horizon (next 48h)?
        is_future = ts > now
        actual_demand = None if is_future else round(true_demand, 1)
        
        # Institutional EIA Day-Ahead Forecast: known +368 MW overforecasting bias & afternoon lag
        eia_noise = np.random.normal(368.0, 1150.0)
        if 14 <= hour <= 18:
            eia_noise -= 850.0  # Underpredicts peak ramps
        eia_forecast = round(true_demand + eia_noise, 1)
        
        # LightGBM Production Hybrid: error-corrects EIA bias
        model_noise = np.random.normal(25.0, 780.0)
        model_forecast = round(true_demand + model_noise, 1)
        
        # 95% Confidence Interval (1.96 * 1587 MW)
        ci_half = 1.96 * 1587.6
        lower_95 = round(model_forecast - ci_half, 1)
        upper_95 = round(model_forecast + ci_half, 1)
        
        # Risk flag
        is_peak_risk = model_forecast >= 75000.0
        
        # Errors
        model_err = round(model_forecast - actual_demand, 1) if actual_demand is not None else None
        model_abs_err = abs(model_err) if model_err is not None else None
        eia_err = round(eia_forecast - actual_demand, 1) if actual_demand is not None else None
        eia_abs_err = abs(eia_err) if eia_err is not None else None
        is_better = (model_abs_err < eia_abs_err) if (model_abs_err is not None and eia_abs_err is not None) else None
        
        rows.append({
            "respondent": "ERCO",
            "timestamp_utc": ts,
            "timestamp_local": ts,
            "date_local": ts.date(),
            "hour_local": hour,
            "day_of_week_local": dow,
            "actual_demand_mw": actual_demand,
            "day_ahead_forecast_mw": eia_forecast,
            "predicted_demand_mw": model_forecast,
            "prediction_lower_95_mw": lower_95,
            "prediction_upper_95_mw": upper_95,
            "is_predicted_peak_risk": is_peak_risk,
            "model_error_mw": model_err,
            "model_abs_error_mw": model_abs_err,
            "eia_error_mw": eia_err,
            "eia_abs_error_mw": eia_abs_err,
            "is_model_better_than_eia": is_better,
            "model_version": "LightGBM_Hybrid_e4b1092a",
            "weighted_temperature_f": round(temp_f, 1)
        })

    return pd.DataFrame(rows)

def load_feature_importance():
    """
    Returns empirical gain-based feature importances from LightGBM Champion.
    Queries Databricks SQL main.gold.feature_importance if available,
    checks local cached sync data/feature_importance.csv, or falls back to baseline.
    """
    conn = get_databricks_connection()
    if conn:
        try:
            query = """
            SELECT 
                feature_name AS Feature,
                gain_importance AS Gain,
                feature_description AS Description
            FROM main.gold.feature_importance
            ORDER BY gain_importance DESC
            LIMIT 15
            """
            cursor = conn.cursor()
            cursor.execute(query)
            cols = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            cursor.close()
            if rows:
                return pd.DataFrame(rows, columns=cols)
        except Exception as e:
            print(f"Notice: Failed to load feature importance from Databricks SQL: {e}")

    # Check local synced CSV
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "feature_importance.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if "feature_name" in df.columns and "gain_importance" in df.columns:
                df = df.rename(columns={
                    "feature_name": "Feature",
                    "gain_importance": "Gain",
                    "feature_description": "Description"
                })
                return df[["Feature", "Gain", "Description"]].head(15)
        except Exception as e:
            print(f"Notice: Failed to load local feature importance CSV: {e}")

    # Default fallback
    features = [
        {"Feature": "weighted_apparent_temperature_f", "Gain": 45210.5, "Description": "Population-weighted Heat Index / Wind Chill"},
        {"Feature": "hour_local", "Gain": 38940.2, "Description": "Discrete diurnal cycle indicator"},
        {"Feature": "temp_rolling_24h", "Gain": 24810.0, "Description": "24h thermal rolling heat accumulation"},
        {"Feature": "cdh_55", "Gain": 21650.8, "Description": "Cooling Degree Hours above 55.2°F"},
        {"Feature": "temp_diff_24h", "Gain": 18420.1, "Description": "Day-over-day temperature shift"},
        {"Feature": "peak_temp_interaction", "Gain": 15890.3, "Description": "Late afternoon (14-18h) thermal boost"},
        {"Feature": "day_of_week_local", "Gain": 12150.0, "Description": "Day of week cycle"},
        {"Feature": "weighted_solar_irradiance_wm2", "Gain": 9840.4, "Description": "Direct normal solar irradiance (cooling drive)"},
        {"Feature": "demand_ratio_24_48", "Gain": 8410.6, "Description": "Scale-invariant momentum ratio"},
        {"Feature": "weekend_diff_24h", "Gain": 7250.0, "Description": "Friday-to-Saturday drop transition"}
    ]
    return pd.DataFrame(features)

