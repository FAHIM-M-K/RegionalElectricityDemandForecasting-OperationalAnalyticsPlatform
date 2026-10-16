"""
Shared Feature Transformations & Column Schema
Regional Electricity Demand Forecasting Platform (ERCOT)

Single source of truth for model feature names, base Delta table columns,
and in-memory physical/calendar feature derivation. Used by both training
(retrain_model) and inference (score_forecast_hourly) to guarantee zero train/serve feature skew.
"""

import numpy as np
import pandas as pd

# Scale-invariant stationary features: strictly observable at day-ahead origin (t0 = T-24)
FEATURE_COLS = [
    # 1. Target-Hour Thermodynamics & Humidity (Stationary across years)
    "weighted_temperature_f",
    "weighted_apparent_temperature_f",
    "weighted_solar_irradiance_wm2",
    "cdh_55",
    "hdh_55",
    "temp_deviation_55_sq",
    "temp_delta_dfw_houston_c",
    "temp_rolling_24h",

    # 2. Weather Differences vs Yesterday (t-24) — The Primary Physical Shift Drivers
    "temp_diff_24h",
    "apparent_temp_diff_24h",
    "cdh_diff_24h",
    "hdh_diff_24h",
    "peak_temp_interaction",

    # 3. Discrete Temporal Features (Enables Sharp Tree Partitions)
    "hour_local",
    "day_of_week_local",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",

    # 4. Calendar & Day-over-Day Step Transitions
    "is_weekend",
    "is_holiday",
    "is_major_shutdown_holiday",
    "is_holiday_adjacent",
    "weekend_diff_24h",
    "holiday_diff_24h",
    "season_summer",
    "season_winter",
    "season_shoulder",

    # 5. Scale-Invariant Load Ratios & Volatility (No raw non-stationary MW)
    "demand_ratio_24_48",
    "demand_ratio_24_168",
    "demand_rolling_cv_24h",
]

# Base columns required from main.gold.features_hourly
BASE_COLS = [
    "respondent",
    "timestamp_utc",
    "timestamp_local",
    "date_local",
    "hour_local",
    "day_of_week_local",
    "actual_demand_mw",
    "day_ahead_forecast_mw",
    "is_valid_demand",
    "cdh_55",
    "hdh_55",
    "temp_deviation_55_sq",
    "temp_delta_dfw_houston_c",
    "weighted_temperature_f",
    "weighted_apparent_temperature_f",
    "weighted_solar_irradiance_wm2",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "is_weekend",
    "is_holiday",
    "is_major_shutdown_holiday",
    "is_holiday_adjacent",
    "season_summer",
    "season_winter",
    "season_shoulder",
    "demand_lag_24h",
    "demand_lag_48h",
    "demand_lag_168h",
    "demand_rolling_avg_24h_lag24",
    "demand_rolling_std_24h_lag24",
]

TARGET_COL = "actual_demand_mw"
BENCHMARK_COL = "day_ahead_forecast_mw"
TIME_COL = "timestamp_utc"


def compute_derived_features(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Computes rolling thermodynamic, calendar transition, and momentum ratio features
    on a pandas DataFrame sorted by timestamp_utc.

    Parameters
    ----------
    pdf : pd.DataFrame
        Must contain base weather, calendar, and demand lag columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with all 33 FEATURE_COLS populated.
    """
    df = pdf.copy()
    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
        df = df.sort_values(TIME_COL).reset_index(drop=True)

    # 1. Weather shifts from yesterday at the exact same hour
    df["temp_diff_24h"] = (
        df["weighted_temperature_f"] - df["weighted_temperature_f"].shift(24)
    ).fillna(0.0)
    df["apparent_temp_diff_24h"] = (
        df["weighted_apparent_temperature_f"]
        - df["weighted_apparent_temperature_f"].shift(24)
    ).fillna(0.0)
    df["cdh_diff_24h"] = (df["cdh_55"] - df["cdh_55"].shift(24)).fillna(0.0)
    df["hdh_diff_24h"] = (df["hdh_55"] - df["hdh_55"].shift(24)).fillna(0.0)

    # 2. Thermal rolling inertia (24h average) & peak-hour interaction
    df["temp_rolling_24h"] = (
        df["weighted_temperature_f"]
        .rolling(24, min_periods=1)
        .mean()
        .fillna(df["weighted_temperature_f"])
    )
    df["peak_temp_interaction"] = (
        df["hour_local"].between(14, 18).astype(float) * df["temp_diff_24h"]
    )

    # 3. Calendar transitions from yesterday (e.g. Friday -> Saturday = -1.0)
    df["weekend_diff_24h"] = (
        df["is_weekend"].astype(float) - df["is_weekend"].astype(float).shift(24)
    ).fillna(0.0)
    df["holiday_diff_24h"] = (
        df["is_holiday"].astype(float) - df["is_holiday"].astype(float).shift(24)
    ).fillna(0.0)

    # 4. Scale-invariant load momentum ratios (normalized by base load to eliminate grid growth bias)
    df["demand_ratio_24_48"] = (
        (df["demand_lag_24h"] - df["demand_lag_48h"])
        / df["demand_lag_48h"].clip(lower=1000.0)
    ).fillna(0.0)
    df["demand_ratio_24_168"] = (
        (df["demand_lag_24h"] - df["demand_lag_168h"])
        / df["demand_lag_168h"].clip(lower=1000.0)
    ).fillna(0.0)
    df["demand_rolling_cv_24h"] = (
        df["demand_rolling_std_24h_lag24"]
        / df["demand_rolling_avg_24h_lag24"].clip(lower=1000.0)
    ).fillna(0.0)

    # Ensure boolean flags are float for tree estimators
    for c in [
        "is_weekend",
        "is_holiday",
        "is_major_shutdown_holiday",
        "is_holiday_adjacent",
        "season_summer",
        "season_winter",
        "season_shoulder",
    ]:
        if c in df.columns:
            df[c] = df[c].astype(float)

    return df
