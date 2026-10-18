"""Shared contract for direct ERCOT demand forecasts.

Each row represents one prediction for one target timestamp, issued at one
forecast origin.  H24 is day ahead and H168 is week ahead.  Training and
scoring import this module so their feature definitions cannot drift apart.
"""

from datetime import timedelta
from typing import Any, Dict, List

HORIZON_H24 = 24
HORIZON_H168 = 168
HORIZONS: List[int] = [HORIZON_H24, HORIZON_H168]
LEAD_HOURS = {HORIZON_H24: range(1, 25), HORIZON_H168: range(1, 169)}

FEATURE_SET_OPERATIONAL = "operational_no_weather"
FEATURE_SCHEMA_VERSION = "v4.0_seasonal_residual_correction"
DEMAND_MIN_VALID_MW = 10_000.0
DEMAND_MAX_VALID_MW = 110_000.0

EXAMPLE_GRAIN_KEYS = [
    "respondent", "forecast_origin_utc", "target_timestamp_utc",
    "horizon_hours", "feature_set",
]
EVALUATION_GRAIN_KEYS = [
    "model_run_id", "horizon_hours", "forecast_origin_utc", "target_timestamp_utc",
]
SERVING_GRAIN_KEYS = ["respondent", "horizon_hours", "target_timestamp_utc"]

OPERATIONAL_FEATURES_H24 = [
    "lead_hours", "seasonal_naive_mw", "hour_local", "day_of_week_local", "is_weekend", "is_holiday",
    "sin_hour", "cos_hour", "sin_dow", "cos_dow", "demand_origin",
    "demand_lag_24h_origin", "demand_lag_48h_origin", "demand_lag_168h_origin",
    "demand_rolling_avg_24h_origin", "demand_rolling_std_24h_origin",
]
OPERATIONAL_FEATURES_H168 = [
    "lead_hours", "seasonal_naive_mw", "hour_local", "day_of_week_local", "is_weekend", "is_holiday",
    "sin_hour", "cos_hour", "sin_dow", "cos_dow", "demand_origin",
    "demand_lag_24h_origin", "demand_lag_168h_origin", "demand_lag_336h_origin",
    "demand_rolling_avg_24h_origin", "demand_rolling_avg_168h_origin",
]


def features_for_horizon(horizon: int) -> List[str]:
    if horizon == HORIZON_H24:
        return OPERATIONAL_FEATURES_H24
    if horizon == HORIZON_H168:
        return OPERATIONAL_FEATURES_H168
    raise ValueError(f"Unsupported horizon: {horizon}")


def validate_example_record(record: Dict[str, Any]) -> None:
    """Raise when a row breaks the point-in-time or physical contract."""
    origin, target = record.get("forecast_origin_utc"), record.get("target_timestamp_utc")
    horizon, lead = record.get("horizon_hours"), record.get("lead_hours")
    if origin is None or target is None or horizon is None or lead is None:
        raise ValueError("Forecast example requires origin, target, horizon, and lead.")
    if horizon not in HORIZONS or lead not in LEAD_HOURS[horizon]:
        raise ValueError(f"Invalid horizon/lead combination: H{horizon}, lead {lead}.")
    if target - origin != timedelta(hours=int(lead)):
        raise ValueError("Target timestamp must equal origin plus lead_hours.")
    actual = record.get("actual_demand_mw")
    if actual is not None and not (DEMAND_MIN_VALID_MW <= actual <= DEMAND_MAX_VALID_MW):
        raise ValueError("Actual demand is outside configured ERCOT plausibility bounds.")
