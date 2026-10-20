"""Shared contract for direct ERCOT demand forecasts.

Each row represents one prediction for one target timestamp, issued at one
forecast origin.  H24 is day ahead and H168 is week ahead.  Training and
scoring import this module so their feature definitions cannot drift apart.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

HORIZON_H24 = 24
HORIZON_H168 = 168
HORIZONS: List[int] = [HORIZON_H24, HORIZON_H168]
LEAD_HOURS = {HORIZON_H24: range(1, 25), HORIZON_H168: range(1, 169)}

FEATURE_SET_OPERATIONAL = "operational_point_in_time_weather"
FEATURE_SET_DEMAND_FOUNDATION = "rolling_demand_foundation"
FEATURE_SCHEMA_VERSION = "v5.0_point_in_time_weather"
LEGACY_FEATURE_SCHEMA_VERSION = "v4.0_seasonal_residual_correction"
DEMAND_MIN_VALID_MW = 10_000.0
DEMAND_MAX_VALID_MW = 110_000.0

EXAMPLE_GRAIN_KEYS = [
    "respondent", "forecast_origin_utc", "target_timestamp_utc",
    "horizon_hours", "feature_set",
]
EVALUATION_GRAIN_KEYS = [
    "respondent", "forecast_issue_id", "horizon_hours", "target_timestamp_utc",
]
SERVING_GRAIN_KEYS = ["respondent", "horizon_hours", "target_timestamp_utc"]
BENCHMARK_GRAIN_KEYS = [
    "benchmark_provider", "respondent", "vintage_id", "target_timestamp_utc",
]

MAX_STALE_ORIGIN_HOURS = 4.0
MAX_FAIL_ORIGIN_HOURS = 6.0


def generate_forecast_issue_id(
    respondent: str, horizon_hours: int, forecast_origin_utc, issued_at_utc,
    model_run_id: str | None = None,
) -> str:
    """Return a deterministic, unique issue identifier separate from model_run_id."""
    def utc_key(value):
        if isinstance(value, str):
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if not isinstance(value, datetime):
            value = value.to_pydatetime()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    origin_str = utc_key(forecast_origin_utc)
    issued_str = utc_key(issued_at_utc)
    run_key = str(model_run_id or "unspecified")
    return f"issue:{respondent}:H{horizon_hours}:{origin_str}:{issued_str}:{run_key}"


WEATHER_FORECAST_FEATURES = [
    "weighted_temperature_f", "cdh_65_forecast", "hdh_65_forecast",
]
OPERATIONAL_FEATURES_H24_BASE = [
    "lead_hours", "seasonal_naive_mw", "hour_local", "day_of_week_local", "is_weekend", "is_holiday",
    "sin_hour", "cos_hour", "sin_dow", "cos_dow", "demand_origin",
    "demand_lag_24h_origin", "demand_lag_48h_origin", "demand_lag_168h_origin",
    "demand_rolling_avg_24h_origin", "demand_rolling_std_24h_origin",
]
OPERATIONAL_FEATURES_H168_BASE = [
    "lead_hours", "seasonal_naive_mw", "hour_local", "day_of_week_local", "is_weekend", "is_holiday",
    "sin_hour", "cos_hour", "sin_dow", "cos_dow", "demand_origin",
    "demand_lag_24h_origin", "demand_lag_168h_origin", "demand_lag_336h_origin",
    "demand_rolling_avg_24h_origin", "demand_rolling_avg_168h_origin",
]


def features_for_horizon(horizon: int, include_weather: bool = True) -> List[str]:
    if horizon == HORIZON_H24:
        base = OPERATIONAL_FEATURES_H24_BASE
    elif horizon == HORIZON_H168:
        base = OPERATIONAL_FEATURES_H168_BASE
    else:
        raise ValueError(f"Unsupported horizon: {horizon}")
    return list(base) + (list(WEATHER_FORECAST_FEATURES) if include_weather else [])


def features_for_schema(horizon: int, schema_version: str) -> List[str]:
    if schema_version == FEATURE_SCHEMA_VERSION:
        return features_for_horizon(horizon, include_weather=True)
    if schema_version == LEGACY_FEATURE_SCHEMA_VERSION:
        return features_for_horizon(horizon, include_weather=False)
    raise ValueError(f"Unsupported H{horizon} feature schema: {schema_version!r}")


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


def validate_forecast_issue_frame(df, expected_horizons: List[int] | None = None) -> None:
    """Pre-publish validation asserting complete curves and physical bounds."""
    if expected_horizons is None:
        expected_horizons = HORIZONS
    required = {
        "respondent", "forecast_issue_id", "forecast_origin_utc", "target_timestamp_utc",
        "issued_at_utc", "horizon_hours", "lead_hours", "predicted_demand_mw",
    }
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f"Forecast issue is missing required columns: {sorted(missing_columns)}")
    if df[list(required)].isna().any().any():
        raise ValueError("Forecast issue contains null identity, time, horizon, lead, or prediction values.")
    for horizon in expected_horizons:
        subset = df[df["horizon_hours"] == horizon]
        if subset.empty:
            raise ValueError(f"Forecast issue is missing H{horizon}.")
        if subset["respondent"].nunique() != 1 or subset["forecast_origin_utc"].nunique() != 1:
            raise ValueError(f"H{horizon} issue mixes respondents or forecast origins.")
        if subset["forecast_issue_id"].nunique() != 1:
            raise ValueError(f"H{horizon} issue has more than one issue identifier.")
        if (subset["issued_at_utc"].nunique() != 1):
            raise ValueError(f"H{horizon} issue mixes issued-at timestamps.")
        if (subset["target_timestamp_utc"] - subset["forecast_origin_utc"] !=
                subset["lead_hours"].map(lambda lead: timedelta(hours=int(lead)))).any():
            raise ValueError(f"H{horizon} contains a target timestamp inconsistent with its lead.")
        if subset["lead_hours"].duplicated().any():
            raise ValueError(f"H{horizon} contains duplicate lead hours.")
    for horizon in expected_horizons:
        subset = df[df["horizon_hours"] == horizon]
        expected_count = len(LEAD_HOURS[horizon])
        if len(subset) != expected_count:
            raise ValueError(
                f"Incomplete H{horizon} curve: expected {expected_count} leads, found {len(subset)}."
            )
        leads = sorted(subset["lead_hours"].tolist())
        if leads != list(LEAD_HOURS[horizon]):
            raise ValueError(f"H{horizon} curve has missing or non-continuous leads: {leads[:5]}...")
        preds = subset["predicted_demand_mw"].dropna()
        if len(preds) != expected_count:
            raise ValueError(f"H{horizon} contains null predicted_demand_mw values.")
        if (preds < DEMAND_MIN_VALID_MW).any() or (preds > DEMAND_MAX_VALID_MW).any():
            raise ValueError(f"H{horizon} predictions exceed physical bounds ({DEMAND_MIN_VALID_MW}-{DEMAND_MAX_VALID_MW} MW).")
        if not all(float(value) == float(value) and abs(float(value)) != float("inf") for value in preds):
            raise ValueError(f"H{horizon} predictions contain non-finite values.")


def validate_benchmark_vintage_record(record: Dict[str, Any]) -> None:
    """Validate an append-only benchmark vintage record."""
    for key in ("benchmark_provider", "respondent", "vintage_id", "target_timestamp_utc", "forecast_mw"):
        if record.get(key) is None:
            raise ValueError(f"Benchmark vintage missing required field: {key}")
    if record.get("first_observed_at_utc") is None:
        raise ValueError("Benchmark vintage requires first_observed_at_utc availability timestamp.")
    val = float(record["forecast_mw"])
    if not (DEMAND_MIN_VALID_MW <= val <= DEMAND_MAX_VALID_MW):
        raise ValueError(f"Benchmark value {val} MW is outside ERCOT plausibility bounds.")
