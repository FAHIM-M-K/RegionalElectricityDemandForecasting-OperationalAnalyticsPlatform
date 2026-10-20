"""Pure Python feature generation for issue-time hourly demand examples.

Kept outside Databricks notebooks so both the scheduled monthly builder and
the isolated hourly weather backtest can import the same implementation.
"""

import math
from typing import Iterable

import numpy as np
import pandas as pd

from src.shared.forecast_contract import (
    DEMAND_MAX_VALID_MW, DEMAND_MIN_VALID_MW, EXAMPLE_GRAIN_KEYS,
    FEATURE_SCHEMA_VERSION, FEATURE_SET_DEMAND_FOUNDATION, HORIZONS, LEAD_HOURS,
    features_for_horizon,
)


def _calendar_features(timestamp: pd.Timestamp, known: dict | None) -> dict:
    local = timestamp.tz_convert("America/Chicago")
    holiday = bool(known.get(timestamp, False)) if known else False
    return {
        "hour_local": local.hour,
        "day_of_week_local": local.dayofweek,
        "is_weekend": int(local.dayofweek >= 5),
        "is_holiday": int(holiday),
        "sin_hour": round(math.sin(2 * math.pi * local.hour / 24), 6),
        "cos_hour": round(math.cos(2 * math.pi * local.hour / 24), 6),
        "sin_dow": round(math.sin(2 * math.pi * local.dayofweek / 7), 6),
        "cos_dow": round(math.cos(2 * math.pi * local.dayofweek / 7), 6),
    }


def _exact_window(values: dict, origin: pd.Timestamp, hours: int) -> np.ndarray | None:
    """Return a complete trailing window ending at origin, or None on a gap."""
    timestamps = [origin - pd.Timedelta(hours=i) for i in range(hours)]
    result = [values.get(ts, np.nan) for ts in timestamps]
    if any(pd.isna(value) or not DEMAND_MIN_VALID_MW <= float(value) <= DEMAND_MAX_VALID_MW
           for value in result):
        return None
    return np.asarray(result, dtype=float)


def _examples_for_respondent(df: pd.DataFrame, horizons: Iterable[int], step_hours: int) -> list[dict]:
    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    duplicated = df.loc[df["timestamp_utc"].duplicated(keep=False), "timestamp_utc"]
    if not duplicated.empty:
        examples = duplicated.drop_duplicates().head(5).astype(str).tolist()
        raise ValueError(
            f"Duplicate demand timestamps for respondent {df['respondent'].iloc[0]}: {examples}"
        )
    df = df.sort_values("timestamp_utc")
    values = dict(zip(df["timestamp_utc"], df["actual_demand_mw"]))
    eia_values = df["day_ahead_forecast_mw"] if "day_ahead_forecast_mw" in df else pd.Series(np.nan, index=df.index)
    holiday_values = df["is_holiday"] if "is_holiday" in df else pd.Series(False, index=df.index)
    eia = dict(zip(df["timestamp_utc"], eia_values))
    holidays = dict(zip(df["timestamp_utc"], holiday_values))
    emergency_values = df["is_grid_emergency"] if "is_grid_emergency" in df else pd.Series(False, index=df.index)
    emergencies = dict(zip(df["timestamp_utc"], emergency_values.astype(bool)))
    respondent = str(df["respondent"].iloc[0])
    max_horizon = max(horizons)
    origins = df["timestamp_utc"].iloc[::step_hours]
    rows: list[dict] = []

    for origin in origins:
        if origin - pd.Timedelta(hours=336) not in values:
            continue
        if origin + pd.Timedelta(hours=max_horizon) not in values:
            continue
        history24, history168 = _exact_window(values, origin, 24), _exact_window(values, origin, 168)
        if history24 is None or history168 is None:
            continue
        history_timestamps = [origin - pd.Timedelta(hours=i) for i in range(337)]
        if any(emergencies.get(ts, False) for ts in history_timestamps):
            continue
        snapshot = {
            "demand_origin": values.get(origin),
            "demand_lag_24h_origin": values.get(origin - pd.Timedelta(hours=24)),
            "demand_lag_48h_origin": values.get(origin - pd.Timedelta(hours=48)),
            "demand_lag_168h_origin": values.get(origin - pd.Timedelta(hours=168)),
            "demand_lag_336h_origin": values.get(origin - pd.Timedelta(hours=336)),
            "demand_rolling_avg_24h_origin": float(history24.mean()),
            "demand_rolling_std_24h_origin": float(history24.std(ddof=0)),
            "demand_rolling_avg_168h_origin": float(history168.mean()),
        }
        if any(pd.isna(v) for v in snapshot.values()) or any(
            not (DEMAND_MIN_VALID_MW <= snapshot[name] <= DEMAND_MAX_VALID_MW)
            for name in ("demand_origin", "demand_lag_24h_origin", "demand_lag_48h_origin",
                         "demand_lag_168h_origin", "demand_lag_336h_origin")
        ):
            continue

        for horizon in horizons:
            for lead in LEAD_HOURS[horizon]:
                target = origin + pd.Timedelta(hours=lead)
                actual = values.get(target, np.nan)
                if emergencies.get(target, False):
                    continue
                row = {
                    "respondent": respondent,
                    "forecast_origin_utc": origin,
                    "target_timestamp_utc": target,
                    "horizon_hours": horizon,
                    "lead_hours": lead,
                    "feature_set": FEATURE_SET_DEMAND_FOUNDATION,
                    "feature_schema_version": "demand_foundation_v1",
                    "source_max_observation_utc": origin,
                    "actual_demand_mw": actual,
                    "daily_naive_mw": values.get(target - pd.Timedelta(hours=24), np.nan) if lead <= 24 else np.nan,
                    "weekly_naive_mw": values.get(target - pd.Timedelta(hours=168), np.nan),
                    "eia_day_ahead_forecast_mw": eia.get(target, np.nan),
                    **_calendar_features(target, holidays),
                    **snapshot,
                }
                row["seasonal_naive_mw"] = row["daily_naive_mw"] if horizon == 24 else row["weekly_naive_mw"]
                # This table is the demand-only rolling example foundation.
                # Weather is attached from archived issue-time vintages by the
                # dedicated training-table builder, never from realized weather.
                required = features_for_horizon(horizon, include_weather=False)
                row["is_valid_training_example"] = bool(
                    pd.notna(actual)
                    and DEMAND_MIN_VALID_MW <= actual <= DEMAND_MAX_VALID_MW
                    and not any(pd.isna(row[column]) for column in required)
                )
                rows.append(row)
    return rows


def generate_forecast_examples_df(
    df_demand: pd.DataFrame,
    horizons: list[int] | None = None,
    step_hours: int = 24,
) -> pd.DataFrame:
    """Create direct H24/H168 examples at a configurable origin cadence."""
    required = {"respondent", "timestamp_utc", "actual_demand_mw"}
    missing = required - set(df_demand.columns)
    if missing:
        raise ValueError(f"Forecast example source is missing columns: {sorted(missing)}")
    horizons = HORIZONS if horizons is None else horizons
    if not set(horizons).issubset(HORIZONS) or step_hours < 1:
        raise ValueError("Use supported horizons and a positive origin cadence.")
    frame = df_demand.copy()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    rows = []
    for _, group in frame.groupby("respondent", sort=False):
        rows.extend(_examples_for_respondent(group, horizons, step_hours))
    result = pd.DataFrame(rows)
    if not result.empty and result.duplicated(EXAMPLE_GRAIN_KEYS).any():
        raise ValueError("Duplicate forecast-example natural keys were generated.")
    return result
