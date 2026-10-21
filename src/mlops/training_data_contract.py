"""Validation for point-in-time demand-model training examples.

This module is pure Python so the same checks can be exercised locally and
called by Databricks training notebooks before fitting a candidate model.
"""

from __future__ import annotations

import pandas as pd

from src.shared.forecast_contract import EXAMPLE_GRAIN_KEYS, features_for_horizon

ROLLING_EXAMPLES_TABLE = "main.gold.forecast_examples_hourly"
WEATHER_ROLLING_EXAMPLES_TABLE = "main.gold.forecast_examples_hourly_weather"
ALLOWED_EXAMPLES_TABLES = {ROLLING_EXAMPLES_TABLE, WEATHER_ROLLING_EXAMPLES_TABLE}


def validate_examples_table(table_name: str) -> str:
    """Allow only the active rolling demand or issue-time-weather source."""
    if table_name not in ALLOWED_EXAMPLES_TABLES:
        raise ValueError(
            f"Unsupported training source {table_name!r}; choose one of "
            f"{sorted(ALLOWED_EXAMPLES_TABLES)}."
        )
    return table_name


def clean_training_examples(examples: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Return valid examples for one horizon after enforcing temporal contracts."""
    features = features_for_horizon(horizon)
    required = set(EXAMPLE_GRAIN_KEYS) | {
        "horizon_hours", "lead_hours", "forecast_origin_utc", "target_timestamp_utc",
        "source_max_observation_utc", "actual_demand_mw", "seasonal_naive_mw",
        "is_valid_training_example", *features,
    }
    missing = required - set(examples.columns)
    if missing:
        raise ValueError(f"Training examples are missing required columns: {sorted(missing)}")

    data = examples.loc[examples["horizon_hours"] == horizon].copy()
    if data.empty:
        raise ValueError(f"No H{horizon} examples available for training.")

    time_columns = ["forecast_origin_utc", "target_timestamp_utc", "source_max_observation_utc"]
    for column in time_columns:
        data[column] = pd.to_datetime(data[column], utc=True, errors="coerce")
        if data[column].isna().any():
            raise ValueError(f"H{horizon} training examples contain an invalid {column}.")

    lead = pd.to_numeric(data["lead_hours"], errors="coerce")
    if lead.isna().any() or (lead % 1 != 0).any():
        raise ValueError(f"H{horizon} training examples contain a non-integer lead_hours value.")
    if not lead.between(1, horizon).all():
        raise ValueError(f"H{horizon} training examples contain leads outside 1..{horizon}.")
    if not (data["target_timestamp_utc"] - data["forecast_origin_utc"] ==
            pd.to_timedelta(lead.astype("int64"), unit="h")).all():
        raise ValueError(f"H{horizon} target timestamps do not equal origin plus lead_hours.")
    if (data["source_max_observation_utc"] > data["forecast_origin_utc"]).any():
        raise ValueError(f"H{horizon} examples include demand observations after their forecast origin.")

    # Keep the established eligibility flag authoritative. The additional null
    # filter protects against malformed rows introduced by a source/table edit.
    valid = data.loc[data["is_valid_training_example"].fillna(False).astype(bool)].copy()
    valid = valid.dropna(subset=features + ["actual_demand_mw", "seasonal_naive_mw"])
    if valid.empty:
        raise ValueError(f"No valid H{horizon} examples remain after contract checks.")
    valid["residual_target_mw"] = valid["actual_demand_mw"] - valid["seasonal_naive_mw"]
    return valid.sort_values("forecast_origin_utc", kind="stable")
