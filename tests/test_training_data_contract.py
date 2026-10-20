import numpy as np
import pandas as pd
import pytest

from src.gold.forecast_example_features import generate_forecast_examples_df
from src.mlops.training_data_contract import (
    DEFAULT_EXAMPLES_TABLE,
    ROLLING_EXAMPLES_TABLE,
    clean_training_examples,
    training_origin_stride_hours,
    validate_examples_table,
)


def test_training_source_allows_daily_and_hourly_tables_only():
    assert validate_examples_table(DEFAULT_EXAMPLES_TABLE) == DEFAULT_EXAMPLES_TABLE
    assert validate_examples_table(ROLLING_EXAMPLES_TABLE) == ROLLING_EXAMPLES_TABLE
    with pytest.raises(ValueError, match="Unsupported training source"):
        validate_examples_table("main.gold.any_other_table")


def test_training_origin_stride_bounds_only_the_hourly_h168_dataset():
    assert training_origin_stride_hours(24, ROLLING_EXAMPLES_TABLE) == 1
    assert training_origin_stride_hours(168, ROLLING_EXAMPLES_TABLE) == 6
    assert training_origin_stride_hours(168, DEFAULT_EXAMPLES_TABLE) == 1


def _examples():
    timestamps = pd.date_range("2024-01-01", periods=1_200, freq="h", tz="UTC")
    demand = pd.DataFrame({
        "respondent": "ERCO",
        "timestamp_utc": timestamps,
        "actual_demand_mw": 50_000 + np.arange(len(timestamps), dtype=float),
        "is_holiday": False,
    })
    return generate_forecast_examples_df(demand, step_hours=24)


def test_training_contract_keeps_valid_examples_and_builds_residual_target():
    examples = _examples()

    cleaned = clean_training_examples(examples, 24)

    assert not cleaned.empty
    assert set(cleaned.horizon_hours) == {24}
    assert np.allclose(
        cleaned.residual_target_mw,
        cleaned.actual_demand_mw - cleaned.seasonal_naive_mw,
    )


def test_training_contract_rejects_target_misaligned_with_origin_and_lead():
    examples = _examples()
    row_index = examples.index[examples.horizon_hours == 24][0]
    examples.loc[row_index, "target_timestamp_utc"] += pd.Timedelta(hours=1)

    with pytest.raises(ValueError, match="origin plus lead_hours"):
        clean_training_examples(examples, 24)


def test_training_contract_rejects_observation_after_forecast_origin():
    examples = _examples()
    row_index = examples.index[examples.horizon_hours == 168][0]
    examples.loc[row_index, "source_max_observation_utc"] += pd.Timedelta(hours=1)

    with pytest.raises(ValueError, match="after their forecast origin"):
        clean_training_examples(examples, 168)


def test_training_contract_rejects_invalid_timestamp():
    examples = _examples()
    row_index = examples.index[examples.horizon_hours == 24][0]
    examples.loc[row_index, "forecast_origin_utc"] = pd.NaT

    with pytest.raises(ValueError, match="invalid forecast_origin_utc"):
        clean_training_examples(examples, 24)
