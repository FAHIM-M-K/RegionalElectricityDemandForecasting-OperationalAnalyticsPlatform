"""Small, local safeguards for the production H24 forecast path."""

import json

import numpy as np
import pandas as pd
import pytest

from src.gold.forecast_example_features import generate_forecast_examples_df
from src.gold.score_forecast_hourly import generate_independent_forecasts
from src.mlops.backtest_multihorizon import evaluate_promotion_gate, generate_chronological_splits
from src.mlops.training_data_contract import clean_training_examples
from src.mlops.weather_api_budget import ApiRequestBudget, ApiRequestBudgetExceeded
from src.mlops.weather_training_data import attach_weather_asof_issue
from src.shared.ercot_zones import canonical_zone_code, source_to_zone_code_map
from src.shared.forecast_contract import (
    features_for_horizon,
    generate_forecast_issue_id,
    validate_forecast_issue_frame,
)


def _demand(hours=800):
    ts = pd.date_range("2024-01-01", periods=hours, freq="h", tz="UTC")
    return pd.DataFrame({
        "respondent": "ERCO", "timestamp_utc": ts,
        "actual_demand_mw": 50_000 + np.arange(hours, dtype=float), "is_holiday": False,
    })


def test_training_examples_reject_gaps_and_keep_issue_time_alignment():
    demand = _demand()
    gap = pd.Timestamp("2024-01-17 00:00", tz="UTC")
    examples = generate_forecast_examples_df(demand[demand.timestamp_utc != gap], step_hours=1)
    assert not (examples.forecast_origin_utc == pd.Timestamp("2024-01-18", tz="UTC")).any()
    assert (examples.target_timestamp_utc - examples.forecast_origin_utc ==
            pd.to_timedelta(examples.lead_hours, unit="h")).all()


def test_training_contract_rejects_future_observations_and_misaligned_targets():
    demand = _demand(1_200)
    examples = generate_forecast_examples_df(demand, step_hours=24)
    examples["weighted_temperature_f"] = 70.0
    examples["cdh_65_forecast"] = 5.0
    examples["hdh_65_forecast"] = 0.0
    row = examples.index[0]

    future = examples.copy()
    future.loc[row, "source_max_observation_utc"] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="after their forecast origin"):
        clean_training_examples(future, 24)

    misaligned = examples.copy()
    misaligned.loc[row, "target_timestamp_utc"] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="origin plus lead_hours"):
        clean_training_examples(misaligned, 24)


def test_weather_join_uses_only_issue_available_vintages():
    origin = pd.Timestamp("2026-01-01T00:00:00Z")
    examples = pd.DataFrame({
        "respondent": ["ERCO", "ERCO"], "forecast_origin_utc": [origin] * 2,
        "target_timestamp_utc": [origin + pd.Timedelta(hours=h) for h in (1, 2)],
        "horizon_hours": [2, 2], "lead_hours": [1, 2],
    })
    weather = pd.DataFrame({
        "respondent": ["ERCO"] * 3,
        "target_timestamp_utc": [origin + pd.Timedelta(hours=h) for h in (1, 1, 2)],
        "weather_available_at_utc": [origin - pd.Timedelta(minutes=2), origin + pd.Timedelta(hours=2), origin - pd.Timedelta(minutes=2)],
        "weather_run_init_utc": [origin - pd.Timedelta(hours=6), origin, origin - pd.Timedelta(hours=6)],
        "weighted_temperature_f": [50.0, 90.0, 51.0],
        "cdh_65_forecast": [0.0, 25.0, 0.0], "hdh_65_forecast": [15.0, 0.0, 14.0],
    })
    selected, coverage = attach_weather_asof_issue(examples, weather)
    assert selected.weighted_temperature_f.tolist() == [50.0, 51.0]
    assert coverage.complete_weather_horizon.all()


def test_serving_emits_one_complete_h24_curve():
    class SeasonalBaseline:
        def predict(self, frame):
            return np.zeros(len(frame))

    demand = _demand(500)
    calendar = set(pd.date_range("2024-01-01", periods=40, freq="D").date)
    forecast = generate_independent_forecasts(
        demand, {24: SeasonalBaseline()}, {24: "test-run"},
        holiday_dates=set(), calendar_dates=calendar,
    )
    assert set(forecast.horizon_hours) == {24}
    assert sorted(forecast.lead_hours) == list(range(1, 25))


def test_publish_contract_rejects_incomplete_or_out_of_bounds_curve():
    origin = pd.Timestamp("2026-10-19T05:00:00Z")
    issue_id = generate_forecast_issue_id("ERCO", 24, origin, origin + pd.Timedelta(hours=1))
    curve = pd.DataFrame({
        "respondent": "ERCO", "forecast_issue_id": issue_id,
        "forecast_origin_utc": origin, "issued_at_utc": origin + pd.Timedelta(hours=1),
        "target_timestamp_utc": [origin + pd.Timedelta(hours=h) for h in range(1, 25)],
        "horizon_hours": 24, "lead_hours": range(1, 25), "predicted_demand_mw": 50_000.0,
    })
    validate_forecast_issue_frame(curve)
    with pytest.raises(ValueError, match="expected 24 leads"):
        validate_forecast_issue_frame(curve.iloc[:-1])
    curve.loc[0, "predicted_demand_mw"] = 5_000.0
    with pytest.raises(ValueError, match="physical bounds"):
        validate_forecast_issue_frame(curve)


def test_weather_feature_schema_and_issue_ids_are_stable():
    assert "weighted_temperature_f" in features_for_horizon(24)
    with pytest.raises(ValueError, match="Unsupported horizon"):
        features_for_horizon(48)
    utc_id = generate_forecast_issue_id("ERCO", 24, "2026-10-19T05:00:00Z", "2026-10-19T06:00:00Z")
    offset_id = generate_forecast_issue_id("ERCO", 24, "2026-10-19T08:00:00+03:00", "2026-10-19T09:00:00+03:00")
    assert utc_id == offset_id
    assert utc_id != generate_forecast_issue_id(
        "ERCO", 24, "2026-10-19T05:00:00Z", "2026-10-19T06:00:00Z", "different-run"
    )


def test_model_promotion_requires_repeatable_skill_and_peak_guard():
    windows = [
        {"candidate_mae": 1_800, "baseline_mae": 2_000,
         "candidate_peak_mae": 1_000, "baseline_peak_mae": 1_000}
        for _ in range(4)
    ]
    assert evaluate_promotion_gate(windows)["accepted"]
    for window in windows:
        window["candidate_peak_mae"] = 1_100
    assert not evaluate_promotion_gate(windows)["accepted"]


def test_final_model_holdout_stays_after_selection_windows():
    origins = pd.date_range("2022-01-01", periods=1_000, freq="D", tz="UTC")
    splits = generate_chronological_splits(pd.Series(origins), n_test_windows=3)
    holdout = splits[-1]
    assert holdout["kind"] == "final_holdout"
    assert all(split["test_end"] <= holdout["test_start"] for split in splits[:-1])


def test_zone_aliases_map_to_all_eight_ercot_zones():
    with open("configs/ercot_config.json", encoding="utf-8") as file:
        config = json.load(file)
    mapping = source_to_zone_code_map(config["eia"]["subregions"])
    assert {canonical_zone_code(code, mapping) for code in mapping} == {
        "NCENT", "COAST", "SCENT", "FWEST", "SOUTH", "EAST", "WEST", "NORTH",
    }


def test_weather_request_budget_stops_before_exceeding_limit():
    budget = ApiRequestBudget(2)
    assert [budget.consume(), budget.consume()] == [1, 2]
    with pytest.raises(ApiRequestBudgetExceeded):
        budget.consume()
