import numpy as np
import pandas as pd

from src.gold.build_forecast_examples import generate_forecast_examples_df
from src.mlops.backtest_multihorizon import compute_naive_baselines, generate_chronological_splits
from src.gold.score_forecast_hourly import generate_independent_forecasts
from src.shared.forecast_contract import validate_example_record


def _hourly(hours=800, gap=None):
    timestamps = pd.date_range("2024-01-01", periods=hours, freq="h", tz="UTC")
    frame = pd.DataFrame({"respondent": "ERCO", "timestamp_utc": timestamps,
                          "actual_demand_mw": 50_000 + np.arange(hours, dtype=float), "is_holiday": False})
    return frame[frame.timestamp_utc != gap].copy() if gap is not None else frame


def test_examples_are_partitioned_by_respondent_and_target_aligned():
    source = pd.concat([_hourly().assign(respondent="ERCO"), _hourly().assign(respondent="TEST")])
    examples = generate_forecast_examples_df(source, step_hours=24)
    assert set(examples.respondent) == {"ERCO", "TEST"}
    assert (examples.target_timestamp_utc - examples.forecast_origin_utc == pd.to_timedelta(examples.lead_hours, unit="h")).all()
    assert not examples.duplicated(["respondent", "forecast_origin_utc", "target_timestamp_utc", "horizon_hours", "feature_set"]).any()


def test_gap_invalidates_rolling_history_instead_of_using_row_offsets():
    source = _hourly(gap=pd.Timestamp("2024-01-17 00:00", tz="UTC"))
    examples = generate_forecast_examples_df(source, horizons=[24], step_hours=1)
    # An origin one day after the gap needs that missing hour in its trailing 24h window.
    assert not (examples.forecast_origin_utc == pd.Timestamp("2024-01-18 00:00", tz="UTC")).any()


def test_invalid_load_inside_rolling_window_invalidates_training_example():
    source = _hourly(hours=800)
    origin = source.timestamp_utc.iloc[500]
    source.loc[source.timestamp_utc == origin - pd.Timedelta(hours=7), "actual_demand_mw"] = 1_000.0
    examples = generate_forecast_examples_df(source, horizons=[24], step_hours=1)
    assert not (examples.forecast_origin_utc == origin).any()


def test_naive_baselines_are_target_aligned_not_origin_persistence():
    source = _hourly()
    examples = generate_forecast_examples_df(source, horizons=[24], step_hours=24)
    sample = examples[examples.lead_hours == 12].iloc[[0]]
    daily, weekly = compute_naive_baselines(sample)
    assert daily[0] == sample.daily_naive_mw.iloc[0]
    assert weekly[0] == sample.weekly_naive_mw.iloc[0]
    assert daily[0] != sample.demand_origin.iloc[0]


def test_contract_rejects_misaligned_target():
    record = {"forecast_origin_utc": pd.Timestamp("2024-01-01", tz="UTC"),
              "target_timestamp_utc": pd.Timestamp("2024-01-02", tz="UTC"), "horizon_hours": 24,
              "lead_hours": 23, "actual_demand_mw": 50_000}
    try:
        validate_example_record(record)
    except ValueError as exc:
        assert "Target timestamp" in str(exc)
    else:
        raise AssertionError("A misaligned forecast example passed validation.")


def test_chronological_splits_reserve_final_holdout():
    origins = pd.date_range("2022-01-01", periods=1000, freq="D", tz="UTC")
    splits = generate_chronological_splits(pd.Series(origins), n_test_windows=3)
    assert splits[-1]["kind"] == "final_holdout"
    assert all(split["test_end"] <= splits[-1]["test_start"] for split in splits[:-1])


def test_scoring_issues_both_products_and_rejects_missing_model():
    class ConstantModel:
        def predict(self, features):
            return np.repeat(55_000.0, len(features))
    history = _hourly(hours=500)
    calendar_dates = set(pd.date_range("2024-01-01", periods=40, freq="D").date)
    output = generate_independent_forecasts(history, {24: ConstantModel(), 168: ConstantModel()},
                                           {24: "run-h24", 168: "run-h168"},
                                           holiday_dates=set(), calendar_dates=calendar_dates,
                                           high_demand_threshold_mw=54_000.0)
    assert set(output.horizon_hours) == {24, 168}
    assert len(output) == 24 + 168
    assert output.high_demand_threshold_mw.eq(54_000.0).all()
    assert output.is_predicted_high_demand.all()
    invalid_history = history.copy()
    invalid_history.loc[invalid_history.index[-8], "actual_demand_mw"] = 1_000.0
    try:
        generate_independent_forecasts(invalid_history, {24: ConstantModel(), 168: ConstantModel()},
                                       {24: "run-h24", 168: "run-h168"},
                                       holiday_dates=set(), calendar_dates=calendar_dates)
    except ValueError as exc:
        assert "invalid load value" in str(exc)
    else:
        raise AssertionError("Scoring accepted a physically invalid value in the rolling history.")
    try:
        generate_independent_forecasts(history, {24: ConstantModel()}, {24: "run-h24"},
                                       holiday_dates=set(), calendar_dates=calendar_dates)
    except ValueError as exc:
        assert "No active production H168" in str(exc)
    else:
        raise AssertionError("Scoring silently accepted a missing production model.")
