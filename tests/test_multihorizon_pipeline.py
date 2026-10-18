import json

import numpy as np
import pandas as pd

from src.gold.build_forecast_examples import generate_forecast_examples_df
from src.mlops.backtest_multihorizon import (
    compute_naive_baselines,
    evaluate_promotion_gate,
    generate_chronological_splits,
)
from src.gold.score_forecast_hourly import generate_independent_forecasts
from src.shared.forecast_contract import validate_example_record
from src.shared.ercot_zones import canonical_zone_code, source_to_zone_code_map


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


def test_model_seasonal_feature_matches_horizon_baseline():
    examples = generate_forecast_examples_df(_hourly(), step_hours=24)
    h24 = examples[examples.horizon_hours == 24]
    h168 = examples[examples.horizon_hours == 168]
    assert np.allclose(h24.seasonal_naive_mw, h24.daily_naive_mw)
    assert np.allclose(h168.seasonal_naive_mw, h168.weekly_naive_mw)


def test_eia_source_zone_aliases_resolve_to_project_codes():
    config = json.loads(open("configs/ercot_config.json", encoding="utf-8").read())
    mapping = source_to_zone_code_map(config["eia"]["subregions"])
    assert {canonical_zone_code(code, mapping) for code in mapping} == {
        "NCENT", "COAST", "SCENT", "FWEST", "SOUTH", "EAST", "WEST", "NORTH"
    }


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


def _promotion_windows(candidate_maes, baseline_maes=None, candidate_peak=1_000.0, baseline_peak=1_000.0):
    baseline_maes = baseline_maes or [2_000.0] * len(candidate_maes)
    return [
        {"candidate_mae": candidate, "baseline_mae": baseline,
         "candidate_peak_mae": candidate_peak, "baseline_peak_mae": baseline_peak}
        for candidate, baseline in zip(candidate_maes, baseline_maes)
    ]


def test_promotion_gate_requires_repeatable_improvement_and_peak_guard():
    accepted = evaluate_promotion_gate(_promotion_windows([1_800.0, 1_900.0, 1_950.0, 1_700.0]))
    assert accepted["accepted"]
    assert accepted["win_rate"] == 1.0

    weak_repeatability = evaluate_promotion_gate(
        _promotion_windows([1_500.0, 1_500.0, 2_100.0, 2_100.0])
    )
    assert not weak_repeatability["accepted"]
    assert "wins only 50%" in weak_repeatability["reason"]

    peak_regression = evaluate_promotion_gate(
        _promotion_windows([1_800.0] * 4, candidate_peak=1_100.0)
    )
    assert not peak_regression["accepted"]
    assert "peak-hour MAE regresses" in peak_regression["reason"]


def test_promotion_gate_rejects_insufficient_or_incomplete_windows():
    insufficient = evaluate_promotion_gate(_promotion_windows([1_800.0] * 3))
    assert not insufficient["accepted"]
    assert "at least 4" in insufficient["reason"]

    incomplete_windows = _promotion_windows([1_800.0] * 4)
    incomplete_windows[2]["candidate_peak_mae"] = np.nan
    incomplete = evaluate_promotion_gate(incomplete_windows)
    assert not incomplete["accepted"]
    assert incomplete["reason"] == "selection window metrics are incomplete"


def test_scoring_issues_both_products_and_rejects_missing_model():
    class ConstantModel:
        def predict(self, features):
            # A zero residual should reproduce the seasonal naive forecast.
            return np.zeros(len(features))
    history = _hourly(hours=500)
    calendar_dates = set(pd.date_range("2024-01-01", periods=40, freq="D").date)
    output = generate_independent_forecasts(history, {24: ConstantModel(), 168: ConstantModel()},
                                           {24: "run-h24", 168: "run-h168"},
                                           holiday_dates=set(), calendar_dates=calendar_dates,
                                           high_demand_threshold_mw=50_000.0)
    assert set(output.horizon_hours) == {24, 168}
    assert len(output) == 24 + 168
    assert output.high_demand_threshold_mw.eq(50_000.0).all()
    assert output.is_predicted_high_demand.all()
    h24_lead1 = output[(output.horizon_hours == 24) & (output.lead_hours == 1)].iloc[0]
    h168_lead168 = output[(output.horizon_hours == 168) & (output.lead_hours == 168)].iloc[0]
    assert h24_lead1.predicted_demand_mw == history.iloc[-24].actual_demand_mw
    assert h168_lead168.predicted_demand_mw == history.iloc[-1].actual_demand_mw
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
