import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.gold.forecast_example_features import generate_forecast_examples_df
from src.mlops.backtest_multihorizon import (
    compute_daily_naive_baseline,
    compute_lead_bucket_metrics,
    compute_peak_event_metrics,
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
    examples = generate_forecast_examples_df(source, step_hours=1)
    # An origin one day after the gap needs that missing hour in its trailing 24h window.
    assert not (examples.forecast_origin_utc == pd.Timestamp("2024-01-18 00:00", tz="UTC")).any()


def test_invalid_load_inside_rolling_window_invalidates_training_example():
    source = _hourly(hours=800)
    origin = source.timestamp_utc.iloc[500]
    source.loc[source.timestamp_utc == origin - pd.Timedelta(hours=7), "actual_demand_mw"] = 1_000.0
    examples = generate_forecast_examples_df(source, step_hours=1)
    assert not (examples.forecast_origin_utc == origin).any()


def test_naive_baselines_are_target_aligned_not_origin_persistence():
    source = _hourly()
    examples = generate_forecast_examples_df(source, step_hours=24)
    sample = examples[examples.lead_hours == 12].iloc[[0]]
    daily = compute_daily_naive_baseline(sample)
    assert daily[0] == sample.daily_naive_mw.iloc[0]
    assert daily[0] != sample.demand_origin.iloc[0]


def test_model_seasonal_feature_uses_h24_daily_baseline():
    examples = generate_forecast_examples_df(_hourly(), step_hours=24)
    assert set(examples.horizon_hours) == {24}
    assert np.allclose(examples.seasonal_naive_mw, examples.daily_naive_mw)


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


def test_lead_bucket_metrics_cover_the_active_h24_curve():
    frame = pd.DataFrame({
        "horizon_hours": 24,
        "lead_hours": [1, 12, 24],
        "actual_demand_mw": [100.0] * 3,
    })
    result = compute_lead_bucket_metrics(
        frame,
        {"model": [101, 102, 104], "daily": [103, 103, 103]},
    )
    assert set(result) == {"lead_1_24"}
    assert result["lead_1_24"]["model"]["sample_count"] == 3
    assert result["lead_1_24"]["model"]["mae"] == 2.33
    assert result["lead_1_24"]["daily"]["mae"] == 3.0


def test_peak_event_metrics_score_peak_height_and_timing_per_complete_origin():
    origin = pd.Timestamp("2024-01-01", tz="UTC")
    leads = np.arange(1, 25)
    actual = np.full(24, 50_000.0)
    actual[4] = 80_000.0
    predicted = np.full(24, 50_000.0)
    predicted[6] = 82_000.0
    frame = pd.DataFrame({
        "forecast_origin_utc": origin,
        "target_timestamp_utc": [origin + pd.Timedelta(hours=int(lead)) for lead in leads],
        "horizon_hours": 24,
        "lead_hours": leads,
        "actual_demand_mw": actual,
    }, index=np.arange(100, 124))
    result = compute_peak_event_metrics(frame, {"model": predicted})["model"]
    assert result["complete_origins"] == 1
    assert result["peak_magnitude_mae_mw"] == 2_000.0
    assert result["peak_timing_mae_hours"] == 2.0


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


def test_nonproduction_horizon_candidate_is_rejected(monkeypatch):
    fake_mlflow = SimpleNamespace(tracking=SimpleNamespace(MlflowClient=lambda: None))
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake_mlflow)
    module_path = Path("src/mlops/validate_model.py")
    spec = importlib.util.spec_from_file_location("validate_model_test_module", module_path)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)

    try:
        validator.validate_and_promote_horizon(48, "unsupported-run")
    except ValueError as exc:
        assert "Unsupported horizon: 48" in str(exc)
    else:
        raise AssertionError("The validator accepted a nonproduction horizon.")


def test_scoring_issues_only_the_h24_product():
    class ConstantModel:
        def predict(self, features):
            # A zero residual should reproduce the seasonal naive forecast.
            return np.zeros(len(features))
    history = _hourly(hours=500)
    calendar_dates = set(pd.date_range("2024-01-01", periods=40, freq="D").date)
    output = generate_independent_forecasts(history, {24: ConstantModel()},
                                           {24: "run-h24"},
                                           holiday_dates=set(), calendar_dates=calendar_dates)
    assert set(output.horizon_hours) == {24}
    assert len(output) == 24
    h24_lead1 = output[(output.horizon_hours == 24) & (output.lead_hours == 1)].iloc[0]
    assert h24_lead1.predicted_demand_mw == history.iloc[-24].actual_demand_mw
    invalid_history = history.copy()
    invalid_history.loc[invalid_history.index[-8], "actual_demand_mw"] = 1_000.0
    try:
        generate_independent_forecasts(invalid_history, {24: ConstantModel()},
                                       {24: "run-h24"},
                                       holiday_dates=set(), calendar_dates=calendar_dates)
    except ValueError as exc:
        assert "invalid load value" in str(exc)
    else:
        raise AssertionError("Scoring accepted a physically invalid value in the rolling history.")
    try:
        generate_independent_forecasts(history, {}, {},
                                       holiday_dates=set(), calendar_dates=calendar_dates)
    except ValueError as exc:
        assert "No active production H24" in str(exc)
    else:
        raise AssertionError("Scoring accepted a missing required H24 production model.")


def test_weather_schema_scoring_uses_issue_available_weather_features():
    from src.shared.forecast_contract import FEATURE_SCHEMA_VERSION, features_for_horizon

    class WeatherModel:
        def __init__(self, columns):
            self.columns = columns
            self.seen = None
        def feature_name(self):
            return self.columns
        def predict(self, frame):
            self.seen = frame.copy()
            return np.zeros(len(frame))

    history = _hourly(hours=500)
    origin = history.timestamp_utc.max()
    issued = origin + pd.Timedelta(minutes=65)
    calendar_dates = set(pd.date_range("2024-01-01", periods=40, freq="D").date)
    weather = pd.DataFrame({
        "target_timestamp_utc": [origin + pd.Timedelta(hours=lead) for lead in range(1, 25)],
        "snapshot_available_at_utc": [issued - pd.Timedelta(minutes=1)] * 24,
        "snapshot_batch_id": ["batch-1"] * 24,
        "weighted_temperature_f": [80.0] * 24,
        "cdh_65_forecast": [15.0] * 24,
        "hdh_65_forecast": [0.0] * 24,
    })
    weather_model = WeatherModel(features_for_horizon(24))
    forecast = generate_independent_forecasts(
        history, {24: weather_model}, {24: "weather-h24"},
        holiday_dates=set(), calendar_dates=calendar_dates, issued_at_utc=issued,
        weather_forecast=weather, model_schemas={24: FEATURE_SCHEMA_VERSION},
    )
    assert weather_model.seen["weighted_temperature_f"].eq(80.0).all()
    assert forecast.weather_snapshot_batch_id.eq("batch-1").all()


def test_weather_schema_scoring_rejects_weather_published_after_issue_time():
    from src.shared.forecast_contract import FEATURE_SCHEMA_VERSION, features_for_horizon

    class Model:
        def feature_name(self):
            return features_for_horizon(24)
        def predict(self, frame):
            return np.zeros(len(frame))

    history = _hourly(hours=500)
    origin = history.timestamp_utc.max()
    issued = origin + pd.Timedelta(minutes=65)
    calendar_dates = set(pd.date_range("2024-01-01", periods=40, freq="D").date)
    weather = pd.DataFrame({
        "target_timestamp_utc": [origin + pd.Timedelta(hours=lead) for lead in range(1, 25)],
        "snapshot_available_at_utc": [issued + pd.Timedelta(minutes=1)] * 24,
        "weighted_temperature_f": [80.0] * 24, "cdh_65_forecast": [15.0] * 24,
        "hdh_65_forecast": [0.0] * 24,
    })
    try:
        generate_independent_forecasts(
            history, {24: Model()}, {24: "weather-h24"},
            holiday_dates=set(), calendar_dates=calendar_dates, issued_at_utc=issued,
            weather_forecast=weather, model_schemas={24: FEATURE_SCHEMA_VERSION},
        )
    except ValueError as exc:
        assert "published after the issue time" in str(exc)
    else:
        raise AssertionError("Scoring accepted weather published after the simulated issue time.")


def test_forecast_issue_id_is_deterministic_and_unique():
    from src.shared.forecast_contract import generate_forecast_issue_id
    origin = pd.Timestamp("2024-06-01 12:00:00", tz="UTC")
    issued = pd.Timestamp("2024-06-01 12:15:00", tz="UTC")
    issue_id_h24 = generate_forecast_issue_id("ERCO", 24, origin, issued)
    assert issue_id_h24.startswith("issue:ERCO:H24:")
    # Deterministic:
    assert issue_id_h24 == generate_forecast_issue_id("ERCO", 24, origin, issued)


def test_validate_forecast_issue_frame_enforces_complete_curves_and_bounds():
    from src.shared.forecast_contract import generate_forecast_issue_id, validate_forecast_issue_frame
    origin = pd.Timestamp("2026-10-19 05:00", tz="UTC")
    issued = pd.Timestamp("2026-10-19 06:00", tz="UTC")

    def issue_rows(horizon, prediction):
        leads = list(range(1, horizon + 1))
        return pd.DataFrame({
            "respondent": ["ERCO"] * horizon,
            "forecast_issue_id": [generate_forecast_issue_id("ERCO", horizon, origin, issued)] * horizon,
            "forecast_origin_utc": [origin] * horizon,
            "target_timestamp_utc": [origin + pd.Timedelta(hours=lead) for lead in leads],
            "issued_at_utc": [issued] * horizon,
            "horizon_hours": [horizon] * horizon,
            "lead_hours": leads,
            "predicted_demand_mw": [prediction] * horizon,
        })

    valid_h24 = issue_rows(24, 50_000.0)
    validate_forecast_issue_frame(valid_h24)
    try:
        validate_forecast_issue_frame(issue_rows(48, 52_000.0))
    except ValueError as exc:
        assert "Unsupported" in str(exc) or "Invalid" in str(exc)
    else:
        raise AssertionError("Unsupported forecast rows passed serving validation.")

    # Incomplete curve
    try:
        validate_forecast_issue_frame(valid_h24.iloc[:-1], expected_horizons=[24])
    except ValueError as exc:
        assert "expected 24 leads" in str(exc)
    else:
        raise AssertionError("Incomplete H24 curve passed validation.")

    # Bound violation
    invalid_pred = valid_h24.copy()
    invalid_pred.loc[0, "predicted_demand_mw"] = 5_000.0
    try:
        validate_forecast_issue_frame(invalid_pred, expected_horizons=[24])
    except ValueError as exc:
        assert "exceed physical bounds" in str(exc)
    else:
        raise AssertionError("Physical bound violation passed validation.")


def test_forecast_issue_id_normalizes_equivalent_utc_offsets():
    from src.shared.forecast_contract import generate_forecast_issue_id

    utc_id = generate_forecast_issue_id(
        "ERCO", 24, "2026-10-19T05:00:00Z", "2026-10-19T06:00:00Z"
    )
    offset_id = generate_forecast_issue_id(
        "ERCO", 24, "2026-10-19T08:00:00+03:00", "2026-10-19T09:00:00+03:00"
    )
    assert utc_id == offset_id
    assert generate_forecast_issue_id(
        "ERCO", 24, "2026-10-19T05:00:00Z", "2026-10-19T06:00:00Z", "run-a"
    ) != generate_forecast_issue_id(
        "ERCO", 24, "2026-10-19T05:00:00Z", "2026-10-19T06:00:00Z", "run-b"
    )


def test_scoring_attaches_issue_id_and_freshness_status():
    class ConstantModel:
        def predict(self, features):
            return np.zeros(len(features))
    history = _hourly(hours=500)
    calendar_dates = set(pd.date_range("2024-01-01", periods=40, freq="D").date)
    origin = history.timestamp_utc.max()

    # Case 1: Fresh input (issued 1h after origin)
    issued_fresh = origin + pd.Timedelta(hours=1)
    df_fresh = generate_independent_forecasts(
        history, {24: ConstantModel()},
        {24: "run-h24"},
        holiday_dates=set(), calendar_dates=calendar_dates,
        issued_at_utc=issued_fresh,
    )
    assert len(df_fresh) == 24
    assert "forecast_issue_id" in df_fresh.columns
    assert df_fresh.freshness_status.eq("OPERATIONAL_FRESH").all()
    assert df_fresh.source_data_age_hours.iloc[0] == 1.0

    # Case 2: Stale input (issued 5h after origin)
    issued_stale = origin + pd.Timedelta(hours=5)
    df_stale = generate_independent_forecasts(
        history, {24: ConstantModel()},
        {24: "run-h24"},
        holiday_dates=set(), calendar_dates=calendar_dates,
        issued_at_utc=issued_stale,
    )
    assert df_stale.freshness_status.eq("STALE_INPUT").all()
    assert df_stale.source_data_age_hours.iloc[0] == 5.0


def test_dst_spring_forward_and_fall_back_continuous_utc():
    # Test March 2024 Spring Forward and November 2024 Fall Back
    for start_str in ["2024-03-09 00:00:00", "2024-11-02 00:00:00"]:
        origin = pd.Timestamp(start_str, tz="UTC")
        leads = list(range(1, 169))
        targets_utc = [origin + pd.Timedelta(hours=lead) for lead in leads]
        # In UTC, differences are exactly 1 hour continuous
        diffs = pd.Series(targets_utc).diff().dropna()
        assert (diffs == pd.Timedelta(hours=1)).all()
        # Local timestamps are valid America/Chicago conversions
        locals_ct = [t.tz_convert("America/Chicago") for t in targets_utc]
        assert len(set(targets_utc)) == 168
        assert len(locals_ct) == 168
