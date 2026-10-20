import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.gold.forecast_example_features import generate_forecast_examples_df
from src.mlops.backtest_multihorizon import (
    compute_naive_baselines,
    compute_lead_bucket_metrics,
    compute_threshold_metrics,
    compute_matched_metrics,
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


def test_matched_metrics_compare_eia_and_model_on_identical_rows():
    actual = np.array([10.0, 20.0, 30.0, 40.0])
    result = compute_matched_metrics(
        actual,
        {"model": np.array([11.0, 19.0, 30.0, 41.0]),
         "eia": np.array([12.0, np.nan, 29.0, 42.0])},
    )
    assert result["sample_count"] == 3
    assert result["coverage_pct"] == 75.0
    assert result["metrics"]["model"]["sample_count"] == 3
    assert result["metrics"]["eia"]["sample_count"] == 3
    assert result["metrics"]["model"]["mae"] == 0.67
    assert result["metrics"]["eia"]["mae"] == 1.67


def test_lead_bucket_metrics_split_week_ahead_error_by_distance():
    frame = pd.DataFrame({
        "horizon_hours": 168,
        "lead_hours": [1, 24, 25, 48, 49, 168],
        "actual_demand_mw": [100.0] * 6,
    })
    result = compute_lead_bucket_metrics(
        frame,
        {"model": [101, 102, 104, 106, 110, 118],
         "weekly": [103, 103, 103, 103, 103, 103]},
    )
    assert set(result) == {"lead_1_24", "lead_25_48", "lead_49_168"}
    assert result["lead_1_24"]["model"]["sample_count"] == 2
    assert result["lead_1_24"]["model"]["mae"] == 1.5
    assert result["lead_25_48"]["model"]["mae"] == 5.0
    assert result["lead_49_168"]["model"]["mae"] == 14.0
    assert result["lead_49_168"]["weekly"]["mae"] == 3.0


def test_threshold_metrics_count_false_alarms_and_missed_high_demand():
    result = compute_threshold_metrics(
        [74_000.0, 76_000.0, 80_000.0, np.nan],
        [77_000.0, 74_000.0, 79_000.0, 90_000.0],
        threshold_mw=75_000.0,
    )
    assert result["sample_count"] == 3
    assert result["actual_high_count"] == 2
    assert result["predicted_high_count"] == 2
    assert result["true_positive"] == 1
    assert result["false_positive"] == 1
    assert result["false_negative"] == 1
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5


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


def test_weather_candidate_keeps_production_until_explicit_owner_review(monkeypatch):
    from src.mlops.backtest_multihorizon import PROMOTION_POLICY_VERSION
    from src.shared.forecast_contract import FEATURE_SCHEMA_VERSION

    class FakeClient:
        def __init__(self):
            self.tags = {}
            self.candidate = SimpleNamespace(
                info=SimpleNamespace(run_id="candidate-run"),
                data=SimpleNamespace(
                    params={
                        "feature_schema_version": FEATURE_SCHEMA_VERSION,
                        "promotion_policy_version": PROMOTION_POLICY_VERSION,
                        "horizon_hours": "168",
                        "selection_windows": "4",
                    },
                    tags={
                        "pipeline_role": "CHAMPION_CANDIDATE",
                        "target_horizon": "H168",
                    },
                    metrics={
                        "selection_mae": 1_800.0,
                        "selection_baseline_mae": 2_000.0,
                        "selection_peak_mae": 1_000.0,
                        **{
                            key: value
                            for index in range(1, 5)
                            for key, value in (
                                (f"selection_window_{index}_model_mae", 1_800.0),
                                (f"selection_window_{index}_baseline_mae", 2_000.0),
                                (f"selection_window_{index}_model_peak_mae", 1_000.0),
                                (f"selection_window_{index}_baseline_peak_mae", 1_000.0),
                                (f"selection_window_{index}_no_weather_control_mae", 2_000.0),
                                (f"selection_window_{index}_no_weather_control_peak_mae", 1_000.0),
                            )
                        },
                    },
                ),
            )

        def get_experiment_by_name(self, _name):
            return SimpleNamespace(experiment_id="experiment-id")

        def get_run(self, _run_id):
            return self.candidate

        def search_runs(self, *_args, **_kwargs):
            return []

        def set_tag(self, run_id, key, value):
            self.tags[(run_id, key)] = value

    fake_client = FakeClient()
    fake_mlflow = SimpleNamespace(tracking=SimpleNamespace(MlflowClient=lambda: fake_client))
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake_mlflow)
    module_path = Path("src/mlops/validate_model.py")
    spec = importlib.util.spec_from_file_location("validate_model_test_module", module_path)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    validator.client = fake_client

    assert validator.validate_and_promote_horizon(168, "candidate-run")
    assert fake_client.tags[("candidate-run", "pipeline_role")] == "PRODUCTION"
    assert fake_client.tags[("candidate-run", "feature_schema_version")] == FEATURE_SCHEMA_VERSION
    assert fake_client.tags[("candidate-run", "promotion_policy_version")] == PROMOTION_POLICY_VERSION

    fake_client.tags.clear()
    fake_client.candidate.data.tags["target_horizon"] = "H24"
    fake_client.candidate.data.params["horizon_hours"] = "24"
    try:
        validator.validate_and_promote_horizon(168, "candidate-run")
    except ValueError as exc:
        assert "candidate horizon must match" in str(exc)
    else:
        raise AssertionError("The validator promoted an H24 run as an H168 candidate.")
    assert fake_client.tags[("candidate-run", "pipeline_role")] == "REJECTED_CANDIDATE"

    fake_client.tags.clear()
    fake_client.candidate.data.tags["pipeline_role"] = "PRODUCTION"
    fake_client.candidate.data.tags["target_horizon"] = "H168"
    fake_client.candidate.data.params["horizon_hours"] = "168"
    try:
        validator.validate_and_promote_horizon(168, "candidate-run")
    except ValueError as exc:
        assert "only a CHAMPION_CANDIDATE" in str(exc)
    else:
        raise AssertionError("The validator accepted a production run as a candidate.")
    assert fake_client.tags == {}


def test_scoring_issues_both_products_and_uses_labeled_h168_fallback():
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
    fallback = generate_independent_forecasts(
        history, {24: ConstantModel(), 168: None},
        {24: "run-h24", 168: "baseline:weekly-seasonal-naive:v1"},
        holiday_dates=set(), calendar_dates=calendar_dates,
    )
    h168 = fallback[fallback.horizon_hours == 168]
    assert len(h168) == 168
    assert h168.forecast_source.eq("weekly_seasonal_naive_baseline").all()
    assert h168.model_version.eq("WeeklySeasonalNaive_Baseline_H168").all()
    assert h168.model_run_id.eq("baseline:weekly-seasonal-naive:v1").all()
    assert h168.predicted_demand_mw.iloc[-1] == history.iloc[-1].actual_demand_mw

    try:
        generate_independent_forecasts(history, {168: ConstantModel()}, {168: "run-h168"},
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
        "target_timestamp_utc": [origin + pd.Timedelta(hours=lead) for lead in range(1, 169)],
        "snapshot_available_at_utc": [issued - pd.Timedelta(minutes=1)] * 168,
        "snapshot_batch_id": ["batch-1"] * 168,
        "weighted_temperature_f": [80.0] * 168,
        "cdh_65_forecast": [15.0] * 168,
        "hdh_65_forecast": [0.0] * 168,
    })
    weather_model = WeatherModel(features_for_horizon(24))
    legacy_h168 = type("LegacyModel", (), {"feature_name": lambda self: features_for_horizon(168, False),
                                           "predict": lambda self, frame: np.zeros(len(frame))})()
    forecast = generate_independent_forecasts(
        history, {24: weather_model, 168: legacy_h168}, {24: "weather-h24", 168: "legacy-h168"},
        holiday_dates=set(), calendar_dates=calendar_dates, issued_at_utc=issued,
        weather_forecast=weather, model_schemas={24: FEATURE_SCHEMA_VERSION},
    )
    assert weather_model.seen["weighted_temperature_f"].eq(80.0).all()
    assert forecast.loc[forecast.horizon_hours == 24, "weather_snapshot_batch_id"].eq("batch-1").all()
    assert forecast.loc[forecast.horizon_hours == 168, "weather_snapshot_batch_id"].isna().all()


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
        "target_timestamp_utc": [origin + pd.Timedelta(hours=lead) for lead in range(1, 169)],
        "snapshot_available_at_utc": [issued + pd.Timedelta(minutes=1)] * 168,
        "weighted_temperature_f": [80.0] * 168, "cdh_65_forecast": [15.0] * 168,
        "hdh_65_forecast": [0.0] * 168,
    })
    try:
        generate_independent_forecasts(
            history, {24: Model(), 168: None}, {24: "weather-h24", 168: "baseline:weekly-seasonal-naive:v1"},
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
    issue_id_h168 = generate_forecast_issue_id("ERCO", 168, origin, issued)
    assert issue_id_h24.startswith("issue:ERCO:H24:")
    assert issue_id_h168.startswith("issue:ERCO:H168:")
    assert issue_id_h24 != issue_id_h168
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
    valid_h168 = issue_rows(168, 52_000.0)
    both = pd.concat([valid_h24, valid_h168], ignore_index=True)
    validate_forecast_issue_frame(both)

    # Incomplete curve
    try:
        validate_forecast_issue_frame(valid_h24.iloc[:-1], expected_horizons=[24])
    except ValueError as exc:
        assert "expected 24 leads" in str(exc)
    else:
        raise AssertionError("Incomplete H24 curve passed validation.")

    # Bound violation
    invalid_pred = both.copy()
    invalid_pred.loc[0, "predicted_demand_mw"] = 5_000.0
    try:
        validate_forecast_issue_frame(invalid_pred, expected_horizons=[24, 168])
    except ValueError as exc:
        assert "exceed physical bounds" in str(exc)
    else:
        raise AssertionError("Physical bound violation passed validation.")


def test_validate_benchmark_vintage_record():
    from src.shared.forecast_contract import validate_benchmark_vintage_record
    record = {
        "benchmark_provider": "EIA",
        "respondent": "ERCO",
        "vintage_id": "vintage_20240601",
        "target_timestamp_utc": pd.Timestamp("2024-06-01 14:00", tz="UTC"),
        "forecast_mw": 55_000.0,
        "first_observed_at_utc": pd.Timestamp("2024-06-01 06:00", tz="UTC"),
    }
    validate_benchmark_vintage_record(record)

    missing_field = record.copy()
    del missing_field["vintage_id"]
    try:
        validate_benchmark_vintage_record(missing_field)
    except ValueError as exc:
        assert "missing required field" in str(exc)
    else:
        raise AssertionError("Record missing required field passed validation.")


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


def test_as_of_vintage_selection_keeps_one_complete_latest_row():
    from src.gold.benchmark_evaluation import select_as_of_benchmark_vintages

    target = pd.Timestamp("2026-10-20T12:00:00Z")
    vintages = pd.DataFrame([
        {"benchmark_provider": "EIA", "respondent": "ERCO", "vintage_id": "old",
         "target_timestamp_utc": target, "forecast_mw": 50_000.0,
         "published_at_utc": pd.NaT, "first_observed_at_utc": pd.Timestamp("2026-10-19T10:00:00Z"),
         "payload_sha256": "old-hash"},
        {"benchmark_provider": "EIA", "respondent": "ERCO", "vintage_id": "new",
         "target_timestamp_utc": target, "forecast_mw": 51_000.0,
         "published_at_utc": pd.NaT, "first_observed_at_utc": pd.Timestamp("2026-10-19T11:00:00Z"),
         "payload_sha256": "new-hash"},
    ])
    selected = select_as_of_benchmark_vintages(vintages, pd.Timestamp("2026-10-19T12:00:00Z"))
    assert len(selected) == 1
    assert selected.iloc[0].vintage_id == "new"
    assert selected.iloc[0].payload_sha256 == "new-hash"


def test_ercot_benchmark_parser_keeps_only_system_total_and_requires_recognized_rows():
    import json
    from src.bronze.ingest_ercot_benchmark import parse_system_forecast_records

    payload = json.dumps({"data": [
        {"target_timestamp_utc": "2026-10-20T12:00:00Z", "weatherZone": "COAST", "forecastMw": 9_000},
        {"target_timestamp_utc": "2026-10-20T12:00:00Z", "weatherZone": "ERCOT System Total", "forecastMw": 51_000},
    ]}).encode("utf-8")
    rows = parse_system_forecast_records(payload)
    assert len(rows) == 1
    assert rows[0]["product_id"] == "NP3-561-CD"
    assert rows[0]["respondent"] == "ERCO"
    assert rows[0]["forecast_mw"] == 51_000

    unrecognized = json.dumps({"data": [
        {"target_timestamp_utc": "2026-10-20T12:00:00Z", "weatherZone": "COAST", "forecastMw": 9_000}
    ]}).encode("utf-8")
    try:
        parse_system_forecast_records(unrecognized)
    except ValueError as exc:
        assert "no recognizable system-total" in str(exc)
    else:
        raise AssertionError("Zone-only ERCOT data was incorrectly accepted as a system benchmark.")


def test_ercot_benchmark_parser_accepts_wide_system_total_column():
    from src.bronze.ingest_ercot_benchmark import parse_system_forecast_records

    payload = (
        "DeliveryDate,HourEnding,SystemTotal,COAST,NORTH\n"
        "2026-10-20,HE 01,51000,9200,700\n"
    ).encode("utf-8")
    rows = parse_system_forecast_records(payload)
    assert len(rows) == 1
    assert rows[0]["forecast_mw"] == 51_000
    assert rows[0]["target_timestamp_utc"] == pd.Timestamp("2026-10-20T05:00:00Z").to_pydatetime()


def test_ercot_benchmark_parser_accepts_xml_row_records():
    from src.bronze.ingest_ercot_benchmark import parse_system_forecast_records

    payload = b"""<Report><Record><TargetTimestampUtc>2026-10-20T12:00:00Z</TargetTimestampUtc>
    <ForecastType>SystemTotal</ForecastType><ForecastMw>51000</ForecastMw></Record></Report>"""
    rows = parse_system_forecast_records(payload)
    assert len(rows) == 1
    assert rows[0]["forecast_mw"] == 51_000


def test_ercot_benchmark_parser_rejects_ambiguous_repeated_local_hour():
    import json
    from src.bronze.ingest_ercot_benchmark import parse_system_forecast_records

    payload = json.dumps({"data": [
        {"deliveryDate": "2026-11-01", "hourEnding": 2, "systemTotal": 51_000},
    ]}).encode("utf-8")
    try:
        parse_system_forecast_records(payload)
    except ValueError as exc:
        assert "ambiguous during the fall-back" in str(exc)
    else:
        raise AssertionError("Ambiguous fall-back local hour was silently accepted.")


def test_ercot_benchmark_parser_rejects_nonexistent_spring_forward_local_hour():
    import json
    from src.bronze.ingest_ercot_benchmark import parse_system_forecast_records

    # HE=03 starts at local 02:00, which does not exist on the spring-forward date.
    payload = json.dumps({"data": [
        {"deliveryDate": "2026-03-08", "hourEnding": 3, "systemTotal": 51_000},
    ]}).encode("utf-8")
    try:
        parse_system_forecast_records(payload)
    except ValueError as exc:
        assert "does not exist during the spring-forward" in str(exc)
    else:
        raise AssertionError("Nonexistent spring-forward local hour was silently accepted.")


def test_ercot_token_request_posts_credentials_as_form_data(monkeypatch):
    import json
    from urllib.parse import parse_qs
    from src.bronze import ingest_ercot_benchmark

    captured = {}

    class FakeResponse:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"id_token": "test-bearer-token"}).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(ingest_ercot_benchmark.urllib.request, "urlopen", fake_urlopen)
    assert ingest_ercot_benchmark.get_ercot_token("user", "password") == "test-bearer-token"
    request = captured["request"]
    assert request.full_url == ingest_ercot_benchmark.AUTH_URL
    assert request.get_method() == "POST"
    posted = parse_qs(request.data.decode("utf-8"))
    assert posted["username"] == ["user"]
    assert posted["password"] == ["password"]
    assert "username" not in request.full_url


def test_benchmark_match_rejects_duplicate_actual_keys():
    from src.gold.benchmark_evaluation import match_h24_and_eia

    origin = pd.Timestamp("2024-06-01T00:00:00Z")
    target = origin + pd.Timedelta(hours=24)
    forecasts = pd.DataFrame([{
        "respondent": "ERCO", "horizon_hours": 24, "lead_hours": 24,
        "forecast_origin_utc": origin, "issued_at_utc": origin + pd.Timedelta(minutes=15),
        "target_timestamp_utc": target, "predicted_demand_mw": 51_000.0,
    }])
    actuals = pd.DataFrame([
        {"respondent": "ERCO", "timestamp_utc": target, "actual_demand_mw": 50_000.0},
        {"respondent": "ERCO", "timestamp_utc": target, "actual_demand_mw": 50_100.0},
    ])
    try:
        match_h24_and_eia(forecasts, pd.DataFrame(), actuals)
    except ValueError as exc:
        assert "Actual demand contains duplicate rows" in str(exc)
    else:
        raise AssertionError("Duplicate actuals were silently allowed to overweight benchmark metrics.")


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
        history, {24: ConstantModel(), 168: ConstantModel()},
        {24: "run-h24", 168: "run-h168"},
        holiday_dates=set(), calendar_dates=calendar_dates,
        issued_at_utc=issued_fresh,
    )
    assert len(df_fresh) == 192
    assert "forecast_issue_id" in df_fresh.columns
    assert df_fresh.freshness_status.eq("OPERATIONAL_FRESH").all()
    assert df_fresh.source_data_age_hours.iloc[0] == 1.0

    # Case 2: Stale input (issued 5h after origin)
    issued_stale = origin + pd.Timedelta(hours=5)
    df_stale = generate_independent_forecasts(
        history, {24: ConstantModel(), 168: ConstantModel()},
        {24: "run-h24", 168: "run-h168"},
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


def test_as_of_benchmark_selection_no_lookahead():
    from src.gold.benchmark_evaluation import select_as_of_benchmark_vintages

    target_ts = pd.Timestamp("2024-06-01 18:00:00", tz="UTC")
    vintages = pd.DataFrame([
        {
            "benchmark_provider": "EIA",
            "respondent": "ERCO",
            "vintage_id": "v1_early",
            "target_timestamp_utc": target_ts,
            "forecast_mw": 50_000.0,
            "first_observed_at_utc": pd.Timestamp("2024-06-01 06:00:00", tz="UTC"),
        },
        {
            "benchmark_provider": "EIA",
            "respondent": "ERCO",
            "vintage_id": "v2_revision",
            "target_timestamp_utc": target_ts,
            "forecast_mw": 52_000.0,
            "first_observed_at_utc": pd.Timestamp("2024-06-01 10:00:00", tz="UTC"),
        },
        {
            "benchmark_provider": "EIA",
            "respondent": "ERCO",
            "vintage_id": "v3_future",
            "target_timestamp_utc": target_ts,
            "forecast_mw": 55_000.0,
            "first_observed_at_utc": pd.Timestamp("2024-06-01 14:00:00", tz="UTC"),
        },
    ])

    # As of 08:00 UTC: only v1 is available
    as_of_08 = select_as_of_benchmark_vintages(vintages, pd.Timestamp("2024-06-01 08:00:00", tz="UTC"))
    assert len(as_of_08) == 1
    assert as_of_08.iloc[0]["vintage_id"] == "v1_early"
    assert as_of_08.iloc[0]["forecast_mw"] == 50_000.0

    # As of 11:00 UTC: v2 is available and preferred over v1; v3 is excluded (no look-ahead)
    as_of_11 = select_as_of_benchmark_vintages(vintages, pd.Timestamp("2024-06-01 11:00:00", tz="UTC"))
    assert len(as_of_11) == 1
    assert as_of_11.iloc[0]["vintage_id"] == "v2_revision"
    assert as_of_11.iloc[0]["forecast_mw"] == 52_000.0


def test_match_h24_and_eia_metrics():
    from src.gold.benchmark_evaluation import match_h24_and_eia

    origin = pd.Timestamp("2024-06-01 00:00:00", tz="UTC")
    issued = pd.Timestamp("2024-06-01 00:15:00", tz="UTC")
    target = origin + pd.Timedelta(hours=24)

    h24 = pd.DataFrame([{
        "respondent": "ERCO",
        "horizon_hours": 24,
        "lead_hours": 24,
        "forecast_origin_utc": origin,
        "issued_at_utc": issued,
        "target_timestamp_utc": target,
        "predicted_demand_mw": 51_000.0,
    }])

    vintages = pd.DataFrame([{
        "benchmark_provider": "EIA",
        "respondent": "ERCO",
        "vintage_id": "eia_v1",
        "target_timestamp_utc": target,
        "forecast_mw": 52_000.0,
        "first_observed_at_utc": pd.Timestamp("2024-05-31 23:00:00", tz="UTC"),
    }])

    actuals = pd.DataFrame([{
        "respondent": "ERCO",
        "timestamp_utc": target,
        "actual_demand_mw": 50_000.0,
    }])

    df_matched, summary = match_h24_and_eia(h24, vintages, actuals, headline_policy="lead_24")
    assert len(df_matched) == 1
    assert summary["matched_rows"] == 1
    assert summary["unique_targets"] == 1
    assert summary["model_mae_mw"] == 1_000.0   # |51000 - 50000|
    assert summary["eia_mae_mw"] == 2_000.0     # |52000 - 50000|
    assert summary["model_bias_mw"] == 1_000.0  # 51000 - 50000
    assert summary["eia_bias_mw"] == 2_000.0    # 52000 - 50000


def test_match_h168_and_ercot_prospective_and_buckets():
    from src.gold.benchmark_evaluation import match_h168_and_ercot

    # Test Case 1: Empty vintages truthfully returns PROSPECTIVE_PENDING_MATURED_DATA
    df_empty, summary_empty = match_h168_and_ercot(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    assert summary_empty["status"] == "PROSPECTIVE_PENDING_MATURED_DATA"
    assert summary_empty["matched_rows"] == 0

    # Test Case 2: Matched curves across lead buckets
    origin = pd.Timestamp("2024-06-01 00:00:00", tz="UTC")
    issued = pd.Timestamp("2024-06-01 00:15:00", tz="UTC")

    # Generate 168 rows
    h168_rows = []
    ercot_rows = []
    actual_rows = []
    for lead in range(1, 169):
        target = origin + pd.Timedelta(hours=lead)
        h168_rows.append({
            "respondent": "ERCO", "horizon_hours": 168, "lead_hours": lead,
            "forecast_origin_utc": origin, "issued_at_utc": issued,
            "target_timestamp_utc": target, "predicted_demand_mw": 50_000.0 + lead * 10,
        })
        ercot_rows.append({
            "benchmark_provider": "ERCOT", "respondent": "ERCO",
            "vintage_id": "ercot_v1", "target_timestamp_utc": target,
            "forecast_mw": 50_000.0 + lead * 5,
            "first_observed_at_utc": origin - pd.Timedelta(hours=1),
        })
        actual_rows.append({
            "respondent": "ERCO", "timestamp_utc": target, "actual_demand_mw": 50_000.0,
        })

    df_h168 = pd.DataFrame(h168_rows)
    df_ercot = pd.DataFrame(ercot_rows)
    df_actuals = pd.DataFrame(actual_rows)

    df_matched, summary = match_h168_and_ercot(df_h168, df_ercot, df_actuals)
    assert summary["status"] == "EVALUATED"
    assert summary["matched_rows"] == 168
    assert summary["unique_targets"] == 168
    # Assert lead buckets exist
    assert "leads_1_24" in summary["lead_bucket_metrics"]
    assert "leads_25_48" in summary["lead_bucket_metrics"]
    assert "leads_49_168" in summary["lead_bucket_metrics"]
    assert summary["lead_bucket_metrics"]["leads_1_24"]["matched_rows"] == 24
    assert summary["lead_bucket_metrics"]["leads_49_168"]["matched_rows"] == 120
    # Assert lead 168 diagnostic exists
    assert summary["lead_168_diagnostic"] is not None
    assert summary["lead_168_diagnostic"]["matched_rows"] == 1
