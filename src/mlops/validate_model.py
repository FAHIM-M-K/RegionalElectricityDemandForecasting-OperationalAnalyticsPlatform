# Databricks notebook source
"""Promote the H24 candidate only when it beats the matched daily baseline.

Candidate selection uses rolling chronological validation metrics.  The final
holdout is retained for reporting and does not affect model promotion.
"""

import numpy as np
import mlflow

from src.mlops.backtest_multihorizon import PROMOTION_POLICY_VERSION, evaluate_promotion_gate
from src.shared.forecast_contract import (
    FEATURE_SCHEMA_VERSION, LEGACY_FEATURE_SCHEMA_VERSION, HORIZONS,
)

EXPERIMENT_NAME = "/Shared/ERCOT_MultiHorizon_Forecast_Platform"
client = mlflow.tracking.MlflowClient()


def _compatible_production_runs(experiment_id: str, horizon: int):
    """Return current-schema production runs, newest first."""
    production = client.search_runs(
        [experiment_id],
        filter_string=f"tags.pipeline_role = 'PRODUCTION' AND tags.target_horizon = 'H{horizon}'",
        order_by=["start_time DESC"],
        max_results=10,
    )
    supported_schemas = {FEATURE_SCHEMA_VERSION, LEGACY_FEATURE_SCHEMA_VERSION}
    return [run for run in production
            if run.data.params.get("feature_schema_version") in supported_schemas]


def _report_retained_production(experiment_id: str, horizon: int) -> bool:
    """Make candidate rejection distinct from loss of the serving model."""
    compatible = _compatible_production_runs(experiment_id, horizon)
    if not compatible:
        print(f"H{horizon}: no current-schema production model is available.")
        return False
    incumbent = compatible[0]
    params, metrics = incumbent.data.params, incumbent.data.metrics
    policy = params.get("promotion_policy_version")
    policy_label = (f"policy {policy}" if policy == PROMOTION_POLICY_VERSION
                    else f"legacy/unrecorded policy {policy!r}")
    baseline = "daily"
    selection = metrics.get("selection_mae")
    naive = metrics.get("selection_baseline_mae")
    selection_text = f"{selection:,.1f}" if selection is not None and np.isfinite(selection) else "unavailable"
    naive_text = f"{naive:,.1f}" if naive is not None and np.isfinite(naive) else "unavailable"
    print(
        f"H{horizon}: retaining production run {incumbent.info.run_id} "
        f"({policy_label}; selection MAE {selection_text} MW "
        f"vs {naive_text} MW {baseline} naive)."
    )
    return True


def validate_and_promote_horizon(horizon: int, candidate_run_id: str | None = None) -> bool:
    if horizon not in HORIZONS:
        raise ValueError(f"Unsupported horizon: {horizon}")
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment is missing: {EXPERIMENT_NAME}")
    if candidate_run_id is None:
        try:
            candidate_run_id = dbutils.jobs.taskValues.get(taskKey="retrain_model", key=f"candidate_run_id_h{horizon}", default=None)
        except NameError:
            pass
        except ValueError as exc:
            # A standalone validation run has no upstream task named
            # ``retrain_model``. In that case, continue to the guarded MLflow
            # candidate search below; propagate unrelated task-value failures.
            if "Task key does not exist in run:" not in str(exc):
                raise
    if candidate_run_id is None:
        candidates = client.search_runs([experiment.experiment_id],
            filter_string=f"tags.pipeline_role = 'CHAMPION_CANDIDATE' AND tags.target_horizon = 'H{horizon}'",
            order_by=["start_time DESC"], max_results=1)
        candidate_run_id = candidates[0].info.run_id if candidates else None
    if candidate_run_id is None:
        raise RuntimeError(f"No H{horizon} candidate run was produced.")

    candidate = client.get_run(candidate_run_id)
    candidate_tags = candidate.data.tags
    candidate_role = candidate_tags.get("pipeline_role")
    if candidate_role != "CHAMPION_CANDIDATE":
        raise ValueError(
            f"Run {candidate_run_id} has role {candidate_role!r}; "
            "only a CHAMPION_CANDIDATE run can be validated for promotion."
        )
    expected_horizon = f"H{horizon}"
    candidate_horizon = candidate_tags.get("target_horizon")
    candidate_horizon_param = candidate.data.params.get("horizon_hours")
    if candidate_horizon != expected_horizon or str(candidate_horizon_param) != str(horizon):
        client.set_tag(candidate_run_id, "pipeline_role", "REJECTED_CANDIDATE")
        raise ValueError(
            f"Run {candidate_run_id} is not an H{horizon} candidate "
            f"(tag={candidate_horizon!r}, parameter={candidate_horizon_param!r}); "
            "candidate horizon must match the validation task."
        )
    candidate_schema = candidate.data.params.get("feature_schema_version")
    if candidate_schema != FEATURE_SCHEMA_VERSION:
        client.set_tag(candidate_run_id, "pipeline_role", "REJECTED_CANDIDATE")
        raise RuntimeError(
            f"H{horizon} candidate uses feature schema {candidate_schema!r}; "
            f"expected {FEATURE_SCHEMA_VERSION!r}. Re-run retraining from the current code."
        )
    candidate_policy = candidate.data.params.get("promotion_policy_version")
    if candidate_policy != PROMOTION_POLICY_VERSION:
        client.set_tag(candidate_run_id, "pipeline_role", "REJECTED_CANDIDATE")
        raise RuntimeError(
            f"H{horizon} candidate was trained under promotion policy {candidate_policy!r}; "
            f"expected {PROMOTION_POLICY_VERSION!r}. Re-run retraining from the current code."
        )
    # Keep the validated contract discoverable in MLflow's tag filters as well
    # as in its parameter history, including for candidates logged by older code.
    client.set_tag(candidate_run_id, "feature_schema_version", candidate_schema)
    client.set_tag(candidate_run_id, "promotion_policy_version", candidate_policy)
    metrics = candidate.data.metrics
    score, baseline, peak_score = (metrics.get(key) for key in
        ("selection_mae", "selection_baseline_mae", "selection_peak_mae"))
    if any(value is None or not np.isfinite(value) for value in (score, baseline, peak_score)):
        client.set_tag(candidate_run_id, "pipeline_role", "REJECTED_CANDIDATE")
        raise ValueError(f"H{horizon} candidate has incomplete rolling-selection metrics.")
    window_count = int(candidate.data.params.get("selection_windows", 0))
    window_metrics = []
    for window_index in range(1, window_count + 1):
        window_metrics.append({
            "candidate_mae": metrics.get(f"selection_window_{window_index}_model_mae", np.nan),
            "baseline_mae": metrics.get(f"selection_window_{window_index}_baseline_mae", np.nan),
            "candidate_peak_mae": metrics.get(f"selection_window_{window_index}_model_peak_mae", np.nan),
            "baseline_peak_mae": metrics.get(f"selection_window_{window_index}_baseline_peak_mae", np.nan),
        })
    decision = evaluate_promotion_gate(window_metrics)
    weather_control_decision = None
    if candidate_schema == FEATURE_SCHEMA_VERSION:
        weather_control_metrics = []
        for window_index in range(1, window_count + 1):
            weather_control_metrics.append({
                "candidate_mae": metrics.get(f"selection_window_{window_index}_model_mae", np.nan),
                "baseline_mae": metrics.get(f"selection_window_{window_index}_no_weather_control_mae", np.nan),
                "candidate_peak_mae": metrics.get(f"selection_window_{window_index}_model_peak_mae", np.nan),
                "baseline_peak_mae": metrics.get(f"selection_window_{window_index}_no_weather_control_peak_mae", np.nan),
            })
        weather_control_decision = evaluate_promotion_gate(weather_control_metrics)
        if not weather_control_decision["accepted"]:
            client.set_tag(candidate_run_id, "pipeline_role", "REJECTED_CANDIDATE")
            print(
                f"H{horizon} weather candidate {candidate_run_id} rejected against its matched "
                f"no-weather control: {weather_control_decision['reason']}. "
                f"Weather MAE {weather_control_decision['candidate_mae']:,.1f} vs "
                f"no-weather MAE {weather_control_decision['baseline_mae']:,.1f} MW; "
                f"window win rate {weather_control_decision['win_rate']:.0%}."
            )
            _report_retained_production(experiment.experiment_id, horizon)
            return False
    if not decision["accepted"]:
        client.set_tag(candidate_run_id, "pipeline_role", "REJECTED_CANDIDATE")
        print(
            f"H{horizon} candidate {candidate_run_id} rejected: {decision['reason']}. "
            f"Rolling MAE {score:,.1f} vs {baseline:,.1f} MW; "
            f"peak MAE {decision['candidate_peak_mae']:,.1f} vs "
            f"{decision['baseline_peak_mae']:,.1f} MW; "
            f"window win rate {decision['win_rate']:.0%}."
        )
        _report_retained_production(experiment.experiment_id, horizon)
        return False

    incumbents = client.search_runs([experiment.experiment_id],
        filter_string=f"tags.pipeline_role = 'PRODUCTION' AND tags.target_horizon = 'H{horizon}'",
        order_by=["start_time DESC"], max_results=10)
    client.set_tag(candidate_run_id, "pipeline_role", "PRODUCTION")
    client.set_tag(candidate_run_id, "target_horizon", f"H{horizon}")
    for incumbent in incumbents:
        client.set_tag(incumbent.info.run_id, "pipeline_role", "ARCHIVED")
    print(
        f"H{horizon} candidate {candidate_run_id} promoted: rolling MAE {score:,.1f} vs {baseline:,.1f} MW; "
        f"peak MAE {decision['candidate_peak_mae']:,.1f} vs "
        f"{decision['baseline_peak_mae']:,.1f} MW; "
        f"window win rate {decision['win_rate']:.0%}"
        + (f"; matched no-weather win rate {weather_control_decision['win_rate']:.0%}."
           if weather_control_decision else ".")
    )
    return True


if __name__ == "__main__":
    decision = validate_and_promote_horizon(24)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    has_production = bool(_compatible_production_runs(experiment.experiment_id, 24))
    status = "PROMOTED" if decision else ("RETAINED EXISTING PRODUCTION" if has_production else "NO COMPATIBLE PRODUCTION MODEL")
    print(f"Validation outcome H24: {status}.")
    if not has_production:
        print("Validation completed without a compatible H24 production model; hourly scoring will remain blocked.")
