# Databricks notebook source
"""Promote H24/H168 candidates only when they beat the matching naive baseline.

Candidate selection uses rolling chronological validation metrics.  The final
holdout is retained for reporting and does not affect model promotion.
"""

import numpy as np
import mlflow

from src.mlops.backtest_multihorizon import PROMOTION_POLICY_VERSION, evaluate_promotion_gate
from src.shared.forecast_contract import FEATURE_SCHEMA_VERSION, HORIZONS

EXPERIMENT_NAME = "/Shared/ERCOT_MultiHorizon_Forecast_Platform"
client = mlflow.tracking.MlflowClient()


def validate_and_promote_horizon(horizon: int, candidate_run_id: str | None = None) -> bool:
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment is missing: {EXPERIMENT_NAME}")
    if horizon not in HORIZONS:
        raise ValueError(f"Unsupported horizon: {horizon}")
    if candidate_run_id is None:
        try:
            candidate_run_id = dbutils.jobs.taskValues.get(taskKey="retrain_model", key=f"candidate_run_id_h{horizon}", default=None)
        except NameError:
            pass
    if candidate_run_id is None:
        candidates = client.search_runs([experiment.experiment_id],
            filter_string=f"tags.pipeline_role = 'CHAMPION_CANDIDATE' AND tags.target_horizon = 'H{horizon}'",
            order_by=["start_time DESC"], max_results=1)
        candidate_run_id = candidates[0].info.run_id if candidates else None
    if candidate_run_id is None:
        raise RuntimeError(f"No H{horizon} candidate run was produced.")

    candidate = client.get_run(candidate_run_id)
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
    metrics = candidate.data.metrics
    score, baseline, peak_score = (metrics.get(key) for key in
        ("selection_mae", "selection_baseline_mae", "selection_peak_mae"))
    if any(value is None or not np.isfinite(value) for value in (score, baseline, peak_score)):
        client.set_tag(candidate_run_id, "pipeline_role", "REJECTED_CANDIDATE")
        raise ValueError(f"H{horizon} candidate has incomplete rolling-selection metrics.")
    baseline_name = "daily" if horizon == 24 else "weekly"
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
    if not decision["accepted"]:
        client.set_tag(candidate_run_id, "pipeline_role", "REJECTED_CANDIDATE")
        print(
            f"H{horizon} rejected: {decision['reason']}. "
            f"Rolling MAE {score:,.1f} vs {baseline:,.1f} MW; "
            f"peak MAE {decision['candidate_peak_mae']:,.1f} vs "
            f"{decision['baseline_peak_mae']:,.1f} MW; "
            f"window win rate {decision['win_rate']:.0%}."
        )
        return False

    incumbents = client.search_runs([experiment.experiment_id],
        filter_string=f"tags.pipeline_role = 'PRODUCTION' AND tags.target_horizon = 'H{horizon}'",
        order_by=["start_time DESC"], max_results=10)
    client.set_tag(candidate_run_id, "pipeline_role", "PRODUCTION")
    client.set_tag(candidate_run_id, "target_horizon", f"H{horizon}")
    for incumbent in incumbents:
        client.set_tag(incumbent.info.run_id, "pipeline_role", "ARCHIVED")
    print(
        f"H{horizon} promoted: rolling MAE {score:,.1f} vs {baseline:,.1f} MW; "
        f"peak MAE {decision['candidate_peak_mae']:,.1f} vs "
        f"{decision['baseline_peak_mae']:,.1f} MW; "
        f"window win rate {decision['win_rate']:.0%}."
    )
    return True


if __name__ == "__main__":
    decisions = {}
    for horizon in HORIZONS:
        decisions[horizon] = validate_and_promote_horizon(horizon)

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    missing_production = []
    for horizon in HORIZONS:
        production = client.search_runs(
            [experiment.experiment_id],
            filter_string=f"tags.pipeline_role = 'PRODUCTION' AND tags.target_horizon = 'H{horizon}'",
            order_by=["start_time DESC"], max_results=10,
        )
        compatible = any(
            run.data.params.get("feature_schema_version") == FEATURE_SCHEMA_VERSION
            for run in production
        )
        if not compatible:
            missing_production.append(f"H{horizon}")
    for horizon in HORIZONS:
        status = "PROMOTED" if horizon not in missing_production and decisions[horizon] else (
            "RETAINED EXISTING PRODUCTION" if horizon not in missing_production else "NO COMPATIBLE PRODUCTION MODEL"
        )
        print(f"Validation outcome H{horizon}: {status}.")
    if missing_production:
        print(
            "Validation completed: " + ", ".join(missing_production) + " has no compatible promoted model. "
            "Daily scoring requires H24; H168 will use the labeled weekly seasonal-naive fallback until promoted."
        )
