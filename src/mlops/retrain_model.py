# Databricks notebook source
# COMMAND ----------
# MAGIC %pip install lightgbm==4.7.0
# COMMAND ----------
"""Train direct, point-in-time LightGBM models for H24 and H168.

Model selection uses rolling chronological windows.  The final 30 days are
logged as an audit holdout and are never used to choose a candidate or promote
it.  EIA is a comparison benchmark only; it is never an input feature.
"""

import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
import lightgbm as lgb

_root = os.getcwd()
for _ in range(5):
    if os.path.isdir(os.path.join(_root, "configs")):
        sys.path.insert(0, _root)
        break
    _root = os.path.dirname(_root)

from src.mlops.backtest_multihorizon import (
    MAX_PEAK_MAE_REGRESSION, MIN_SELECTION_WINDOWS, MIN_WINDOW_WIN_RATE,
    PROMOTION_POLICY_VERSION, compute_lead_bucket_metrics, compute_matched_metrics, compute_metrics, compute_naive_baselines,
    compute_peak_event_metrics, compute_threshold_metrics,
    generate_chronological_splits,
)
from src.shared.forecast_contract import FEATURE_SCHEMA_VERSION, HORIZONS, features_for_horizon

try:
    import mlflow
    import mlflow.lightgbm
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

CATALOG = "main"
EXPERIMENT_NAME = "/Shared/ERCOT_MultiHorizon_Forecast_Platform"
MODEL_PARAMS = {"objective": "regression_l1", "metric": "l1", "learning_rate": 0.05,
                "num_leaves": 31, "min_child_samples": 40, "feature_fraction": 0.85,
                "bagging_fraction": 0.85, "bagging_freq": 1, "seed": 42, "verbose": -1}


def _clean_examples(examples: pd.DataFrame, horizon: int) -> pd.DataFrame:
    features = features_for_horizon(horizon)
    data = examples.loc[(examples["horizon_hours"] == horizon) & (examples["is_valid_training_example"] == True)].copy()
    data["forecast_origin_utc"] = pd.to_datetime(data["forecast_origin_utc"], utc=True)
    data["target_timestamp_utc"] = pd.to_datetime(data["target_timestamp_utc"], utc=True)
    data = data.dropna(subset=features + ["actual_demand_mw"]).sort_values("forecast_origin_utc")
    if data.empty:
        raise ValueError(f"No valid H{horizon} examples available for training.")
    data["residual_target_mw"] = data["actual_demand_mw"] - data["seasonal_naive_mw"]
    return data


def _fit(train: pd.DataFrame, validation: pd.DataFrame, features: list[str]):
    model = lgb.train(MODEL_PARAMS, lgb.Dataset(train[features], label=train["residual_target_mw"]),
                      num_boost_round=1200,
                      valid_sets=[lgb.Dataset(validation[features], label=validation["residual_target_mw"])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
    return model


def _score(model, frame: pd.DataFrame, horizon: int, include_eia: bool = False,
           high_demand_threshold_mw: float = 75_000.0) -> Dict[str, Dict[str, float]]:
    # The booster predicts an adjustment to the forecast-origin-available
    # seasonal naive value. Score the reconstructed demand forecast.
    prediction = frame["seasonal_naive_mw"].to_numpy(dtype=float) + model.predict(frame[model.feature_name()])
    peak = frame["hour_local"].between(14, 18).to_numpy()
    daily, weekly = compute_naive_baselines(frame)
    eia = frame["eia_day_ahead_forecast_mw"] if "eia_day_ahead_forecast_mw" in frame else pd.Series(np.nan, index=frame.index)
    model_metrics = compute_metrics(frame["actual_demand_mw"], prediction, peak)
    daily_metrics = compute_metrics(frame["actual_demand_mw"], daily, peak)
    weekly_metrics = compute_metrics(frame["actual_demand_mw"], weekly, peak)
    selection_baseline = daily_metrics if horizon == 24 else weekly_metrics
    if model_metrics["sample_count"] != selection_baseline["sample_count"]:
        raise ValueError(f"H{horizon} model and seasonal baseline were scored on different row counts.")
    matched_eia = compute_matched_metrics(
        frame["actual_demand_mw"],
        {"model": prediction, "daily_naive": daily, "eia": eia},
        peak_mask=peak,
    ) if include_eia else {"sample_count": 0, "coverage_pct": 0.0, "metrics": {}}
    lead_buckets = compute_lead_bucket_metrics(
        frame, {"model": prediction, "daily": daily, "weekly": weekly}, peak
    )
    leads = pd.to_numeric(frame["lead_hours"], errors="coerce").to_numpy(float)
    bucket_ranges = {"lead_1_24": (1, 24)} if horizon == 24 else {
        "lead_1_24": (1, 24), "lead_25_48": (25, 48), "lead_49_168": (49, 168)
    }
    threshold_metrics = {}
    for bucket, (lower, upper) in bucket_ranges.items():
        in_bucket = np.isfinite(leads) & (leads >= lower) & (leads <= upper)
        threshold_metrics[bucket] = {
            name: compute_threshold_metrics(frame["actual_demand_mw"].to_numpy()[in_bucket], values[in_bucket],
                                            high_demand_threshold_mw)
            for name, values in {"model": prediction, "daily": daily, "weekly": weekly}.items()
        }
    peak_event_metrics = compute_peak_event_metrics(
        frame, {"model": prediction, "daily": daily, "weekly": weekly}
    )
    return {"model": model_metrics,
            "daily": daily_metrics,
            "weekly": weekly_metrics,
            "lead_buckets": lead_buckets,
            "threshold_metrics": threshold_metrics,
            "peak_event_metrics": peak_event_metrics,
            "eia_matched": matched_eia,
            "eia": compute_metrics(frame["actual_demand_mw"], eia, peak) if include_eia else
            {"mae": np.nan, "rmse": np.nan, "bias": np.nan, "smape": np.nan, "peak_mae": np.nan, "sample_count": 0}}


def _mean_metric(results: list[Dict[str, Dict[str, float]]], model_name: str, metric: str) -> float:
    values = [result[model_name][metric] for result in results if np.isfinite(result[model_name][metric])]
    return float(np.mean(values)) if values else np.nan


def train_single_horizon_model(examples: pd.DataFrame, horizon: int, n_windows: int = 4,
                               high_demand_threshold_mw: float = 75_000.0) -> dict:
    """Fit and score one direct horizon model without random splitting."""
    features, data = features_for_horizon(horizon), _clean_examples(examples, horizon)
    splits = generate_chronological_splits(data["forecast_origin_utc"], n_test_windows=n_windows)
    selection_results = []
    for split in (item for item in splits if item["kind"] == "selection"):
        validation_start = split["test_start"] - pd.Timedelta(days=30)
        fit_train = data[(data.forecast_origin_utc < validation_start) & (data.target_timestamp_utc < validation_start)]
        validation = data[(data.forecast_origin_utc >= validation_start) &
                          (data.forecast_origin_utc < split["test_start"]) &
                          (data.target_timestamp_utc < split["test_start"])]
        test = data[(data.forecast_origin_utc >= split["test_start"]) & (data.forecast_origin_utc < split["test_end"])]
        if fit_train.empty or validation.empty or test.empty:
            continue
        window_result = _score(
            _fit(fit_train, validation, features), test, horizon, include_eia=(horizon == 24),
            high_demand_threshold_mw=high_demand_threshold_mw,
        )
        window_result["test_start_utc"] = split["test_start"].isoformat()
        window_result["test_end_utc"] = split["test_end"].isoformat()
        selection_results.append(window_result)
    if len(selection_results) < MIN_SELECTION_WINDOWS:
        raise ValueError(
            f"H{horizon} has {len(selection_results)} usable rolling selection windows; "
            f"at least {MIN_SELECTION_WINDOWS} are required for promotion evaluation."
        )

    final = next(item for item in splits if item["kind"] == "final_holdout")
    validation_start = final["test_start"] - pd.Timedelta(days=30)
    production_train = data[(data.forecast_origin_utc < validation_start) & (data.target_timestamp_utc < validation_start)]
    validation = data[(data.forecast_origin_utc >= validation_start) &
                      (data.forecast_origin_utc < final["test_start"]) &
                      (data.target_timestamp_utc < final["test_start"])]
    pre_holdout = data[(data.forecast_origin_utc < final["test_start"]) &
                       (data.target_timestamp_utc < final["test_start"])]
    final_holdout = data[(data.forecast_origin_utc >= final["test_start"]) & (data.forecast_origin_utc < final["test_end"])]
    if production_train.empty or validation.empty or final_holdout.empty:
        raise ValueError(f"H{horizon} does not have sufficient pre-holdout training or final test data.")
    selected_model = _fit(production_train, validation, features)
    booster = lgb.train(MODEL_PARAMS, lgb.Dataset(pre_holdout[features], label=pre_holdout["residual_target_mw"]),
                        num_boost_round=selected_model.best_iteration)
    final_result = _score(booster, final_holdout, horizon, include_eia=(horizon == 24),
                          high_demand_threshold_mw=high_demand_threshold_mw)
    selection_mae = _mean_metric(selection_results, "model", "mae")
    selection_baseline = _mean_metric(selection_results, "daily" if horizon == 24 else "weekly", "mae")

    result = {"horizon": horizon, "booster": booster, "features": features, "selection_mae": selection_mae,
              "selection_baseline_mae": selection_baseline, "selection_peak_mae": _mean_metric(selection_results, "model", "peak_mae"),
              "selection_windows": len(selection_results), "final": final_result, "run_id": None}
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name=f"SeasonalResidual_LightGBM_H{horizon}") as run:
            result["run_id"] = run.info.run_id
            mlflow.log_params({**MODEL_PARAMS, "horizon_hours": horizon, "feature_schema_version": FEATURE_SCHEMA_VERSION,
                               "feature_count": len(features), "selection_windows": len(selection_results),
                               "training_rows": len(pre_holdout), "training_origin_start": str(pre_holdout.forecast_origin_utc.min()),
                               "training_origin_end": str(pre_holdout.forecast_origin_utc.max())})
            logged_metrics = {"selection_mae": selection_mae, "selection_peak_mae": result["selection_peak_mae"],
                              "selection_baseline_mae": selection_baseline,
                              "final_holdout_mae": final_result["model"]["mae"],
                              "final_holdout_peak_mae": final_result["model"]["peak_mae"],
                              "final_holdout_daily_naive_mae": final_result["daily"]["mae"],
                              "final_holdout_weekly_naive_mae": final_result["weekly"]["mae"]}
            baseline_name = "daily" if horizon == 24 else "weekly"
            for label, model_key in (("model", "model"), ("baseline", baseline_name)):
                peak = final_result["peak_event_metrics"][model_key]
                logged_metrics[f"final_holdout_{label}_complete_peak_origins"] = peak["complete_origins"]
                for metric in ("peak_magnitude_mae_mw", "peak_timing_mae_hours"):
                    if np.isfinite(peak[metric]):
                        logged_metrics[f"final_holdout_{label}_{metric}"] = peak[metric]
            for bucket, forecasts in final_result["threshold_metrics"].items():
                for label, model_key in (("model", "model"), ("baseline", baseline_name)):
                    alert = forecasts[model_key]
                    for metric in ("actual_high_count", "predicted_high_count", "true_positive", "false_positive", "false_negative"):
                        logged_metrics[f"final_holdout_{bucket}_{label}_{metric}"] = alert[metric]
                    for metric in ("precision", "recall"):
                        if np.isfinite(alert[metric]):
                            logged_metrics[f"final_holdout_{bucket}_{label}_{metric}"] = alert[metric]
            for bucket, bucket_results in final_result["lead_buckets"].items():
                for label, model_key in (("model", "model"), ("baseline", baseline_name)):
                    metric = bucket_results[model_key]
                    if np.isfinite(metric["mae"]):
                        logged_metrics[f"final_holdout_{bucket}_{label}_mae"] = metric["mae"]
                    logged_metrics[f"final_holdout_{bucket}_{label}_rows"] = metric["sample_count"]
            if horizon == 24:
                matched = final_result["eia_matched"]
                logged_metrics.update({
                    "final_holdout_eia_matched_rows": matched["sample_count"],
                    "final_holdout_eia_coverage_pct": matched["coverage_pct"],
                })
                if matched["sample_count"]:
                    logged_metrics.update({
                        "final_holdout_model_mae_on_eia_hours": matched["metrics"]["model"]["mae"],
                        "final_holdout_daily_naive_mae_on_eia_hours": matched["metrics"]["daily_naive"]["mae"],
                        "final_holdout_eia_mae_on_matched_hours": matched["metrics"]["eia"]["mae"],
                    })
            logged_metrics["selection_baseline_peak_mae"] = _mean_metric(
                selection_results, baseline_name, "peak_mae"
            )
            for window_index, window_result in enumerate(selection_results, start=1):
                mlflow.log_params({
                    f"selection_window_{window_index}_start_utc": window_result["test_start_utc"],
                    f"selection_window_{window_index}_end_utc": window_result["test_end_utc"],
                })
                logged_metrics.update({
                    f"selection_window_{window_index}_model_mae": window_result["model"]["mae"],
                    f"selection_window_{window_index}_baseline_mae": window_result[baseline_name]["mae"],
                    f"selection_window_{window_index}_model_peak_mae": window_result["model"]["peak_mae"],
                    f"selection_window_{window_index}_baseline_peak_mae": window_result[baseline_name]["peak_mae"],
                })
                for label, model_key in (("model", "model"), ("baseline", baseline_name)):
                    peak = window_result["peak_event_metrics"][model_key]
                    logged_metrics[f"selection_window_{window_index}_{label}_complete_peak_origins"] = peak["complete_origins"]
                    for metric in ("peak_magnitude_mae_mw", "peak_timing_mae_hours"):
                        if np.isfinite(peak[metric]):
                            logged_metrics[f"selection_window_{window_index}_{label}_{metric}"] = peak[metric]
                for bucket, bucket_results in window_result["lead_buckets"].items():
                    for label, model_key in (("model", "model"), ("baseline", baseline_name)):
                        metric = bucket_results[model_key]
                        if np.isfinite(metric["mae"]):
                            logged_metrics[f"selection_window_{window_index}_{bucket}_{label}_mae"] = metric["mae"]
                        logged_metrics[f"selection_window_{window_index}_{bucket}_{label}_rows"] = metric["sample_count"]
                for bucket, forecasts in window_result["threshold_metrics"].items():
                    for label, model_key in (("model", "model"), ("baseline", baseline_name)):
                        alert = forecasts[model_key]
                        for metric in ("actual_high_count", "predicted_high_count", "true_positive", "false_positive", "false_negative"):
                            logged_metrics[f"selection_window_{window_index}_{bucket}_{label}_{metric}"] = alert[metric]
                        for metric in ("precision", "recall"):
                            if np.isfinite(alert[metric]):
                                logged_metrics[f"selection_window_{window_index}_{bucket}_{label}_{metric}"] = alert[metric]
                if horizon == 24:
                    matched = window_result["eia_matched"]
                    logged_metrics.update({
                        f"selection_window_{window_index}_eia_matched_rows": matched["sample_count"],
                        f"selection_window_{window_index}_eia_coverage_pct": matched["coverage_pct"],
                    })
                    if matched["sample_count"]:
                        logged_metrics.update({
                            f"selection_window_{window_index}_model_mae_on_eia_hours": matched["metrics"]["model"]["mae"],
                            f"selection_window_{window_index}_daily_naive_mae_on_eia_hours": matched["metrics"]["daily_naive"]["mae"],
                            f"selection_window_{window_index}_eia_mae_on_matched_hours": matched["metrics"]["eia"]["mae"],
                        })
            mlflow.log_metrics(logged_metrics)
            mlflow.log_params({"promotion_policy_version": PROMOTION_POLICY_VERSION,
                               "minimum_selection_windows": MIN_SELECTION_WINDOWS,
                               "minimum_window_win_rate": MIN_WINDOW_WIN_RATE,
                               "maximum_peak_mae_regression": MAX_PEAK_MAE_REGRESSION,
                               "high_demand_threshold_mw": high_demand_threshold_mw})
            mlflow.log_dict({"horizon_hours": horizon, "importance_type": "gain",
                             "features": [{"feature_name": name, "gain_importance": float(value)}
                                          for name, value in zip(booster.feature_name(), booster.feature_importance("gain"))]},
                            "feature_importance.json")
            mlflow.set_tags({"pipeline_role": "CHAMPION_CANDIDATE", "target_horizon": f"H{horizon}",
                             "model_type": "seasonal_naive_residual_lightgbm", "weather_assumption": "no_weather"})
            mlflow.lightgbm.log_model(booster, artifact_path=f"model_seasonal_residual_h{horizon}")
    return result


if __name__ == "__main__" and "spark" in globals():
    from configs.loader import load_platform_config

    examples = spark.table(f"{CATALOG}.gold.forecast_examples").toPandas()
    threshold = float(load_platform_config("ercot_config.json")["region"]["default_peak_threshold_mw"])
    results = {horizon: train_single_horizon_model(examples, horizon,
                                                   high_demand_threshold_mw=threshold) for horizon in HORIZONS}
    for horizon, result in results.items():
        print(f"H{horizon}: rolling-selection MAE {result['selection_mae']:.1f}; final holdout MAE {result['final']['model']['mae']:.1f}")
        if result["run_id"]:
            dbutils.jobs.taskValues.set(key=f"candidate_run_id_h{horizon}", value=result["run_id"])
