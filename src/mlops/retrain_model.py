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

from src.mlops.backtest_multihorizon import compute_metrics, compute_naive_baselines, generate_chronological_splits
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
    return data


def _fit(train: pd.DataFrame, validation: pd.DataFrame, features: list[str]):
    model = lgb.train(MODEL_PARAMS, lgb.Dataset(train[features], label=train["actual_demand_mw"]),
                      num_boost_round=1200,
                      valid_sets=[lgb.Dataset(validation[features], label=validation["actual_demand_mw"])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
    return model


def _score(model, frame: pd.DataFrame, include_eia: bool = False) -> Dict[str, Dict[str, float]]:
    prediction = model.predict(frame[model.feature_name()])
    peak = frame["hour_local"].between(14, 18).to_numpy()
    daily, weekly = compute_naive_baselines(frame)
    eia = frame["eia_day_ahead_forecast_mw"] if "eia_day_ahead_forecast_mw" in frame else pd.Series(np.nan, index=frame.index)
    return {"model": compute_metrics(frame["actual_demand_mw"], prediction, peak),
            "daily": compute_metrics(frame["actual_demand_mw"], daily, peak),
            "weekly": compute_metrics(frame["actual_demand_mw"], weekly, peak),
            "eia": compute_metrics(frame["actual_demand_mw"], eia, peak) if include_eia else
            {"mae": np.nan, "rmse": np.nan, "bias": np.nan, "smape": np.nan, "peak_mae": np.nan, "sample_count": 0}}


def _mean_metric(results: list[Dict[str, Dict[str, float]]], model_name: str, metric: str) -> float:
    values = [result[model_name][metric] for result in results if np.isfinite(result[model_name][metric])]
    return float(np.mean(values)) if values else np.nan


def train_single_horizon_model(examples: pd.DataFrame, horizon: int, n_windows: int = 4) -> dict:
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
        selection_results.append(_score(_fit(fit_train, validation, features), test, include_eia=(horizon == 24)))
    if not selection_results:
        raise ValueError(f"H{horizon} has no usable rolling selection windows.")

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
    booster = lgb.train(MODEL_PARAMS, lgb.Dataset(pre_holdout[features], label=pre_holdout["actual_demand_mw"]),
                        num_boost_round=selected_model.best_iteration)
    final_result = _score(booster, final_holdout, include_eia=(horizon == 24))
    selection_mae = _mean_metric(selection_results, "model", "mae")
    selection_baseline = _mean_metric(selection_results, "daily" if horizon == 24 else "weekly", "mae")

    result = {"horizon": horizon, "booster": booster, "features": features, "selection_mae": selection_mae,
              "selection_baseline_mae": selection_baseline, "selection_peak_mae": _mean_metric(selection_results, "model", "peak_mae"),
              "selection_windows": len(selection_results), "final": final_result, "run_id": None}
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name=f"Direct_LightGBM_H{horizon}") as run:
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
            if np.isfinite(final_result["eia"]["mae"]):
                logged_metrics["final_holdout_eia_mae"] = final_result["eia"]["mae"]
            mlflow.log_metrics(logged_metrics)
            mlflow.log_dict({"horizon_hours": horizon, "importance_type": "gain",
                             "features": [{"feature_name": name, "gain_importance": float(value)}
                                          for name, value in zip(booster.feature_name(), booster.feature_importance("gain"))]},
                            "feature_importance.json")
            mlflow.set_tags({"pipeline_role": "CHAMPION_CANDIDATE", "target_horizon": f"H{horizon}",
                             "model_type": "direct_supervised_lightgbm", "weather_assumption": "no_weather"})
            mlflow.lightgbm.log_model(booster, artifact_path=f"model_direct_h{horizon}")
    return result


if __name__ == "__main__" and "spark" in globals():
    examples = spark.table(f"{CATALOG}.gold.forecast_examples").toPandas()
    results = {horizon: train_single_horizon_model(examples, horizon) for horizon in HORIZONS}
    for horizon, result in results.items():
        print(f"H{horizon}: rolling-selection MAE {result['selection_mae']:.1f}; final holdout MAE {result['final']['model']['mae']:.1f}")
        if result["run_id"]:
            dbutils.jobs.taskValues.set(key=f"candidate_run_id_h{horizon}", value=result["run_id"])
