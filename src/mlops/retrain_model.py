# Databricks notebook source
# COMMAND ----------
# MAGIC %pip install lightgbm==4.7.0
# COMMAND ----------
"""Train the direct, point-in-time rolling H24 LightGBM model.

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
    PROMOTION_POLICY_VERSION, compute_lead_bucket_metrics, compute_matched_metrics, compute_metrics, compute_daily_naive_baseline,
    compute_peak_event_metrics,
    generate_chronological_splits,
)
from src.shared.forecast_contract import FEATURE_SCHEMA_VERSION, HORIZONS, features_for_horizon
from src.shared.forecast_contract import EXAMPLE_GRAIN_KEYS
from src.mlops.training_data_contract import (
    ROLLING_EXAMPLES_TABLE,
    WEATHER_ROLLING_EXAMPLES_TABLE,
    clean_training_examples,
    validate_examples_table,
)

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
    return clean_training_examples(examples, horizon)


def _fit(train: pd.DataFrame, validation: pd.DataFrame, features: list[str]):
    model = lgb.train(MODEL_PARAMS, lgb.Dataset(train[features], label=train["residual_target_mw"]),
                      num_boost_round=1200,
                      valid_sets=[lgb.Dataset(validation[features], label=validation["residual_target_mw"])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
    return model


def _score(model, frame: pd.DataFrame, horizon: int, include_eia: bool = False) -> Dict[str, Dict[str, float]]:
    # The booster predicts an adjustment to the forecast-origin-available
    # seasonal naive value. Score the reconstructed demand forecast.
    prediction = frame["seasonal_naive_mw"].to_numpy(dtype=float) + model.predict(frame[model.feature_name()])
    peak = frame["hour_local"].between(14, 18).to_numpy()
    daily = compute_daily_naive_baseline(frame)
    eia = frame["eia_day_ahead_forecast_mw"] if "eia_day_ahead_forecast_mw" in frame else pd.Series(np.nan, index=frame.index)
    model_metrics = compute_metrics(frame["actual_demand_mw"], prediction, peak)
    daily_metrics = compute_metrics(frame["actual_demand_mw"], daily, peak)
    selection_baseline = daily_metrics
    if model_metrics["sample_count"] != selection_baseline["sample_count"]:
        raise ValueError(f"H{horizon} model and seasonal baseline were scored on different row counts.")
    matched_eia = compute_matched_metrics(
        frame["actual_demand_mw"],
        {"model": prediction, "daily_naive": daily, "eia": eia},
        peak_mask=peak,
    ) if include_eia else {"sample_count": 0, "coverage_pct": 0.0, "metrics": {}}
    lead_buckets = compute_lead_bucket_metrics(
        frame, {"model": prediction, "daily": daily}, peak
    )
    peak_event_metrics = compute_peak_event_metrics(
        frame, {"model": prediction, "daily": daily}
    )
    return {"model": model_metrics,
            "daily": daily_metrics,
            "lead_buckets": lead_buckets,
            "peak_event_metrics": peak_event_metrics,
            "eia_matched": matched_eia,
            "eia": compute_metrics(frame["actual_demand_mw"], eia, peak) if include_eia else
            {"mae": np.nan, "rmse": np.nan, "bias": np.nan, "smape": np.nan, "peak_mae": np.nan, "sample_count": 0}}


def _mean_metric(results: list[Dict[str, Dict[str, float]]], model_name: str, metric: str) -> float:
    values = [result[model_name][metric] for result in results if np.isfinite(result[model_name][metric])]
    return float(np.mean(values)) if values else np.nan


def train_single_horizon_model(examples: pd.DataFrame, horizon: int, n_windows: int = 4,
                               examples_table: str = WEATHER_ROLLING_EXAMPLES_TABLE) -> dict:
    """Fit and score one direct horizon model without random splitting."""
    features = features_for_horizon(horizon)
    control_features = features_for_horizon(horizon, include_weather=False)
    data = _clean_examples(examples, horizon)
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
        weather_model = _fit(fit_train, validation, features)
        control_model = _fit(fit_train, validation, control_features)
        window_result = _score(weather_model, test, horizon, include_eia=True)
        control_result = _score(control_model, test, horizon, include_eia=True)
        window_result["no_weather_control"] = control_result["model"]
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
    selected_control_model = _fit(production_train, validation, control_features)
    booster = lgb.train(MODEL_PARAMS, lgb.Dataset(pre_holdout[features], label=pre_holdout["residual_target_mw"]),
                        num_boost_round=selected_model.best_iteration)
    control_booster = lgb.train(
        MODEL_PARAMS, lgb.Dataset(pre_holdout[control_features], label=pre_holdout["residual_target_mw"]),
        num_boost_round=selected_control_model.best_iteration,
    )
    final_result = _score(booster, final_holdout, horizon, include_eia=True)
    control_final = _score(control_booster, final_holdout, horizon, include_eia=True)
    final_result["no_weather_control"] = control_final["model"]
    selection_mae = _mean_metric(selection_results, "model", "mae")
    baseline_name = "daily"
    selection_baseline = _mean_metric(selection_results, baseline_name, "mae")
    selection_peak_baseline = _mean_metric(selection_results, baseline_name, "peak_mae")
    selection_window_wins = sum(
        window["model"]["mae"] < window[baseline_name]["mae"]
        for window in selection_results
    )
    selection_weather_control_wins = sum(
        window["model"]["mae"] < window["no_weather_control"]["mae"]
        for window in selection_results
    )

    result = {"horizon": horizon, "booster": booster, "features": features, "selection_mae": selection_mae,
              "selection_baseline_mae": selection_baseline,
              "selection_peak_mae": _mean_metric(selection_results, "model", "peak_mae"),
              "selection_baseline_peak_mae": selection_peak_baseline,
              "selection_window_wins": selection_window_wins,
              "selection_no_weather_control_mae": _mean_metric(selection_results, "no_weather_control", "mae"),
              "selection_no_weather_control_peak_mae": _mean_metric(selection_results, "no_weather_control", "peak_mae"),
              "selection_weather_control_wins": selection_weather_control_wins,
              "selection_windows": len(selection_results), "final": final_result, "run_id": None}
    holdout_period = {
        "final_holdout_rows": len(final_holdout),
        "final_holdout_origin_start_utc": str(final_holdout["forecast_origin_utc"].min()),
        "final_holdout_origin_end_utc": str(final_holdout["forecast_origin_utc"].max()),
        "final_holdout_target_start_utc": str(final_holdout["target_timestamp_utc"].min()),
        "final_holdout_target_end_utc": str(final_holdout["target_timestamp_utc"].max()),
        "final_holdout_test_start_utc": str(final["test_start"]),
        "final_holdout_test_end_utc": str(final["test_end"]),
    }
    print(
        f"H{horizon} untouched holdout: {len(final_holdout):,} rows; "
        f"origin interval [{final['test_start']}, {final['test_end']}) UTC; "
        f"targets {holdout_period['final_holdout_target_start_utc']} through "
        f"{holdout_period['final_holdout_target_end_utc']} UTC."
    )
    print(
        f"H{horizon} matched weather ablation: selection MAE "
        f"{selection_mae:,.1f} MW with weather vs "
        f"{result['selection_no_weather_control_mae']:,.1f} MW without weather "
        f"({selection_weather_control_wins}/{len(selection_results)} windows won); "
        f"untouched holdout {final_result['model']['mae']:,.1f} vs "
        f"{final_result['no_weather_control']['mae']:,.1f} MW."
    )
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name=f"SeasonalResidual_LightGBM_H{horizon}") as run:
            result["run_id"] = run.info.run_id
            mlflow.log_params({**MODEL_PARAMS, "horizon_hours": horizon, "feature_schema_version": FEATURE_SCHEMA_VERSION,
                               "training_examples_table": examples_table,
                               "feature_count": len(features), "selection_windows": len(selection_results),
                               "training_rows": len(pre_holdout), "training_origin_start": str(pre_holdout.forecast_origin_utc.min()),
                               "training_origin_end": str(pre_holdout.forecast_origin_utc.max()),
                               **holdout_period})
            logged_metrics = {"selection_mae": selection_mae, "selection_peak_mae": result["selection_peak_mae"],
                              "selection_baseline_mae": selection_baseline,
                              "selection_weather_control_wins": selection_weather_control_wins,
                              "selection_weather_control_win_rate": selection_weather_control_wins / len(selection_results),
                              "selection_no_weather_control_mae": result["selection_no_weather_control_mae"],
                              "selection_no_weather_control_peak_mae": result["selection_no_weather_control_peak_mae"],
                              "final_holdout_mae": final_result["model"]["mae"],
                              "final_holdout_peak_mae": final_result["model"]["peak_mae"],
                              "final_holdout_no_weather_control_mae": final_result["no_weather_control"]["mae"],
                              "final_holdout_no_weather_control_peak_mae": final_result["no_weather_control"]["peak_mae"],
                              "final_holdout_daily_naive_mae": final_result["daily"]["mae"]}
            baseline_name = "daily"
            for label, model_key in (("model", "model"), ("baseline", baseline_name)):
                peak = final_result["peak_event_metrics"][model_key]
                logged_metrics[f"final_holdout_{label}_complete_peak_origins"] = peak["complete_origins"]
                for metric in ("peak_magnitude_mae_mw", "peak_timing_mae_hours"):
                    if np.isfinite(peak[metric]):
                        logged_metrics[f"final_holdout_{label}_{metric}"] = peak[metric]
            for bucket, bucket_results in final_result["lead_buckets"].items():
                for label, model_key in (("model", "model"), ("baseline", baseline_name)):
                    metric = bucket_results[model_key]
                    if np.isfinite(metric["mae"]):
                        logged_metrics[f"final_holdout_{bucket}_{label}_mae"] = metric["mae"]
                    logged_metrics[f"final_holdout_{bucket}_{label}_rows"] = metric["sample_count"]
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
                    f"selection_window_{window_index}_no_weather_control_mae": window_result["no_weather_control"]["mae"],
                    f"selection_window_{window_index}_no_weather_control_peak_mae": window_result["no_weather_control"]["peak_mae"],
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
                               "weather_control_policy": "same_rows_same_folds_no_weather_ablation",
                               "minimum_weather_control_win_rate": MIN_WINDOW_WIN_RATE})
            mlflow.log_dict({"horizon_hours": horizon, "importance_type": "gain",
                             "features": [{"feature_name": name, "gain_importance": float(value)}
                                          for name, value in zip(booster.feature_name(), booster.feature_importance("gain"))]},
                            "feature_importance.json")
            mlflow.set_tags({"pipeline_role": "CHAMPION_CANDIDATE", "target_horizon": f"H{horizon}",
                             "model_type": "seasonal_naive_residual_lightgbm", "weather_assumption": "point_in_time_ecmwf_forecast",
                             "training_origin_cadence": "hourly" if examples_table in {
                                 ROLLING_EXAMPLES_TABLE, WEATHER_ROLLING_EXAMPLES_TABLE
                             } else "unknown",
                             "feature_schema_version": FEATURE_SCHEMA_VERSION,
                             "promotion_policy_version": PROMOTION_POLICY_VERSION})
            # Keep the model's input contract in MLflow so registry consumers
            # can validate the feature columns and types at load time.
            mlflow.lightgbm.log_model(
                booster,
                artifact_path=f"model_seasonal_residual_h{horizon}",
                input_example=pre_holdout[features].head(3),
            )
    return result


if __name__ == "__main__" and "spark" in globals():
    if "dbutils" in globals():
        try:
            dbutils.widgets.get("examples_table")
        except Exception:
            dbutils.widgets.text("examples_table", WEATHER_ROLLING_EXAMPLES_TABLE, "Training examples table")
        examples_table = dbutils.widgets.get("examples_table")
    else:
        examples_table = WEATHER_ROLLING_EXAMPLES_TABLE
    examples_table = validate_examples_table(examples_table)

    if not spark.catalog.tableExists(examples_table):
        raise RuntimeError(f"Training examples table does not exist: {examples_table}")

    from pyspark.sql import functions as F

    results = {}
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    for horizon in HORIZONS:
        examples_sdf = spark.table(examples_table).filter(F.col("horizon_hours") == horizon)
        # Project only training, temporal-contract, and scoring columns before
        # crossing the Spark/Pandas boundary. The source table also contains
        # serving/lineage fields that the trainer never reads.
        feature_columns = features_for_horizon(horizon)
        required_columns = set(EXAMPLE_GRAIN_KEYS) | set(feature_columns) | {
            "source_max_observation_utc", "actual_demand_mw", "is_valid_training_example",
            "daily_naive_mw", "eia_day_ahead_forecast_mw",
        }
        projected_columns = [name for name in examples_sdf.columns if name in required_columns]
        missing_columns = required_columns - set(projected_columns)
        if missing_columns:
            raise RuntimeError(
                f"Training source {examples_table} is missing required H{horizon} columns: "
                f"{sorted(missing_columns)}"
            )
        examples_sdf = examples_sdf.select(*projected_columns)
        examples = examples_sdf.toPandas()
        print(f"H{horizon}: collected {len(examples):,} projected rows from {examples_table}.")
        results[horizon] = train_single_horizon_model(
            examples,
            horizon,
            examples_table=examples_table,
        )
        del examples
    for horizon, result in results.items():
        baseline_name = "daily"
        baseline_label = "daily naive"
        line = (
            f"H{horizon} candidate {result['run_id'] or 'not logged'}: "
            f"rolling MAE {result['selection_mae']:,.1f} vs {result['selection_baseline_mae']:,.1f} MW {baseline_label}; "
            f"peak MAE {result['selection_peak_mae']:,.1f} vs {result['selection_baseline_peak_mae']:,.1f} MW; "
            f"window wins {result['selection_window_wins']}/{result['selection_windows']}; "
            f"final holdout MAE {result['final']['model']['mae']:,.1f} vs "
            f"{result['final'][baseline_name]['mae']:,.1f} MW {baseline_label}."
        )
        matched = result["final"]["eia_matched"]
        line += f" EIA matched hours: {matched['sample_count']:,} ({matched['coverage_pct']:.1f}% coverage)."
        print(line)
        model_peak = result["final"]["peak_event_metrics"]["model"]
        baseline_peak = result["final"]["peak_event_metrics"][baseline_name]
        print(
            f"  Holdout peak events: {model_peak['complete_origins']} complete forecast origins; "
            f"magnitude MAE {model_peak['peak_magnitude_mae_mw']:,.1f} vs "
            f"{baseline_peak['peak_magnitude_mae_mw']:,.1f} MW {baseline_label}; "
            f"timing MAE {model_peak['peak_timing_mae_hours']:,.1f} vs "
            f"{baseline_peak['peak_timing_mae_hours']:,.1f} hours."
        )
        for bucket, metrics in result["final"]["lead_buckets"].items():
            model_bucket, baseline_bucket = metrics["model"], metrics[baseline_name]
            print(
                f"  Holdout {bucket.replace('_', ' ')}: MAE "
                f"{model_bucket['mae']:,.1f} vs {baseline_bucket['mae']:,.1f} MW "
                f"{baseline_label}; rows {model_bucket['sample_count']:,}."
            )
        if result["run_id"]:
            try:
                dbutils.jobs.taskValues.set(key=f"candidate_run_id_h{horizon}", value=result["run_id"])
            except Exception as exc:
                # Interactive candidate runs have no downstream validation
                # task; the MLflow run remains available for manual review.
                print(f"Candidate run ID is logged in MLflow; task value unavailable: {exc}")
