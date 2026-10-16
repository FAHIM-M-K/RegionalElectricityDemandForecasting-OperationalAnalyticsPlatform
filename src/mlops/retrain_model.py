# Databricks notebook source
# MAGIC %md
# MAGIC # MLOps Automated Retraining: EIA-Hybrid Model Candidate
# MAGIC
# MAGIC **Input Table:** `main.gold.features_hourly`  
# MAGIC **Output Model:** Logged to MLflow as `LightGBM_EIA_Hybrid_Candidate_{YYYYMMDD}`  
# MAGIC **Schedule:** Monthly via Databricks Workflows DAG
# MAGIC
# MAGIC ### Logic Overview:
# MAGIC 1. Dynamic Rolling-Window Slicing:
# MAGIC    - Train: 36 months
# MAGIC    - Validation: 3 months
# MAGIC    - Holdout: 1 month
# MAGIC 2. Features: Imported from single source of truth `src.shared.feature_transforms`
# MAGIC 3. Model: LightGBM Error-Correction Production Architecture
# MAGIC 4. Downstream Inter-Task Messaging: Emits `candidate_run_id` and holdout metrics via `dbutils.jobs.taskValues`.

# COMMAND ----------
# MAGIC %pip install lightgbm --quiet

# COMMAND ----------
# DBTITLE 1,Imports & Platform Configuration
import os
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
try:
    from pyspark.sql import functions as F
except ImportError:
    F = None

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from mlflow.models.signature import infer_signature

# Dynamic repository locator for Databricks workspace execution
_repo_root = os.getcwd()
for _ in range(5):
    if os.path.isdir(os.path.join(_repo_root, "configs")):
        if _repo_root not in sys.path:
            sys.path.insert(0, _repo_root)
        break
    _parent = os.path.dirname(_repo_root)
    if _parent == _repo_root:
        break
    _repo_root = _parent

from configs.loader import load_platform_config
from src.shared.feature_transforms import (
    FEATURE_COLS,
    BASE_COLS,
    TARGET_COL,
    BENCHMARK_COL,
    TIME_COL,
    compute_derived_features,
)

config = load_platform_config("ercot_config.json")
BA_CODE = config.get("region", {}).get("code", "ERCO")
CATALOG = "main"
FEATURES_TABLE = f"{CATALOG}.gold.features_hourly"
EXPERIMENT_NAME = "/Shared/regional_demand_forecasting"

mlflow.set_experiment(EXPERIMENT_NAME)

try:
    spark.sql(f"USE CATALOG {CATALOG}")
    spark.sql("USE SCHEMA gold")
except Exception as e:
    print(f"Catalog setup notice: {e}")

# COMMAND ----------
# DBTITLE 1,Dynamic Rolling Window Definition (No Hardcoded Dates)
def get_rolling_window_dates(max_ts: pd.Timestamp):
    """
    Computes dynamic rolling-window boundaries relative to the latest available data.
    - Holdout (Test): 1 month
    - Validation: 3 months
    - Training: 36 months
    """
    holdout_end = max_ts
    holdout_start = holdout_end - pd.DateOffset(months=1)
    val_end = holdout_start
    val_start = val_end - pd.DateOffset(months=3)
    train_end = val_start
    train_start = train_end - pd.DateOffset(months=36)
    
    return {
        "train_start": train_start,
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
        "holdout_start": holdout_start,
        "holdout_end": holdout_end,
    }

# COMMAND ----------
# DBTITLE 1,Extract Ingestion Data & Apply Shared Transformations
print(f"Loading eligible training records from {FEATURES_TABLE} for BA: {BA_CODE}...")

try:
    df_spark = (
        spark.table(FEATURES_TABLE)
        .filter(F.col("respondent") == BA_CODE)
        .filter(F.col("is_valid_training_sample_h24") == True)
        .filter(F.col(BENCHMARK_COL).isNotNull())
        .select(*[c for c in BASE_COLS if c in spark.table(FEATURES_TABLE).columns])
    )
    pdf_raw = df_spark.toPandas()
except Exception as e:
    print(f"Reading from features table skipped in local mode: {e}")
    pdf_raw = pd.DataFrame()

if len(pdf_raw) == 0:
    print("Generating standard rolling simulation data for standalone test execution...")
    date_rng = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=8760 * 3, freq="h")
    sim_base = 45000 + 18000 * np.sin((date_rng.hour - 6) * np.pi / 12)
    pdf_raw = pd.DataFrame({
        "timestamp_utc": date_rng,
        "respondent": BA_CODE,
        "actual_demand_mw": sim_base + np.random.normal(0, 800, len(date_rng)),
        "day_ahead_forecast_mw": sim_base + np.random.normal(300, 1200, len(date_rng)),
        "hour_local": date_rng.hour,
        "day_of_week_local": date_rng.dayofweek,
        "weighted_temperature_f": 75.0 + 15.0 * np.sin(date_rng.hour * np.pi / 12),
        "weighted_apparent_temperature_f": 77.0 + 18.0 * np.sin(date_rng.hour * np.pi / 12),
        "weighted_solar_irradiance_wm2": (np.sin((date_rng.hour - 6) * np.pi / 12).clip(min=0) * 800).astype(float),
        "cdh_55": np.maximum(0, (75.0 + 15.0 * np.sin(date_rng.hour * np.pi / 12)) - 55.2),
        "hdh_55": np.maximum(0, 55.2 - (75.0 + 15.0 * np.sin(date_rng.hour * np.pi / 12))),
        "temp_deviation_55_sq": ((75.0 + 15.0 * np.sin(date_rng.hour * np.pi / 12)) - 55.2)**2,
        "temp_delta_dfw_houston_c": np.random.normal(2.0, 1.0, len(date_rng)),
        "hour_sin": np.sin(2 * np.pi * date_rng.hour / 24),
        "hour_cos": np.cos(2 * np.pi * date_rng.hour / 24),
        "day_of_week_sin": np.sin(2 * np.pi * date_rng.dayofweek / 7),
        "day_of_week_cos": np.cos(2 * np.pi * date_rng.dayofweek / 7),
        "month_sin": np.sin(2 * np.pi * date_rng.month / 12),
        "month_cos": np.cos(2 * np.pi * date_rng.month / 12),
        "day_of_year_sin": np.sin(2 * np.pi * date_rng.dayofyear / 365),
        "day_of_year_cos": np.cos(2 * np.pi * date_rng.dayofyear / 365),
        "is_weekend": (date_rng.dayofweek >= 5).astype(float),
        "is_holiday": 0.0,
        "is_major_shutdown_holiday": 0.0,
        "is_holiday_adjacent": 0.0,
        "season_summer": date_rng.month.isin([6, 7, 8]).astype(float),
        "season_winter": date_rng.month.isin([12, 1, 2]).astype(float),
        "season_shoulder": (~date_rng.month.isin([6, 7, 8, 12, 1, 2])).astype(float),
        "demand_lag_24h": sim_base,
        "demand_lag_48h": sim_base,
        "demand_lag_168h": sim_base,
        "demand_rolling_avg_24h_lag24": sim_base,
        "demand_rolling_std_24h_lag24": 1500.0,
    })

pdf_all = compute_derived_features(pdf_raw)
latest_timestamp = pdf_all[TIME_COL].max()
windows = get_rolling_window_dates(latest_timestamp)

print(f"\nDynamic Rolling Window Splits (Latest Record: {latest_timestamp.strftime('%Y-%m-%d %H:%M UTC')}):")
print(f"  - Train Split (36 mo):   {windows['train_start'].strftime('%Y-%m-%d')} -> {windows['train_end'].strftime('%Y-%m-%d')}")
print(f"  - Validation (3 mo):     {windows['val_start'].strftime('%Y-%m-%d')} -> {windows['val_end'].strftime('%Y-%m-%d')}")
print(f"  - Holdout Test (1 mo):   {windows['holdout_start'].strftime('%Y-%m-%d')} -> {windows['holdout_end'].strftime('%Y-%m-%d')}")

# COMMAND ----------
# DBTITLE 1,Partition Datasets & Build Residual Target
mask_train = (pdf_all[TIME_COL] >= windows["train_start"]) & (pdf_all[TIME_COL] < windows["train_end"])
mask_val   = (pdf_all[TIME_COL] >= windows["val_start"]) & (pdf_all[TIME_COL] < windows["val_end"])
mask_test  = (pdf_all[TIME_COL] >= windows["holdout_start"]) & (pdf_all[TIME_COL] <= windows["holdout_end"])

train_data = pdf_all[mask_train].copy()
val_data   = pdf_all[mask_val].copy()
test_data  = pdf_all[mask_test].copy()

print(f"\nSamples by Split:")
print(f"  Train:   {len(train_data):,} hours")
print(f"  Val:     {len(val_data):,} hours")
print(f"  Holdout: {len(test_data):,} hours")

assert len(train_data) >= 8760, f"Insufficient training data: {len(train_data)} hours (< 1 year)"
assert len(val_data) >= 720, f"Insufficient validation data: {len(val_data)} hours"
assert len(test_data) >= 672, f"Insufficient holdout data: {len(test_data)} hours"

X_train = train_data[FEATURE_COLS]
y_train_eia_err = train_data[TARGET_COL] - train_data[BENCHMARK_COL]

X_val = val_data[FEATURE_COLS]
y_val_eia_err = val_data[TARGET_COL] - val_data[BENCHMARK_COL]

X_test = test_data[FEATURE_COLS]
y_test_eia_err = test_data[TARGET_COL] - test_data[BENCHMARK_COL]
y_test_actual = test_data[TARGET_COL].values
y_test_eia = test_data[BENCHMARK_COL].values

# COMMAND ----------
# DBTITLE 1,Train Candidate Model
params_hybrid = {
    "objective": "regression_l1",
    "metric": "l1",
    "boosting_type": "gbdt",
    "n_estimators": 1500,
    "learning_rate": 0.03,
    "num_leaves": 45,
    "max_depth": 8,
    "min_child_samples": 25,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

model_candidate = lgb.LGBMRegressor(**params_hybrid)
callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=True),
    lgb.log_evaluation(period=200),
]

print("Fitting Candidate EIA Hybrid LightGBM Model...")
model_candidate.fit(
    X_train,
    y_train_eia_err,
    eval_set=[(X_val, y_val_eia_err)],
    callbacks=callbacks,
)

best_iter = int(model_candidate.best_iteration_)
print(f"Candidate training completed! Best iteration: {best_iter}")

# COMMAND ----------
# DBTITLE 1,Evaluate Candidate on 1-Month Holdout
pred_residual = model_candidate.predict(X_test)
pred_candidate_mw = y_test_eia + pred_residual

# Calculate metrics
err_candidate = pred_candidate_mw - y_test_actual
abs_err_candidate = np.abs(err_candidate)
candidate_mae = float(np.mean(abs_err_candidate))
candidate_rmse = float(np.sqrt(np.mean(err_candidate**2)))
candidate_mape = float(np.mean((abs_err_candidate / y_test_actual) * 100.0))
candidate_bias = float(np.mean(err_candidate))

hour_local_test = test_data["hour_local"].values
peak_mask = (hour_local_test >= 14) & (hour_local_test <= 18)
candidate_peak_mae = float(np.mean(abs_err_candidate[peak_mask])) if np.any(peak_mask) else np.nan

# Official EIA benchmark on same holdout
err_eia = y_test_eia - y_test_actual
abs_err_eia = np.abs(err_eia)
eia_mae = float(np.mean(abs_err_eia))
eia_peak_mae = float(np.mean(abs_err_eia[peak_mask])) if np.any(peak_mask) else np.nan

lift_vs_eia = ((eia_mae - candidate_mae) / eia_mae) * 100.0
peak_lift_vs_eia = ((eia_peak_mae - candidate_peak_mae) / eia_peak_mae) * 100.0

print(f"\n=================== CANDIDATE HOLDOUT EVALUATION ===================")
print(f"Holdout Window:           {windows['holdout_start'].strftime('%Y-%m-%d')} to {windows['holdout_end'].strftime('%Y-%m-%d')}")
print(f"Candidate Global MAE:     {candidate_mae:,.1f} MW ({lift_vs_eia:+.2f}% vs EIA {eia_mae:,.1f} MW)")
print(f"Candidate Peak MAE:       {candidate_peak_mae:,.1f} MW ({peak_lift_vs_eia:+.2f}% vs EIA {eia_peak_mae:,.1f} MW)")
print(f"Candidate RMSE:           {candidate_rmse:,.1f} MW")
print(f"Candidate MAPE:           {candidate_mape:.2f}%")
print(f"Candidate Bias:           {candidate_bias:+,.1f} MW")
print(f"====================================================================")

# COMMAND ----------
# DBTITLE 1,Log Candidate Model to MLflow
run_date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
candidate_run_name = f"LightGBM_EIA_Hybrid_Candidate_{run_date_str}"

input_example = X_train.head(5)
signature = infer_signature(input_example, y_train_eia_err.head(5))

with mlflow.start_run(run_name=candidate_run_name) as run:
    candidate_run_id = run.info.run_id
    
    # Tags
    mlflow.set_tag("pipeline_role", "CANDIDATE")
    mlflow.set_tag("model_type", "LightGBM_EIA_Hybrid")
    mlflow.set_tag("candidate_date", run_date_str)
    mlflow.set_tag("holdout_start", windows["holdout_start"].strftime("%Y-%m-%d"))
    mlflow.set_tag("holdout_end", windows["holdout_end"].strftime("%Y-%m-%d"))
    
    # Parameters
    mlflow.log_params(params_hybrid)
    mlflow.log_param("best_iteration", best_iter)
    mlflow.log_param("feature_count", len(FEATURE_COLS))
    mlflow.log_param("train_samples", len(train_data))
    mlflow.log_param("val_samples", len(val_data))
    mlflow.log_param("holdout_samples", len(test_data))
    
    # Metrics
    mlflow.log_metric("holdout_mae", candidate_mae)
    mlflow.log_metric("holdout_rmse", candidate_rmse)
    mlflow.log_metric("holdout_mape", candidate_mape)
    mlflow.log_metric("holdout_bias", candidate_bias)
    mlflow.log_metric("holdout_peak_mae", candidate_peak_mae)
    mlflow.log_metric("lift_vs_eia_pct", lift_vs_eia)
    mlflow.log_metric("peak_lift_vs_eia_pct", peak_lift_vs_eia)
    
    # Model Artifact
    mlflow.lightgbm.log_model(
        lgb_model=model_candidate.booster_,
        artifact_path="lightgbm_eia_hybrid",
        signature=signature,
        input_example=input_example,
    )
    
    # Feature Importances artifact
    df_feat_imp = pd.DataFrame({
        "feature_name": FEATURE_COLS,
        "gain_importance": model_candidate.booster_.feature_importance(importance_type="gain"),
        "split_importance": model_candidate.booster_.feature_importance(importance_type="split"),
    }).sort_values("gain_importance", ascending=False)
    
    temp_imp_csv = f"/tmp/candidate_feature_importance_{run_date_str}.csv"
    try:
        df_feat_imp.to_csv(temp_imp_csv, index=False)
        mlflow.log_artifact(temp_imp_csv, artifact_path="feature_importance")
    except Exception as e:
        pass

print(f"\nCandidate logged to MLflow successfully:")
print(f"  Run Name: {candidate_run_name}")
print(f"  Run ID:   {candidate_run_id}")

# COMMAND ----------
# DBTITLE 1,Pass Task Values Downstream to Validation Gate
try:
    dbutils.jobs.taskValues.set(key="candidate_run_id", value=candidate_run_id)
    dbutils.jobs.taskValues.set(key="candidate_run_name", value=candidate_run_name)
    dbutils.jobs.taskValues.set(key="candidate_mae", value=float(candidate_mae))
    dbutils.jobs.taskValues.set(key="candidate_peak_mae", value=float(candidate_peak_mae))
    print("Emitted task values via dbutils.jobs.taskValues successfully!")
except NameError:
    print("dbutils not found (running in standalone/test mode). Task values output skipped.")
