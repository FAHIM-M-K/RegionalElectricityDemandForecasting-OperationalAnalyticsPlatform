# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 4: Modeling — Step 3: Champion Model (LightGBM Standalone & EIA Hybrid)
# MAGIC
# MAGIC **Input Table:** `main.gold.features_hourly`  
# MAGIC
# MAGIC ### Core Modeling Tracks:
# MAGIC 1. **Track 1: LightGBM Standalone (Weather & Physical Delta):**
# MAGIC    - Completely independent of institutional forecasts.
# MAGIC    - Predicts day-over-day demand shift ($\Delta y_t = y_t - y_{t-24}$) using scale-invariant features:
# MAGIC      - Discrete hours (`hour_local`) and day-of-week (`day_of_week_local`)
# MAGIC      - Apparent temperature (heat index / wind chill) and temperature shifts ($\Delta \text{temp}_{24}, \Delta \text{apparent}_{24}$)
# MAGIC      - Solar irradiance and late-afternoon peak interactions (`peak_temp_interaction`)
# MAGIC      - 24h rolling thermal inertia (`temp_rolling_24h`)
# MAGIC 2. **Track 2: LightGBM + EIA Hybrid (Institutional Error-Correction Champion):**
# MAGIC    - Learns the systematic residual errors of the official EIA forecast ($e_t = y_t - F_{\text{EIA}, t}$).
# MAGIC    - Corrects EIA's known +368 MW overforecasting bias and heatwave ramp lags to achieve target $\ge 15\%$ lift.

# COMMAND ----------
# MAGIC %pip install lightgbm --quiet

# COMMAND ----------
# DBTITLE 1,Imports & Setup
import os
import sys
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from mlflow.models.signature import infer_signature

CATALOG = "main"
FEATURES_TABLE = f"{CATALOG}.gold.features_hourly"

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql("USE SCHEMA gold")

# COMMAND ----------
# DBTITLE 1,Dynamic Configuration Locator
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

# COMMAND ----------
# DBTITLE 1,Define Feature Architecture & Leakage Contract
print(f"Selected {len(FEATURE_COLS)} stationary scale-invariant features for Modeling:")
for i, col in enumerate(FEATURE_COLS, 1):
    print(f"  {i:02d}. {col}")

# COMMAND ----------
# DBTITLE 1,Extract Datasets & Compute In-Memory Physical Features
df_spark = (
    spark.table(FEATURES_TABLE)
    .filter(F.col("respondent") == BA_CODE)
    .filter(F.col("is_valid_training_sample_h24") == True)
    .filter(F.col(BENCHMARK_COL).isNotNull())
    .select(*[c for c in BASE_COLS if c in spark.table(FEATURES_TABLE).columns])
)

pdf_raw = df_spark.toPandas()
pdf_all = compute_derived_features(pdf_raw)

# Chronological split masks
SPLIT_TRAIN_END = pd.to_datetime("2024-01-01 00:00:00")
SPLIT_VAL_END   = pd.to_datetime("2024-07-01 00:00:00")

train_mask = pdf_all[TIME_COL] < SPLIT_TRAIN_END
val_mask   = (pdf_all[TIME_COL] >= SPLIT_TRAIN_END) & (pdf_all[TIME_COL] < SPLIT_VAL_END)
test_mask  = pdf_all[TIME_COL] >= SPLIT_VAL_END

X_train = pdf_all.loc[train_mask, FEATURE_COLS].copy()
y_train = pdf_all.loc[train_mask, TARGET_COL].copy()

X_val = pdf_all.loc[val_mask, FEATURE_COLS].copy()
y_val = pdf_all.loc[val_mask, TARGET_COL].copy()

X_test = pdf_all.loc[test_mask, FEATURE_COLS].copy()
y_test = pdf_all.loc[test_mask, TARGET_COL].copy()
test_meta = pdf_all.loc[test_mask, [TIME_COL, TARGET_COL, BENCHMARK_COL, "hour_local", "demand_lag_24h", "demand_lag_168h"]].copy()

# Target Formulations:
# 1. Delta Target (for standalone model): y_delta = actual_demand_mw - demand_lag_24h
y_train_delta = y_train - pdf_all.loc[train_mask, "demand_lag_24h"]
y_val_delta   = y_val - pdf_all.loc[val_mask, "demand_lag_24h"]

# 2. Residual Target (for EIA error-correction hybrid): y_eia_err = actual_demand_mw - day_ahead_forecast_mw
y_train_eia_err = y_train - pdf_all.loc[train_mask, BENCHMARK_COL]
y_val_eia_err   = y_val - pdf_all.loc[val_mask, BENCHMARK_COL]

print(f"Dataset Split Sizes:")
print(f"  - Train Cohort (2021-03 through 2023): {len(X_train):,} samples")
print(f"  - Validation Cohort (2024-H1):         {len(X_val):,} samples")
print(f"  - Holdout Test Cohort (2024-H2):       {len(X_test):,} samples")

# COMMAND ----------
# DBTITLE 1,Train Track 1: LightGBM Standalone (Physical Delta Model)
params_standalone = {
    "objective": "regression_l1",
    "metric": "l1",
    "boosting_type": "gbdt",
    "n_estimators": 1500,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": 9,
    "min_child_samples": 20,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1
}

model_standalone = lgb.LGBMRegressor(**params_standalone)

print("Training Track 1: LightGBM Standalone (Physical Delta Model)...")
callbacks_standalone = [
    lgb.early_stopping(stopping_rounds=50, verbose=True),
    lgb.log_evaluation(period=200)
]

model_standalone.fit(
    X_train,
    y_train_delta,
    eval_set=[(X_val, y_val_delta)],
    callbacks=callbacks_standalone
)

# Predict Standalone Forecast: y_pred = y_(t-24) + delta_pred
y_pred_delta = model_standalone.predict(X_test)
test_meta["pred_standalone"] = test_meta["demand_lag_24h"].values + y_pred_delta
print(f"Track 1 completed! Optimal trees: {model_standalone.best_iteration_}")

# COMMAND ----------
# DBTITLE 1,Train Track 2: LightGBM + EIA Hybrid (Error Corrector)
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
    "verbose": -1
}

model_hybrid = lgb.LGBMRegressor(**params_hybrid)

print("\nTraining Track 2: LightGBM + EIA Hybrid (Error-Correction Champion)...")
callbacks_hybrid = [
    lgb.early_stopping(stopping_rounds=50, verbose=True),
    lgb.log_evaluation(period=200)
]

model_hybrid.fit(
    X_train,
    y_train_eia_err,
    eval_set=[(X_val, y_val_eia_err)],
    callbacks=callbacks_hybrid
)

# Predict Hybrid Forecast: y_pred = EIA_forecast + predicted_eia_error
y_pred_eia_correction = model_hybrid.predict(X_test)
test_meta["pred_hybrid"] = test_meta[BENCHMARK_COL].values + y_pred_eia_correction
print(f"Track 2 completed! Optimal trees: {model_hybrid.best_iteration_}")

# COMMAND ----------
# DBTITLE 1,Standard Operational Metrics Evaluator
def evaluate_series(actual, pred, hour_local, model_name="Model"):
    error = pred - actual
    abs_error = np.abs(error)
    sq_error = error ** 2
    ape = (abs_error / actual) * 100.0
    
    mae = float(np.mean(abs_error))
    rmse = float(np.sqrt(np.mean(sq_error)))
    mape = float(np.mean(ape))
    bias = float(np.mean(error))
    p90 = float(np.percentile(abs_error, 90))
    
    # Peak window (14:00 - 18:00 local time)
    peak_mask = (hour_local >= 14) & (hour_local <= 18)
    peak_mae = float(np.mean(abs_error[peak_mask])) if np.any(peak_mask) else np.nan
    peak_p90 = float(np.percentile(abs_error[peak_mask], 90)) if np.any(peak_mask) else np.nan
    
    # Extreme conditions (> 75,000 MW)
    ext_mask = actual >= 75000.0
    ext_mae = float(np.mean(abs_error[ext_mask])) if np.any(ext_mask) else np.nan
    
    return {
        "Model": model_name,
        "Hours": len(actual),
        "MAE (MW)": round(mae, 1),
        "RMSE (MW)": round(rmse, 1),
        "MAPE (%)": round(mape, 2),
        "Bias (MW)": round(bias, 1),
        "P90 Error (MW)": round(p90, 1),
        "Peak MAE (MW)": round(peak_mae, 1),
        "Peak P90 (MW)": round(peak_p90, 1),
        "Extreme (>75GW) MAE": round(ext_mae, 1) if not np.isnan(ext_mae) else "N/A"
    }

# COMMAND ----------
# DBTITLE 1,Official Scorecard Comparison: All Models
scorecard = []

# 1. Naive Persistence (t-24)
scorecard.append(evaluate_series(
    actual=test_meta[TARGET_COL].values,
    pred=test_meta["demand_lag_24h"].values,
    hour_local=test_meta["hour_local"].values,
    model_name="1. Persistence (t-24)"
))

# 2. Seasonal Naive (t-168)
scorecard.append(evaluate_series(
    actual=test_meta[TARGET_COL].values,
    pred=test_meta["demand_lag_168h"].values,
    hour_local=test_meta["hour_local"].values,
    model_name="2. Seasonal Naive (t-168)"
))

# 3. Official EIA Day-Ahead Benchmark
scorecard.append(evaluate_series(
    actual=test_meta[TARGET_COL].values,
    pred=test_meta[BENCHMARK_COL].values,
    hour_local=test_meta["hour_local"].values,
    model_name="3. Official EIA Benchmark"
))

# 4. LightGBM Standalone (Physical Delta)
scorecard.append(evaluate_series(
    actual=test_meta[TARGET_COL].values,
    pred=test_meta["pred_standalone"].values,
    hour_local=test_meta["hour_local"].values,
    model_name="4. LightGBM Standalone (Physical)"
))

# 5. LightGBM + EIA Hybrid (Error Corrector Champion)
scorecard.append(evaluate_series(
    actual=test_meta[TARGET_COL].values,
    pred=test_meta["pred_hybrid"].values,
    hour_local=test_meta["hour_local"].values,
    model_name="5. LightGBM + EIA Hybrid (Champion)"
))

df_scorecard = pd.DataFrame(scorecard)

print("=========================================================================================")
print("  OFFICIAL HOLDOUT TEST SCORECARD — ALL 5 MODELS COMPARED (2024-H2)")
print("=========================================================================================")
display(df_scorecard)

eia_mae = df_scorecard.loc[df_scorecard["Model"].str.contains("EIA Benchmark"), "MAE (MW)"].values[0]
stand_mae = df_scorecard.loc[df_scorecard["Model"].str.contains("Standalone"), "MAE (MW)"].values[0]
hyb_mae = df_scorecard.loc[df_scorecard["Model"].str.contains("Hybrid"), "MAE (MW)"].values[0]

eia_peak = df_scorecard.loc[df_scorecard["Model"].str.contains("EIA Benchmark"), "Peak MAE (MW)"].values[0]
hyb_peak = df_scorecard.loc[df_scorecard["Model"].str.contains("Hybrid"), "Peak MAE (MW)"].values[0]

stand_lift = ((eia_mae - stand_mae) / eia_mae) * 100.0
hyb_lift   = ((eia_mae - hyb_mae) / eia_mae) * 100.0
peak_hyb_lift = ((eia_peak - hyb_peak) / eia_peak) * 100.0

print(f"\n==================== FINAL SCORECARD BENCHMARKS ====================")
print(f"Official EIA Benchmark MAE:         {eia_mae:,.1f} MW")
print(f"Track 1 (LightGBM Standalone) MAE:  {stand_mae:,.1f} MW ({stand_lift:+.2f}% vs EIA)")
print(f"Track 2 (LightGBM + EIA Hybrid) MAE: {hyb_mae:,.1f} MW ({hyb_lift:+.2f}% vs EIA)")
print(f"\nPeak Hours (14:00-18:00) Comparison:")
print(f"Official EIA Peak MAE:              {eia_peak:,.1f} MW")
print(f"Track 2 Hybrid Peak MAE:            {hyb_peak:,.1f} MW ({peak_hyb_lift:+.2f}% vs EIA)")
print("====================================================================")

# COMMAND ----------
# DBTITLE 1,Feature Importance: Physical & Error-Correction Drivers
fig_importance = pd.DataFrame({
    "Feature": FEATURE_COLS,
    "Gain_Standalone": model_standalone.booster_.feature_importance(importance_type="gain"),
    "Gain_Hybrid": model_hybrid.booster_.feature_importance(importance_type="gain")
})
fig_importance["Gain_Pct_Hybrid"] = (fig_importance["Gain_Hybrid"] / fig_importance["Gain_Hybrid"].sum()) * 100.0
fig_importance = fig_importance.sort_values("Gain_Pct_Hybrid", ascending=False).reset_index(drop=True)

print("\n--- Top Features Driving EIA Error Correction (Gain Metric) ---")
display(fig_importance.head(15))

# COMMAND ----------
# DBTITLE 1,Log Both Models & Validated Signatures to MLflow
mlflow.set_experiment("/Shared/regional_demand_forecasting")

input_example = X_train.head(5)
sig_standalone = infer_signature(input_example, y_train_delta.head(5))
sig_hybrid     = infer_signature(input_example, y_train_eia_err.head(5))

# Log Track 1: Standalone Model
with mlflow.start_run(run_name="LightGBM_Standalone_H24"):
    mlflow.log_param("track", "Standalone_Physical_Delta")
    mlflow.log_param("best_iteration", int(model_standalone.best_iteration_))
    mlflow.log_param("feature_count", len(FEATURE_COLS))
    
    m_stand = next(r for r in scorecard if "Standalone" in r["Model"])
    mlflow.log_metric("test_mae", m_stand["MAE (MW)"])
    mlflow.log_metric("test_rmse", m_stand["RMSE (MW)"])
    mlflow.log_metric("test_mape", m_stand["MAPE (%)"])
    mlflow.log_metric("test_bias", m_stand["Bias (MW)"])
    mlflow.log_metric("test_peak_mae", m_stand["Peak MAE (MW)"])
    
    mlflow.lightgbm.log_model(
        lgb_model=model_standalone.booster_,
        artifact_path="lightgbm_standalone",
        signature=sig_standalone,
        input_example=input_example
    )

# Log Track 2: Champion Hybrid Model
with mlflow.start_run(run_name="LightGBM_EIA_Hybrid_Champion_H24"):
    mlflow.log_param("track", "EIA_Residual_Error_Corrector")
    mlflow.log_param("best_iteration", int(model_hybrid.best_iteration_))
    mlflow.log_param("feature_count", len(FEATURE_COLS))
    
    m_hyb = next(r for r in scorecard if "Hybrid" in r["Model"])
    mlflow.log_metric("test_mae", m_hyb["MAE (MW)"])
    mlflow.log_metric("test_rmse", m_hyb["RMSE (MW)"])
    mlflow.log_metric("test_mape", m_hyb["MAPE (%)"])
    mlflow.log_metric("test_bias", m_hyb["Bias (MW)"])
    mlflow.log_metric("test_peak_mae", m_hyb["Peak MAE (MW)"])
    mlflow.log_metric("lift_over_eia_pct", float(hyb_lift))
    mlflow.log_metric("peak_lift_over_eia_pct", float(peak_hyb_lift))
    
    mlflow.lightgbm.log_model(
        lgb_model=model_hybrid.booster_,
        artifact_path="lightgbm_eia_hybrid",
        signature=sig_hybrid,
        input_example=input_example
    )

print("\nSuccessfully logged both Standalone and Hybrid Champion models to MLflow!")
