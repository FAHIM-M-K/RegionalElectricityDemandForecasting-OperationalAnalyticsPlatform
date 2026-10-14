# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 4: Modeling — Step 3: Champion Non-Linear Model (LightGBM)
# MAGIC
# MAGIC **Input Table:** `main.gold.features_hourly`  
# MAGIC
# MAGIC ### Why LightGBM?
# MAGIC 1. **Non-linear Temperature Splitting:** Captures steep exponential air-conditioning ramps when temperatures cross 95°F–100°F.
# MAGIC 2. **Interaction Effects:** Automatically learns complex conditional logic (e.g., 98°F at 4:00 PM requires far more generation than 98°F at 11:00 AM).
# MAGIC 3. **MAE Loss Optimization (`regression_l1`):** Directly minimizes Mean Absolute Error (our primary target metric) rather than squared errors.
# MAGIC 4. **Speed & Efficiency:** Histogram-based binning trains in seconds on Databricks with minimal memory overhead compared to traditional XGBoost.
# MAGIC
# MAGIC ### Success Hurdle:
# MAGIC - **Target 1 (Global MAE):** Beat EIA Benchmark (1,192.9 MW) by >= 15% (Target: **< 1,014.0 MW**)
# MAGIC - **Target 2 (Peak Hour MAE):** Beat EIA Peak MAE (1,464.6 MW) by >= 15% (Target: **< 1,244.9 MW**)

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

config = load_platform_config("ercot_config.json")
BA_CODE = config.get("region", {}).get("code", "ERCO")

# COMMAND ----------
# DBTITLE 1,Define Feature Architecture & Leakage Contract
# Explicit feature set: strictly observable at day-ahead origin (t0 = T-24)
FEATURE_COLS = [
    # 1. Empirical Thermodynamic Features (55.2°F balance point)
    "cdh_55",
    "hdh_55",
    "temp_deviation_55_sq",
    "temp_delta_dfw_houston_c",
    
    # 2. Raw Temperature & Weather Context
    "weighted_temperature_f",
    
    # 3. Multi-Year Secular Growth Anchor
    "year_local",
    
    # 4. Cyclical Temporal Encodings (smooth circular time)
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    
    # 5. Calendar & Holiday Encodings
    "is_weekend",
    "is_holiday",
    "is_major_shutdown_holiday",
    "is_holiday_adjacent",
    "season_summer",
    "season_winter",
    "season_shoulder",
    
    # 6. Day-Ahead Horizon (H=24) Autoregressive Lags & Rolling Context
    "demand_lag_24h",
    "demand_lag_48h",
    "demand_lag_168h",
    "demand_diff_24_48",
    "demand_diff_24_168",
    "demand_rolling_avg_24h_lag24",
    "demand_rolling_std_24h_lag24"
]

TARGET_COL = "actual_demand_mw"
BENCHMARK_COL = "day_ahead_forecast_mw"
TIME_COL = "timestamp_utc"

print(f"Selected {len(FEATURE_COLS)} leakage-safe features for LightGBM Champion:")
for i, col in enumerate(FEATURE_COLS, 1):
    print(f"  {i:02d}. {col}")

# COMMAND ----------
# DBTITLE 1,Extract Datasets & Apply Strict Chronological Splits
# Separate base columns stored in Delta table from derived momentum columns computed in-memory
BASE_FEATURE_COLS = [c for c in FEATURE_COLS if c not in ["demand_diff_24_48", "demand_diff_24_168"]]
cols_to_pull = [TIME_COL, TARGET_COL, BENCHMARK_COL, "hour_local"] + BASE_FEATURE_COLS

df_spark = (
    spark.table(FEATURES_TABLE)
    .filter(F.col("respondent") == BA_CODE)
    .filter(F.col("is_valid_training_sample_h24") == True)
    .filter(F.col(BENCHMARK_COL).isNotNull())
    .select(*cols_to_pull)
)

pdf_all = df_spark.toPandas()
pdf_all[TIME_COL] = pd.to_datetime(pdf_all[TIME_COL])
pdf_all = pdf_all.sort_values(TIME_COL).reset_index(drop=True)

# Compute Day-Ahead momentum / delta features
pdf_all["demand_diff_24_48"] = pdf_all["demand_lag_24h"] - pdf_all["demand_lag_48h"]
pdf_all["demand_diff_24_168"] = pdf_all["demand_lag_24h"] - pdf_all["demand_lag_168h"]

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

# Delta Target Formulation: y_delta = actual_demand_mw - demand_lag_24h
# Eliminates secular grid growth extrapolation bias: yesterday's demand sets modern base level
y_train_delta = y_train - X_train["demand_lag_24h"]
y_val_delta   = y_val - X_val["demand_lag_24h"]

# Cast boolean flags to float
for c in ["is_weekend", "is_holiday"]:
    if c in X_train.columns:
        X_train[c] = X_train[c].astype(float)
        X_val[c] = X_val[c].astype(float)
        X_test[c] = X_test[c].astype(float)

print(f"Dataset Split Sizes:")
print(f"  - Train Cohort (2021-03 through 2023): {len(X_train):,} samples")
print(f"  - Validation Cohort (2024-H1):         {len(X_val):,} samples")
print(f"  - Holdout Test Cohort (2024-H2):       {len(X_test):,} samples")

# COMMAND ----------
# DBTITLE 1,Train LightGBM with Early Stopping on Validation Cohort
# Directly optimize L1 loss (MAE) to align training directly with operational target
params = {
    "objective": "regression_l1",
    "metric": "l1",
    "boosting_type": "gbdt",
    "n_estimators": 1500,
    "learning_rate": 0.03,
    "num_leaves": 45,
    "max_depth": 8,
    "min_child_samples": 20,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1
}

model = lgb.LGBMRegressor(**params)

print("Training LightGBM Regressor with early stopping on validation cohort...")
callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=True),
    lgb.log_evaluation(period=100)
]

model.fit(
    X_train,
    y_train_delta,
    eval_set=[(X_val, y_val_delta)],
    callbacks=callbacks
)

best_iteration = model.best_iteration_
print(f"Training completed! Optimal trees: {best_iteration}")

# COMMAND ----------
# DBTITLE 1,Generate Predictions on Holdout Test Cohort
# Reconstruct full demand: Forecast = Yesterday's Demand (t-24) + Predicted Delta
y_pred_delta = model.predict(X_test)
y_pred_test = test_meta["demand_lag_24h"].values + y_pred_delta
test_meta["pred_lightgbm"] = y_pred_test

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
# DBTITLE 1,Evaluate Scorecard & Compare Against All Benchmarks
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

# 4. LightGBM Champion
scorecard.append(evaluate_series(
    actual=test_meta[TARGET_COL].values,
    pred=test_meta["pred_lightgbm"].values,
    hour_local=test_meta["hour_local"].values,
    model_name="4. LightGBM Champion (ML)"
))

df_scorecard = pd.DataFrame(scorecard)

print("=========================================================================================")
print("  OFFICIAL HOLDOUT TEST SCORECARD — ALL MODELS COMPARED (2024-H2)")
print("=========================================================================================")
display(df_scorecard)

eia_mae = df_scorecard.loc[df_scorecard["Model"].str.contains("EIA"), "MAE (MW)"].values[0]
lgb_mae = df_scorecard.loc[df_scorecard["Model"].str.contains("LightGBM"), "MAE (MW)"].values[0]
lift_pct = ((eia_mae - lgb_mae) / eia_mae) * 100.0

eia_peak = df_scorecard.loc[df_scorecard["Model"].str.contains("EIA"), "Peak MAE (MW)"].values[0]
lgb_peak = df_scorecard.loc[df_scorecard["Model"].str.contains("LightGBM"), "Peak MAE (MW)"].values[0]
peak_lift = ((eia_peak - lgb_peak) / eia_peak) * 100.0

print(f"\n==================== SCORECARD RESULTS ====================")
print(f"Global MAE: EIA = {eia_mae:,.1f} MW | LightGBM = {lgb_mae:,.1f} MW")
if lift_pct > 0:
    print(f"SUCCESS: LightGBM BEATS EIA benchmark by {lift_pct:.2f}% (Reduction: {eia_mae - lgb_mae:,.1f} MW)")
else:
    print(f"GAP: LightGBM is within {abs(lift_pct):.2f}% of EIA (Gap: {lgb_mae - eia_mae:,.1f} MW)")

print(f"\nPeak Hour (14:00-18:00) MAE: EIA = {eia_peak:,.1f} MW | LightGBM = {lgb_peak:,.1f} MW")
if peak_lift > 0:
    print(f"SUCCESS: LightGBM BEATS EIA on Peak Hours by {peak_lift:.2f}% (Reduction: {eia_peak - lgb_peak:,.1f} MW)")
else:
    print(f"GAP: LightGBM Peak Gap: {lgb_peak - eia_peak:,.1f} MW")
print("===========================================================")

# COMMAND ----------
# DBTITLE 1,Feature Importance: Split & Gain Rankings
importance_df = pd.DataFrame({
    "Feature": FEATURE_COLS,
    "Gain_Importance": model.booster_.feature_importance(importance_type="gain"),
    "Split_Count": model.booster_.feature_importance(importance_type="split")
}).sort_values("Gain_Importance", ascending=False).reset_index(drop=True)

# Normalized percentage gain
importance_df["Gain_Pct"] = (importance_df["Gain_Importance"] / importance_df["Gain_Importance"].sum()) * 100.0

print("\n--- LightGBM Top 15 Feature Importances (Gain Metric) ---")
display(importance_df.head(15))

# COMMAND ----------
# DBTITLE 1,Log Run & Validated Signature to MLflow
mlflow.set_experiment("/Shared/regional_demand_forecasting")

# Infer signature cleanly from input sample and delta target
input_example = X_train.head(5)
signature = infer_signature(input_example, y_train_delta.head(5))

with mlflow.start_run(run_name="LightGBM_Champion_H24"):
    # Log parameters
    mlflow.log_param("model_family", "Gradient_Boosted_Trees")
    mlflow.log_param("library", "LightGBM")
    mlflow.log_param("best_iteration", int(best_iteration))
    mlflow.log_param("feature_count", len(FEATURE_COLS))
    for k, v in params.items():
        mlflow.log_param(k, v)
        
    # Log metrics
    lgb_metrics = next(r for r in scorecard if "LightGBM" in r["Model"])
    mlflow.log_metric("test_mae", lgb_metrics["MAE (MW)"])
    mlflow.log_metric("test_rmse", lgb_metrics["RMSE (MW)"])
    mlflow.log_metric("test_mape", lgb_metrics["MAPE (%)"])
    mlflow.log_metric("test_bias", lgb_metrics["Bias (MW)"])
    mlflow.log_metric("test_p90_error", lgb_metrics["P90 Error (MW)"])
    mlflow.log_metric("test_peak_mae", lgb_metrics["Peak MAE (MW)"])
    mlflow.log_metric("lift_over_eia_pct", float(lift_pct))
    mlflow.log_metric("peak_lift_over_eia_pct", float(peak_lift))
    
    # Log model with explicit signature and input example (eliminates MLflow warning)
    mlflow.lightgbm.log_model(
        lgb_model=model.booster_,
        artifact_path="lightgbm_champion",
        signature=signature,
        input_example=input_example
    )
    print("Successfully logged LightGBM Champion run and signed model artifact to MLflow!")
