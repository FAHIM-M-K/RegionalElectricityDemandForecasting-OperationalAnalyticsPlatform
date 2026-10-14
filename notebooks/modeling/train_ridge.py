# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 4: Modeling — Step 2: Linear Regularized Baseline (Ridge Regression)
# MAGIC
# MAGIC **Input Table:** `main.gold.features_hourly`  
# MAGIC
# MAGIC ### Objectives:
# MAGIC 1. Train a regularized linear model (`RidgeCV`) using engineered thermodynamic, calendar, and autoregressive lag features.
# MAGIC 2. Measure out-of-sample accuracy on the exact same 2024-H2 holdout test set (19,916 hours).
# MAGIC 3. Compare directly against the EIA benchmark (1,192.9 MW MAE) and naive baselines.
# MAGIC 4. Inspect linear feature coefficients to interpret physical drivers.
# MAGIC 5. Log parameters, metrics, and model artifacts to **MLflow**.

# COMMAND ----------
# DBTITLE 1,Imports & Setup
import os
import sys
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

import mlflow
import mlflow.sklearn
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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
    
    # 2. Cyclical Temporal Encodings (smooth circular time)
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    
    # 3. Calendar & Holiday Encodings
    "is_weekend",
    "is_holiday",
    "is_major_shutdown_holiday",
    "is_holiday_adjacent",
    "season_summer",
    "season_winter",
    "season_shoulder",
    
    # 4. Day-Ahead Horizon (H=24) Autoregressive Lags & Rolling Context
    "demand_lag_24h",
    "demand_lag_48h",
    "demand_lag_168h",
    "demand_rolling_avg_24h_lag24",
    "demand_rolling_std_24h_lag24"
]

TARGET_COL = "actual_demand_mw"
BENCHMARK_COL = "day_ahead_forecast_mw"
TIME_COL = "timestamp_utc"

print(f"Selected {len(FEATURE_COLS)} leakage-safe features for Ridge Regression:")
for i, col in enumerate(FEATURE_COLS, 1):
    print(f"  {i:02d}. {col}")

# COMMAND ----------
# DBTITLE 1,Extract Datasets & Apply Strict Chronological Splits
cols_to_pull = [TIME_COL, TARGET_COL, BENCHMARK_COL, "hour_local"] + FEATURE_COLS

df_spark = (
    spark.table(FEATURES_TABLE)
    .filter(F.col("respondent") == BA_CODE)
    .filter(F.col("is_valid_training_sample_h24") == True)
    .filter(F.col(BENCHMARK_COL).isNotNull())
    .select(*cols_to_pull)
)

# Convert to Pandas for scikit-learn model training
pdf_all = df_spark.toPandas()

# Sort chronologically
pdf_all[TIME_COL] = pd.to_datetime(pdf_all[TIME_COL])
pdf_all = pdf_all.sort_values(TIME_COL).reset_index(drop=True)

# Chronological split masks
SPLIT_TRAIN_END = pd.to_datetime("2024-01-01 00:00:00")
SPLIT_VAL_END   = pd.to_datetime("2024-07-01 00:00:00")

train_mask = pdf_all[TIME_COL] < SPLIT_TRAIN_END
val_mask   = (pdf_all[TIME_COL] >= SPLIT_TRAIN_END) & (pdf_all[TIME_COL] < SPLIT_VAL_END)
test_mask  = pdf_all[TIME_COL] >= SPLIT_VAL_END

X_train = pdf_all.loc[train_mask, FEATURE_COLS]
y_train = pdf_all.loc[train_mask, TARGET_COL]

X_val = pdf_all.loc[val_mask, FEATURE_COLS]
y_val = pdf_all.loc[val_mask, TARGET_COL]

X_test = pdf_all.loc[test_mask, FEATURE_COLS]
y_test = pdf_all.loc[test_mask, TARGET_COL]
test_meta = pdf_all.loc[test_mask, [TIME_COL, TARGET_COL, BENCHMARK_COL, "hour_local", "demand_lag_24h", "demand_lag_168h"]].copy()

print(f"Dataset Split Sizes:")
print(f"  - Train: {len(X_train):,} samples")
print(f"  - Val:   {len(X_val):,} samples")
print(f"  - Test:  {len(X_test):,} samples")

# Cast boolean flags to float for modeling
for c in ["is_weekend", "is_holiday"]:
    if c in X_train.columns:
        X_train[c] = X_train[c].astype(float)
        X_val[c] = X_val[c].astype(float)
        X_test[c] = X_test[c].astype(float)

# COMMAND ----------
# DBTITLE 1,Train Ridge Regression with Cross-Validated Regularization
# Combine Train and Validation for cross-validated ridge penalty selection
X_train_val = pd.concat([X_train, X_val], axis=0)
y_train_val = pd.concat([y_train, y_val], axis=0)

# Alphas to sweep across logarithmic scale
alphas = np.logspace(-2, 4, 20)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_absolute_error"))
])

print("Training RidgeCV model across 5-fold CV...")
pipeline.fit(X_train_val, y_train_val)

best_alpha = pipeline.named_steps["ridge"].alpha_
print(f"Optimal L2 Regularization Parameter (alpha): {best_alpha:.4f}")

# COMMAND ----------
# DBTITLE 1,Generate Predictions on Holdout Test Cohort
y_pred_test = pipeline.predict(X_test)
test_meta["pred_ridge"] = y_pred_test

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
# DBTITLE 1,Evaluate Scorecard & Compare with EIA Benchmark
results = []

# 1. Naive Persistence (t-24)
results.append(evaluate_series(
    actual=test_meta[TARGET_COL].values,
    pred=test_meta["demand_lag_24h"].values,
    hour_local=test_meta["hour_local"].values,
    model_name="1. Persistence (t-24)"
))

# 2. Seasonal Naive (t-168)
results.append(evaluate_series(
    actual=test_meta[TARGET_COL].values,
    pred=test_meta["demand_lag_168h"].values,
    hour_local=test_meta["hour_local"].values,
    model_name="2. Seasonal Naive (t-168)"
))

# 3. Official EIA Day-Ahead Forecast
results.append(evaluate_series(
    actual=test_meta[TARGET_COL].values,
    pred=test_meta[BENCHMARK_COL].values,
    hour_local=test_meta["hour_local"].values,
    model_name="3. Official EIA Benchmark"
))

# 4. Our Ridge Regression Model
results.append(evaluate_series(
    actual=test_meta[TARGET_COL].values,
    pred=test_meta["pred_ridge"].values,
    hour_local=test_meta["hour_local"].values,
    model_name="4. Ridge Regression (ML)"
))

df_scorecard = pd.DataFrame(results)

print("=========================================================================================")
print("  MODEL COMPARISON SCORECARD — HOLDOUT TEST SET (2024-H2)")
print("=========================================================================================")
display(df_scorecard)

eia_mae = df_scorecard.loc[df_scorecard["Model"].str.contains("EIA"), "MAE (MW)"].values[0]
ridge_mae = df_scorecard.loc[df_scorecard["Model"].str.contains("Ridge"), "MAE (MW)"].values[0]
lift_pct = ((eia_mae - ridge_mae) / eia_mae) * 100.0

print(f"\n--- Performance Summary ---")
print(f"EIA Benchmark MAE:     {eia_mae:,.1f} MW")
print(f"Ridge Regression MAE:  {ridge_mae:,.1f} MW")
if lift_pct > 0:
    print(f"Result: Ridge BEATS EIA benchmark by {lift_pct:.2f}% (Reduced average miss by {eia_mae - ridge_mae:,.1f} MW)")
else:
    print(f"Result: Ridge is within {abs(lift_pct):.2f}% of EIA benchmark (Gap: {ridge_mae - eia_mae:,.1f} MW)")

# COMMAND ----------
# DBTITLE 1,Feature Importance: Physical Driver Interpretation
# Scaled coefficients indicate relative impact per 1 standard deviation change
scaler = pipeline.named_steps["scaler"]
ridge_model = pipeline.named_steps["ridge"]

coef_df = pd.DataFrame({
    "Feature": FEATURE_COLS,
    "Standardized_Coefficient": ridge_model.coef_,
    "Absolute_Impact": np.abs(ridge_model.coef_)
}).sort_values("Absolute_Impact", ascending=False).reset_index(drop=True)

print("\n--- Top Physical & Temporal Drivers in Ridge Model ---")
display(coef_df)

# COMMAND ----------
# DBTITLE 1,Log Run & Metrics to MLflow
mlflow.set_experiment("/Shared/regional_demand_forecasting")

with mlflow.start_run(run_name="Ridge_Regression_H24"):
    # Log parameters
    mlflow.log_param("model_family", "Linear_Ridge")
    mlflow.log_param("best_alpha", float(best_alpha))
    mlflow.log_param("feature_count", len(FEATURE_COLS))
    mlflow.log_param("train_samples", len(X_train_val))
    mlflow.log_param("test_samples", len(X_test))
    
    # Log metrics
    ridge_metrics = next(r for r in results if "Ridge" in r["Model"])
    mlflow.log_metric("test_mae", ridge_metrics["MAE (MW)"])
    mlflow.log_metric("test_rmse", ridge_metrics["RMSE (MW)"])
    mlflow.log_metric("test_mape", ridge_metrics["MAPE (%)"])
    mlflow.log_metric("test_bias", ridge_metrics["Bias (MW)"])
    mlflow.log_metric("test_p90_error", ridge_metrics["P90 Error (MW)"])
    mlflow.log_metric("test_peak_mae", ridge_metrics["Peak MAE (MW)"])
    
    # Log model
    mlflow.sklearn.log_model(pipeline, "ridge_model")
    print("Successfully logged run and metrics to MLflow!")
