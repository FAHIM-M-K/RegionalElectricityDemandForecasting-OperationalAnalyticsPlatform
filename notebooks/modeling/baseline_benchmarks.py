# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 4: Modeling — Step 1: Standard Evaluation Protocol & Baselines
# MAGIC
# MAGIC **Input Table:** `main.gold.features_hourly`  
# MAGIC
# MAGIC ### Core Objectives:
# MAGIC 1. **Establish the Strict Time-Based Split:**
# MAGIC    - **Train:** `2021-03-01` to `2023-12-31` (robust multi-year historical depth, excluding Uri blackout & contaminated lookback).
# MAGIC    - **Validation:** `2024-01-01` to `2024-06-30` (model selection & hyperparameter tuning).
# MAGIC    - **Test:** `2024-07-01` to `2024-12-31` (out-of-sample holdout covering peak summer heatwaves and winter transition).
# MAGIC 2. **Benchmark Ground-Truth Baselines (Must-Beat Criteria):**
# MAGIC    - **Baseline 1: Persistence ($t-24$):** Yesterday's demand at the same hour (`demand_lag_24h`).
# MAGIC    - **Baseline 2: Seasonal Naive ($t-168$):** Last week's demand at the same hour (`demand_lag_168h`).
# MAGIC    - **Baseline 3: Official EIA Day-Ahead Forecast:** Institutional grid operator forecast (`day_ahead_forecast_mw`).
# MAGIC 3. **Standardized Metrics Evaluated:**
# MAGIC    - **MAE (Mean Absolute Error, MW):** Primary operational accuracy KPI.
# MAGIC    - **RMSE (Root Mean Squared Error, MW):** High-loss penalty for severe operational surprises.
# MAGIC    - **MAPE (%):** Mean Absolute Percentage Error.
# MAGIC    - **Bias / Mean Signed Error (MW):** Systematic over/under-forecasting.
# MAGIC    - **Peak Window MAE (14:00–18:00 Local):** Performance during critical grid stress hours.
# MAGIC    - **Extreme Load MAE (> 75 GW):** Operational stress regime performance.

# COMMAND ----------
# DBTITLE 1,Imports & Setup
import os
import sys
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

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
# DBTITLE 1,Load Feature Store & Apply Leakage-Safe Training Filter
assert spark.catalog.tableExists(FEATURES_TABLE), f"Features table {FEATURES_TABLE} does not exist!"

# Filter to valid Day-Ahead (H=24) eligible samples
# (Guarantees valid actual demand, valid lag_24h, lag_168h, and excludes Uri blackout contamination)
df_all = (
    spark.table(FEATURES_TABLE)
    .filter(F.col("respondent") == BA_CODE)
    .filter(F.col("is_valid_training_sample_h24") == True)
    .filter(F.col("day_ahead_forecast_mw").isNotNull())
)

total_eligible_hours = df_all.count()
print(f"Total leakage-safe, validated hours loaded: {total_eligible_hours:,}")

# COMMAND ----------
# DBTITLE 1,Define Strict Chronological Data Splits
# Split boundaries (UTC timestamps)
SPLIT_TRAIN_END = "2024-01-01 00:00:00"
SPLIT_VAL_END   = "2024-07-01 00:00:00"

df_train = df_all.filter(F.col("timestamp_utc") < SPLIT_TRAIN_END)
df_val   = df_all.filter((F.col("timestamp_utc") >= SPLIT_TRAIN_END) & (F.col("timestamp_utc") < SPLIT_VAL_END))
df_test  = df_all.filter(F.col("timestamp_utc") >= SPLIT_VAL_END)

n_train = df_train.count()
n_val   = df_val.count()
n_test  = df_test.count()

print("Time-Series Cohort Split (Chronological, Zero Shuffling):")
print(f"  - Train Cohort (2021-03 through 2023): {n_train:,} hours ({n_train / total_eligible_hours * 100:.1f}%)")
print(f"  - Validation Cohort (2024-H1):         {n_val:,} hours ({n_val / total_eligible_hours * 100:.1f}%)")
print(f"  - Holdout Test Cohort (2024-H2):       {n_test:,} hours ({n_test / total_eligible_hours * 100:.1f}%)")

assert n_test > 0, "Test cohort is empty! Check split timestamp boundaries."

# COMMAND ----------
# DBTITLE 1,Standard Evaluation Metrics Engine
def evaluate_forecast(df, actual_col="actual_demand_mw", pred_col="pred_demand_mw", model_name="Model"):
    """
    Computes standard operational accuracy metrics for an electricity demand forecast.
    """
    # Base error columns
    df_eval = (
        df.filter(F.col(actual_col).isNotNull() & F.col(pred_col).isNotNull())
        .withColumn("error_mw", F.col(pred_col) - F.col(actual_col))
        .withColumn("abs_error_mw", F.abs(F.col("error_mw")))
        .withColumn("sq_error_mw", F.pow(F.col("error_mw"), 2.0))
        .withColumn("ape_pct", (F.abs(F.col("error_mw")) / F.col(actual_col)) * 100.0)
    )
    
    # Global metrics
    global_stats = df_eval.select(
        F.count("*").alias("n"),
        F.avg("abs_error_mw").alias("mae"),
        F.sqrt(F.avg("sq_error_mw")).alias("rmse"),
        F.avg("ape_pct").alias("mape"),
        F.avg("error_mw").alias("bias"),
        F.expr("percentile_approx(abs_error_mw, 0.90)").alias("p90_error")
    ).first()
    
    # Peak-Window metrics (14:00 - 18:00 local time)
    peak_stats = (
        df_eval.filter(F.col("hour_local").between(14, 18))
        .select(
            F.avg("abs_error_mw").alias("peak_mae"),
            F.expr("percentile_approx(abs_error_mw, 0.90)").alias("peak_p90")
        )
        .first()
    )
    
    # Extreme Load metrics (> 75,000 MW high-stress conditions)
    extreme_stats = (
        df_eval.filter(F.col(actual_col) >= 75000.0)
        .select(
            F.count("*").alias("n_extreme"),
            F.avg("abs_error_mw").alias("extreme_mae")
        )
        .first()
    )
    extreme_mae = extreme_stats["extreme_mae"] if extreme_stats and extreme_stats["n_extreme"] > 0 else np.nan
    
    return {
        "Model": model_name,
        "Hours Evaluated": int(global_stats["n"]),
        "MAE (MW)": round(float(global_stats["mae"]), 1),
        "RMSE (MW)": round(float(global_stats["rmse"]), 1),
        "MAPE (%)": round(float(global_stats["mape"]), 2),
        "Bias (MW)": round(float(global_stats["bias"]), 1),
        "P90 Error (MW)": round(float(global_stats["p90_error"]), 1),
        "Peak Window MAE (MW)": round(float(peak_stats["peak_mae"]), 1),
        "Peak Window P90 (MW)": round(float(peak_stats["peak_p90"]), 1),
        "Extreme (>75GW) MAE (MW)": round(float(extreme_mae), 1) if not np.isnan(extreme_mae) else "N/A"
    }

# COMMAND ----------
# DBTITLE 1,Execute Baseline Evaluations on Holdout Test Cohort
baselines = [
    {
        "name": "1. Persistence (t-24)",
        "pred_col": "demand_lag_24h"
    },
    {
        "name": "2. Seasonal Naive (t-168)",
        "pred_col": "demand_lag_168h"
    },
    {
        "name": "3. Official EIA Day-Ahead Forecast",
        "pred_col": "day_ahead_forecast_mw"
    }
]

scorecard = []
for b in baselines:
    res = evaluate_forecast(
        df=df_test,
        actual_col="actual_demand_mw",
        pred_col=b["pred_col"],
        model_name=b["name"]
    )
    scorecard.append(res)

df_scorecard = pd.DataFrame(scorecard)

print("=========================================================================================")
print("  PHASE 4: HOLDOUT TEST SET BASELINE SCORECARD (2024-H2)")
print("=========================================================================================")
display(df_scorecard)

# COMMAND ----------
# DBTITLE 1,Export Baseline Summary Metrics
# Save scorecard summary as reference dictionary for ML model training notebooks
import json
output_summary = df_scorecard.to_dict(orient="records")

print("\nOfficial Baseline Targets to Beat in Step 2-4:")
eia_bench = next(m for m in output_summary if "EIA" in m["Model"])
print(f"  Target 1: Beat Global EIA MAE ({eia_bench['MAE (MW)']} MW) by >= 15% (Target: < {eia_bench['MAE (MW)'] * 0.85:.1f} MW)")
print(f"  Target 2: Beat Peak-Hour EIA MAE ({eia_bench['Peak Window MAE (MW)']} MW) by >= 15% (Target: < {eia_bench['Peak Window MAE (MW)'] * 0.85:.1f} MW)")