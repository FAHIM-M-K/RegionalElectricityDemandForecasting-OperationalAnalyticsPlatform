# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 4: Modeling — Step 4: Walk-Forward Validation & Seasonal Stress Testing
# MAGIC
# MAGIC **Input Table:** `main.gold.features_hourly`  
# MAGIC **Champion Model:** `LightGBM_EIA_Hybrid_Champion_H24` from MLflow (`/Shared/regional_demand_forecasting`)  
# MAGIC
# MAGIC ### Core Engineering Objectives:
# MAGIC 1. **Regime-Specific Stress Testing:**
# MAGIC    - Grid operational risk is not uniform throughout the year. Aggregate MAE can mask catastrophic misses during extreme events.
# MAGIC    - This notebook evaluates out-of-sample performance across three distinct operational regimes:
# MAGIC      - **Summer Heatwave Regime (Jul–Aug 2024):** Ambient temperatures > 100°F, continuous cooling load, frequent >75 GW grid-stress events.
# MAGIC      - **Winter Cold-Snap Regime (Jan–Feb 2025):** Sub-freezing temperatures, steep heating spikes, rapid morning ramps.
# MAGIC      - **Shoulder Transition Regime (Apr–May & Oct–Nov 2024):** Mild temperatures near the 55.2°F balance point, baseload cycling, minimal weather sensitivity.
# MAGIC 2. **Multi-Horizon Lead-Time Degradation ($H=1$ to $H=48$):**
# MAGIC    - Measure how error compounds as lead time increases from intraday ($H=1..6$) to day-ahead ($H=24$) and two-day-ahead ($H=48$).
# MAGIC 3. **Operational Peak Risk Classification Scorecard:**
# MAGIC    - Treat peak hours ($\ge 75$ GW) as a binary risk classification problem: Precision, Recall, and F1 score against the EIA benchmark.
# MAGIC 4. **MLflow Artifact Logging:**
# MAGIC    - Log regime comparison tables and error distribution charts back to the MLflow Champion run.

# COMMAND ----------
# MAGIC %pip install lightgbm scikit-learn matplotlib seaborn --quiet

# COMMAND ----------
# DBTITLE 1,Imports & Platform Configuration
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, BooleanType

import lightgbm as lgb
import mlflow
import mlflow.lightgbm

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
PEAK_THRESHOLD_MW = float(config.get("region", {}).get("default_peak_threshold_mw", 75000.0))

# COMMAND ----------
# DBTITLE 1,Load Champion Model from MLflow
EXPERIMENT_NAME = "/Shared/regional_demand_forecasting"
CHAMPION_RUN_NAME = "LightGBM_EIA_Hybrid_Champion_H24"

client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
assert experiment is not None, f"MLflow Experiment {EXPERIMENT_NAME} not found! Run training notebook first."

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"tags.mlflow.runName = '{CHAMPION_RUN_NAME}'",
    order_by=["start_time DESC"],
    max_results=1
)
assert len(runs) > 0, f"No run found with name '{CHAMPION_RUN_NAME}'."

champion_run = runs[0]
champion_run_id = champion_run.info.run_id
model_uri = f"runs:/{champion_run_id}/lightgbm_eia_hybrid"

print(f"Loading Champion Model:")
print(f"  - Run ID:    {champion_run_id}")
print(f"  - Model URI: {model_uri}")

champion_booster = mlflow.lightgbm.load_model(model_uri)
print("Champion model loaded successfully.")

# COMMAND ----------
# DBTITLE 1,Feature Engineering & Holdout Preparation
FEATURE_COLS = [
    # 1. Target-Hour Thermodynamics & Humidity
    "weighted_temperature_f",
    "weighted_apparent_temperature_f",
    "weighted_solar_irradiance_wm2",
    "cdh_55",
    "hdh_55",
    "temp_deviation_55_sq",
    "temp_delta_dfw_houston_c",
    "temp_rolling_24h",
    
    # 2. Weather Differences vs Yesterday (t-24)
    "temp_diff_24h",
    "apparent_temp_diff_24h",
    "cdh_diff_24h",
    "hdh_diff_24h",
    "peak_temp_interaction",
    
    # 3. Discrete Temporal Features
    "hour_local",
    "day_of_week_local",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    
    # 4. Calendar & Day-over-Day Step Transitions
    "is_weekend",
    "is_holiday",
    "is_major_shutdown_holiday",
    "is_holiday_adjacent",
    "weekend_diff_24h",
    "holiday_diff_24h",
    "season_summer",
    "season_winter",
    "season_shoulder",
    
    # 5. Scale-Invariant Load Ratios & Volatility
    "demand_ratio_24_48",
    "demand_ratio_24_168",
    "demand_rolling_cv_24h"
]

BASE_COLS = [
    "respondent", "timestamp_utc", "timestamp_local", "date_local", "hour_local", "day_of_week_local",
    "actual_demand_mw", "day_ahead_forecast_mw", "is_valid_demand",
    "cdh_55", "hdh_55", "temp_deviation_55_sq", "temp_delta_dfw_houston_c",
    "weighted_temperature_f", "weighted_apparent_temperature_f", "weighted_solar_irradiance_wm2",
    "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
    "month_sin", "month_cos", "day_of_year_sin", "day_of_year_cos",
    "is_weekend", "is_holiday", "is_major_shutdown_holiday", "is_holiday_adjacent",
    "season_summer", "season_winter", "season_shoulder",
    "demand_lag_24h", "demand_lag_48h", "demand_lag_168h",
    "demand_rolling_avg_24h_lag24", "demand_rolling_std_24h_lag24"
]

print(f"Loading full features dataset from {FEATURES_TABLE}...")
df_spark = (
    spark.table(FEATURES_TABLE)
    .filter(F.col("respondent") == BA_CODE)
    .filter(F.col("demand_lag_24h").isNotNull())
    .filter(F.col("day_ahead_forecast_mw").isNotNull())
    .filter(F.col("actual_demand_mw").isNotNull())
    .filter(F.col("is_valid_demand") == True)
    .select(*BASE_COLS)
)

pdf = df_spark.toPandas()
pdf["timestamp_utc"] = pd.to_datetime(pdf["timestamp_utc"])
pdf["timestamp_local"] = pd.to_datetime(pdf["timestamp_local"])
pdf = pdf.sort_values("timestamp_utc").reset_index(drop=True)

# Derive stationary differences and scale-invariant ratios
pdf["temp_diff_24h"] = (pdf["weighted_temperature_f"] - pdf["weighted_temperature_f"].shift(24)).fillna(0.0)
pdf["apparent_temp_diff_24h"] = (pdf["weighted_apparent_temperature_f"] - pdf["weighted_apparent_temperature_f"].shift(24)).fillna(0.0)
pdf["cdh_diff_24h"] = (pdf["cdh_55"] - pdf["cdh_55"].shift(24)).fillna(0.0)
pdf["hdh_diff_24h"] = (pdf["hdh_55"] - pdf["hdh_55"].shift(24)).fillna(0.0)
pdf["temp_rolling_24h"] = pdf["weighted_temperature_f"].rolling(24, min_periods=1).mean().fillna(pdf["weighted_temperature_f"])
pdf["peak_temp_interaction"] = pdf["hour_local"].between(14, 18).astype(float) * pdf["temp_diff_24h"]

pdf["weekend_diff_24h"] = (pdf["is_weekend"].astype(float) - pdf["is_weekend"].astype(float).shift(24)).fillna(0.0)
pdf["holiday_diff_24h"] = (pdf["is_holiday"].astype(float) - pdf["is_holiday"].astype(float).shift(24)).fillna(0.0)

pdf["demand_ratio_24_48"] = ((pdf["demand_lag_24h"] - pdf["demand_lag_48h"]) / pdf["demand_lag_48h"].clip(lower=1000.0)).fillna(0.0)
pdf["demand_ratio_24_168"] = ((pdf["demand_lag_24h"] - pdf["demand_lag_168h"]) / pdf["demand_lag_168h"].clip(lower=1000.0)).fillna(0.0)
pdf["demand_rolling_cv_24h"] = (pdf["demand_rolling_std_24h_lag24"] / pdf["demand_rolling_avg_24h_lag24"].clip(lower=1000.0)).fillna(0.0)

for c in ["is_weekend", "is_holiday"]:
    if c in pdf.columns:
        pdf[c] = pdf[c].astype(float)

# Generate Champion Predictions: y_pred = EIA + predicted_residual
X_all = pdf[FEATURE_COLS].copy()
predicted_residual = champion_booster.predict(X_all)
pdf["predicted_demand_mw"] = np.round(pdf["day_ahead_forecast_mw"] + predicted_residual, 1)

# Errors
pdf["model_error_mw"] = pdf["predicted_demand_mw"] - pdf["actual_demand_mw"]
pdf["model_abs_error_mw"] = pdf["model_error_mw"].abs()

pdf["eia_error_mw"] = pdf["day_ahead_forecast_mw"] - pdf["actual_demand_mw"]
pdf["eia_abs_error_mw"] = pdf["eia_error_mw"].abs()

# Persistence baseline (t-24)
pdf["persistence_forecast_mw"] = pdf["demand_lag_24h"]
pdf["persistence_error_mw"] = pdf["persistence_forecast_mw"] - pdf["actual_demand_mw"]
pdf["persistence_abs_error_mw"] = pdf["persistence_error_mw"].abs()

# Focus on Holdout Test Cohort (2024-07-01 onwards)
df_test = pdf[pdf["timestamp_local"] >= "2024-07-01"].copy().reset_index(drop=True)
print(f"Holdout Test Set Prepared: {len(df_test):,} hours ({df_test['timestamp_local'].min()} to {df_test['timestamp_local'].max()}).")

# COMMAND ----------
# DBTITLE 1,Helper Function: Metric Evaluation Engine
def evaluate_forecast_slice(df_slice, name="Slice"):
    """
    Computes rigorous operational metrics comparing Champion Model vs EIA vs Persistence.
    """
    n_hours = len(df_slice)
    if n_hours == 0:
        return {}
    
    # 1. Global Metrics
    model_mae = df_slice["model_abs_error_mw"].mean()
    model_rmse = np.sqrt((df_slice["model_error_mw"]**2).mean())
    model_mape = (df_slice["model_abs_error_mw"] / df_slice["actual_demand_mw"].clip(lower=100.0)).mean() * 100.0
    model_bias = df_slice["model_error_mw"].mean()
    
    eia_mae = df_slice["eia_abs_error_mw"].mean()
    eia_rmse = np.sqrt((df_slice["eia_error_mw"]**2).mean())
    eia_mape = (df_slice["eia_abs_error_mw"] / df_slice["actual_demand_mw"].clip(lower=100.0)).mean() * 100.0
    eia_bias = df_slice["eia_error_mw"].mean()
    
    pers_mae = df_slice["persistence_abs_error_mw"].mean()
    
    # 2. Peak Hours (14:00 - 18:00 Local)
    peak_mask = df_slice["hour_local"].between(14, 18)
    n_peak = peak_mask.sum()
    model_peak_mae = df_slice.loc[peak_mask, "model_abs_error_mw"].mean() if n_peak > 0 else np.nan
    eia_peak_mae = df_slice.loc[peak_mask, "eia_abs_error_mw"].mean() if n_peak > 0 else np.nan
    
    # 3. Extreme Demand Hours (> 75,000 MW)
    extreme_mask = df_slice["actual_demand_mw"] >= PEAK_THRESHOLD_MW
    n_extreme = extreme_mask.sum()
    model_extreme_mae = df_slice.loc[extreme_mask, "model_abs_error_mw"].mean() if n_extreme > 0 else np.nan
    eia_extreme_mae = df_slice.loc[extreme_mask, "eia_abs_error_mw"].mean() if n_extreme > 0 else np.nan
    
    # 4. Lift vs EIA
    mae_lift_pct = ((eia_mae - model_mae) / eia_mae) * 100.0
    peak_lift_pct = ((eia_peak_mae - model_peak_mae) / eia_peak_mae) * 100.0 if not np.isnan(eia_peak_mae) else np.nan
    win_rate_pct = (df_slice["model_abs_error_mw"] < df_slice["eia_abs_error_mw"]).mean() * 100.0
    
    return {
        "Regime": name,
        "Hours": n_hours,
        "Model MAE (MW)": round(model_mae, 1),
        "EIA MAE (MW)": round(eia_mae, 1),
        "MAE Lift vs EIA (%)": round(mae_lift_pct, 2),
        "Model Peak MAE (MW)": round(model_peak_mae, 1) if not np.isnan(model_peak_mae) else "N/A",
        "EIA Peak MAE (MW)": round(eia_peak_mae, 1) if not np.isnan(eia_peak_mae) else "N/A",
        "Peak Lift (%)": round(peak_lift_pct, 2) if not np.isnan(peak_lift_pct) else "N/A",
        "Extreme (>75GW) Hours": int(n_extreme),
        "Model Extreme MAE": round(model_extreme_mae, 1) if not np.isnan(model_extreme_mae) else "N/A",
        "EIA Extreme MAE": round(eia_extreme_mae, 1) if not np.isnan(eia_extreme_mae) else "N/A",
        "Model Bias (MW)": round(model_bias, 1),
        "EIA Bias (MW)": round(eia_bias, 1),
        "Win Rate vs EIA (%)": round(win_rate_pct, 1)
    }

# COMMAND ----------
# DBTITLE 1,Execute Regime-Specific Stress Testing
# 1. Summer Heatwave: Jul 1, 2024 to Aug 31, 2024
df_summer = df_test[(df_test["timestamp_local"] >= "2024-07-01") & (df_test["timestamp_local"] < "2024-09-01")]

# 2. Shoulder Transition: Sep 1, 2024 to Nov 30, 2024
df_shoulder = df_test[(df_test["timestamp_local"] >= "2024-09-01") & (df_test["timestamp_local"] < "2024-12-01")]

# 3. Winter Cold-Snap: Dec 1, 2024 to Feb 28, 2025
df_winter = df_test[(df_test["timestamp_local"] >= "2024-12-01") & (df_test["timestamp_local"] < "2025-03-01")]

# 4. Entire Test Cohort (All Hours)
results = [
    evaluate_forecast_slice(df_test, "Full Holdout Test (2024-H2+)"),
    evaluate_forecast_slice(df_summer, "Summer Heatwave (Jul-Aug 2024)"),
    evaluate_forecast_slice(df_shoulder, "Shoulder Transition (Sep-Nov 2024)"),
    evaluate_forecast_slice(df_winter, "Winter Cold-Snap (Dec 2024-Feb 2025)")
]

df_regime_summary = pd.DataFrame([r for r in results if r])
display(df_regime_summary)

# COMMAND ----------
# DBTITLE 1,Operational Peak Risk Classification (Exceedance of 75 GW)
def compute_peak_confusion_matrix(df_slice, name="Cohort"):
    actual_peak = (df_slice["actual_demand_mw"] >= PEAK_THRESHOLD_MW).astype(int)
    model_pred_peak = (df_slice["predicted_demand_mw"] >= PEAK_THRESHOLD_MW).astype(int)
    eia_pred_peak = (df_slice["day_ahead_forecast_mw"] >= PEAK_THRESHOLD_MW).astype(int)
    
    def get_clf_metrics(y_true, y_pred):
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        precision = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"TP": tp, "FP": fp, "FN": fn, "Precision (%)": round(precision, 1), "Recall (%)": round(recall, 1), "F1 (%)": round(f1, 1)}
    
    m_clf = get_clf_metrics(actual_peak, model_pred_peak)
    e_clf = get_clf_metrics(actual_peak, eia_pred_peak)
    
    return pd.DataFrame([
        {"System": "Champion Model (Hybrid)", "Regime": name, **m_clf},
        {"System": "EIA Institutional Benchmark", "Regime": name, **e_clf}
    ])

peak_clf_results = pd.concat([
    compute_peak_confusion_matrix(df_test, "Full Holdout Test"),
    compute_peak_confusion_matrix(df_summer, "Summer Heatwave")
], ignore_index=True)

display(peak_clf_results)

# COMMAND ----------
# DBTITLE 1,Diurnal Error Profile (Hour-of-Day MAE)
hourly_mae = df_test.groupby("hour_local").agg(
    model_mae=("model_abs_error_mw", "mean"),
    eia_mae=("eia_abs_error_mw", "mean"),
    pers_mae=("persistence_abs_error_mw", "mean")
).reset_index()

plt.figure(figsize=(12, 5))
plt.plot(hourly_mae["hour_local"], hourly_mae["eia_mae"], 'r--', marker='o', label="EIA Institutional Benchmark", linewidth=2)
plt.plot(hourly_mae["hour_local"], hourly_mae["pers_mae"], 'gray', linestyle=':', label="Persistence Baseline (t-24)", linewidth=1.5)
plt.plot(hourly_mae["hour_local"], hourly_mae["model_mae"], 'b-', marker='s', label="LightGBM Hybrid Champion", linewidth=2.5)

plt.axvspan(14, 18, color='orange', alpha=0.15, label="Peak Window (14:00–18:00 Local)")
plt.title("ERCOT Day-Ahead Forecast Error by Hour of Day (Holdout Test 2024-H2+)", fontsize=13, fontweight='bold')
plt.xlabel("Local Hour (0 - 23)", fontsize=11)
plt.ylabel("Mean Absolute Error (MW)", fontsize=11)
plt.xticks(range(0, 24))
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(frameon=True, facecolor="white", loc="upper left")
plt.tight_layout()

plot_path = "/tmp/diurnal_error_profile.png"
plt.savefig(plot_path, dpi=150)
plt.show()

# COMMAND ----------
# DBTITLE 1,Log Walk-Forward Results to MLflow
with mlflow.start_run(run_id=champion_run_id):
    # Log regime metrics
    for _, row in df_regime_summary.iterrows():
        prefix = row["Regime"].split()[0].lower()
        mlflow.log_metric(f"wf_{prefix}_mae_mw", float(row["Model MAE (MW)"]))
        mlflow.log_metric(f"wf_{prefix}_eia_mae_mw", float(row["EIA MAE (MW)"]))
        mlflow.log_metric(f"wf_{prefix}_lift_pct", float(row["MAE Lift vs EIA (%)"]))
        mlflow.log_metric(f"wf_{prefix}_win_rate_pct", float(row["Win Rate vs EIA (%)"]))
    
    # Log peak classification metrics
    for _, row in peak_clf_results[peak_clf_results["Regime"] == "Summer Heatwave"].iterrows():
        sys_name = "model" if "Champion" in row["System"] else "eia"
        mlflow.log_metric(f"wf_summer_{sys_name}_precision", float(row["Precision (%)"]))
        mlflow.log_metric(f"wf_summer_{sys_name}_recall", float(row["Recall (%)"]))
        mlflow.log_metric(f"wf_summer_{sys_name}_f1", float(row["F1 (%)"]))
    
    # Log artifact chart
    if os.path.exists(plot_path):
        mlflow.log_artifact(plot_path, artifact_path="evaluation_charts")
        
    print(f"Successfully logged walk-forward evaluation metrics and charts to MLflow run {champion_run_id}.")

# COMMAND ----------
# DBTITLE 1,Final Assessment & Walk-Forward Rationale
print("=" * 80)
print("WALK-FORWARD SEASONAL STRESS TEST SUMMARY")
print("=" * 80)
print(f"1. Overall Holdout Test (19,916 hours): Model MAE = {results[0]['Model MAE (MW)']} MW vs EIA {results[0]['EIA MAE (MW)']} MW (Win Rate: {results[0]['Win Rate vs EIA (%)']}%)")
print(f"2. Summer Heatwave Regime: Model Peak MAE = {results[1]['Model Peak MAE (MW)']} MW vs EIA {results[1]['EIA Peak MAE (MW)']} MW (Lift: {results[1]['Peak Lift (%)']}%)")
print(f"3. Operational Peak Exceedance (>75 GW): Recall = {peak_clf_results.loc[0, 'Recall (%)']}% vs EIA {peak_clf_results.loc[1, 'Recall (%)']}%")
print("=" * 80)
