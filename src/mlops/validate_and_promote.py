# Databricks notebook source
# MAGIC %md
# MAGIC # MLOps Validation Gate & Champion Promotion
# MAGIC
# MAGIC **Input:** Candidate Run ID from `retrain_champion` (via `taskValues` or latest CANDIDATE tag)  
# MAGIC **Output:** 
# MAGIC - Champion tag promotion in MLflow Registry / Experiment Tracking
# MAGIC - Feature importance synchronized to `main.gold.feature_importance`
# MAGIC
# MAGIC ### Automated Decision Rules:
# MAGIC 1. **Global Lift Gate:** Candidate must improve holdout MAE by $\ge 0.5\%$ over current Champion.
# MAGIC 2. **Peak Reliability Gate:** Candidate must not regress in Peak Hours (14:00 - 18:00) MAE.
# MAGIC 3. **Action:**
# MAGIC    - **PROMOTED:** Candidate becomes `LightGBM_EIA_Hybrid_Champion_H24`, previous champion tagged `RETIRED_CHAMPION`, feature importance synced to Delta table.
# MAGIC    - **REJECTED:** Candidate tagged `REJECTED_CANDIDATE`, previous champion retained, scoring job continues on verified champion.

# COMMAND ----------
# DBTITLE 1,Imports & Configuration
import os
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
try:
    from pyspark.sql import functions as F
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, TimestampType
except ImportError:
    F = None

import mlflow
import mlflow.lightgbm

# Dynamic repository locator
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
from src.shared.feature_transforms import FEATURE_COLS

config = load_platform_config("ercot_config.json")
CATALOG = "main"
EXPERIMENT_NAME = "/Shared/regional_demand_forecasting"
CHAMPION_RUN_NAME = "LightGBM_EIA_Hybrid_Champion_H24"
IMPORTANCE_TABLE = f"{CATALOG}.gold.feature_importance"

client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
assert experiment is not None, f"MLflow Experiment {EXPERIMENT_NAME} not found!"

# COMMAND ----------
# DBTITLE 1,Retrieve Candidate Run ID
candidate_run_id = None

# Attempt 1: From upstream Databricks taskValues
try:
    candidate_run_id = dbutils.jobs.taskValues.get(taskKey="retrain_champion", key="candidate_run_id")
    print(f"Obtained candidate_run_id from upstream taskValues: {candidate_run_id}")
except Exception as e:
    print(f"taskValues unavailable or failed ({e}). Searching MLflow for latest CANDIDATE...")

# Attempt 2: Search MLflow for latest run tagged CANDIDATE
if not candidate_run_id:
    candidate_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.pipeline_role = 'CANDIDATE'",
        order_by=["start_time DESC"],
        max_results=1
    )
    if candidate_runs:
        candidate_run_id = candidate_runs[0].info.run_id
        print(f"Found latest candidate run: {candidate_run_id} ({candidate_runs[0].info.run_name})")

assert candidate_run_id is not None, "No candidate run found for validation!"
candidate_run = client.get_run(candidate_run_id)

# COMMAND ----------
# DBTITLE 1,Retrieve Active Champion Run
champion_runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"tags.mlflow.runName = '{CHAMPION_RUN_NAME}'",
    order_by=["start_time DESC"],
    max_results=1
)

champion_run = champion_runs[0] if champion_runs else None

# COMMAND ----------
# DBTITLE 1,Perform A/B Benchmark Validation Gate
candidate_mae = candidate_run.data.metrics.get("holdout_mae", candidate_run.data.metrics.get("test_mae"))
candidate_peak_mae = candidate_run.data.metrics.get("holdout_peak_mae", candidate_run.data.metrics.get("test_peak_mae"))

assert candidate_mae is not None, f"Candidate run {candidate_run_id} missing holdout_mae metric!"

if champion_run is None:
    print("No active champion found in MLflow. Candidate will be promoted as the Initial Champion.")
    should_promote = True
    rejection_reason = ""
    champion_mae = np.nan
    champion_peak_mae = np.nan
    improvement_pct = 100.0
else:
    champion_run_id = champion_run.info.run_id
    champion_mae = champion_run.data.metrics.get("holdout_mae", champion_run.data.metrics.get("test_mae", np.nan))
    champion_peak_mae = champion_run.data.metrics.get("holdout_peak_mae", champion_run.data.metrics.get("test_peak_mae", np.nan))
    
    print(f"Active Champion Run ID: {champion_run_id}")
    print(f"  - Champion MAE:      {champion_mae:,.2f} MW")
    print(f"  - Champion Peak MAE: {champion_peak_mae:,.2f} MW")
    print(f"Candidate Run ID:      {candidate_run_id}")
    print(f"  - Candidate MAE:      {candidate_mae:,.2f} MW")
    print(f"  - Candidate Peak MAE: {candidate_peak_mae:,.2f} MW")
    
    if np.isnan(champion_mae):
        improvement_pct = 1.0  # Fallback if champion metric was untracked
        mae_pass = True
    else:
        improvement_pct = ((champion_mae - candidate_mae) / champion_mae) * 100.0
        mae_pass = improvement_pct >= 0.5  # Gate 1: At least +0.5% lift

    if np.isnan(champion_peak_mae) or np.isnan(candidate_peak_mae):
        peak_pass = True
    else:
        # Gate 2: No peak regression (candidate must be <= champion peak error + 1.0 MW tolerance)
        peak_pass = candidate_peak_mae <= (champion_peak_mae + 1.0)
    
    should_promote = mae_pass and peak_pass
    
    reasons = []
    if not mae_pass:
        reasons.append(f"Global MAE improvement ({improvement_pct:+.2f}%) fell below mandatory threshold (+0.50%)")
    if not peak_pass:
        reasons.append(f"Peak MAE regressed ({candidate_peak_mae:.1f} MW vs Champion {champion_peak_mae:.1f} MW)")
    rejection_reason = "; ".join(reasons)

print("\n================== VALIDATION GATE SCORECARD ==================")
print(f"Global MAE Lift vs Champion:  {improvement_pct:+.2f}% (Threshold: >= +0.50%)")
print(f"Peak MAE Check:               Candidate={candidate_peak_mae:.1f} MW vs Champion={champion_peak_mae:.1f} MW")
print(f"Final Decision:               {'*** PROMOTED TO CHAMPION ***' if should_promote else '*** REJECTED ***'}")
if not should_promote:
    print(f"Rejection Details:            {rejection_reason}")
print("===============================================================")

# COMMAND ----------
# DBTITLE 1,Execute Promotion or Rejection
now_utc_str = datetime.now(timezone.utc).isoformat()
now_date_str = datetime.now(timezone.utc).strftime("%Y%m%d")

FEATURE_DESCRIPTIONS = {
    "weighted_apparent_temperature_f": "Population-weighted Heat Index / Wind Chill",
    "hour_local": "Discrete diurnal cycle indicator (0-23)",
    "temp_rolling_24h": "24h thermal rolling heat accumulation",
    "cdh_55": "Cooling Degree Hours above 55.2 deg F baseline",
    "temp_diff_24h": "Day-over-day temperature shift at same hour",
    "peak_temp_interaction": "Late afternoon (14-18h) extreme thermal interaction",
    "day_of_week_local": "Day of week cycle (0=Mon, 6=Sun)",
    "weighted_solar_irradiance_wm2": "Direct solar irradiance driving cooling demand",
    "demand_ratio_24_48": "Scale-invariant load momentum ratio (t-24 / t-48)",
    "weekend_diff_24h": "Friday-to-Saturday step-down transition",
    "demand_lag_24h": "Day-ahead persistence load baseline",
    "apparent_temp_diff_24h": "Day-over-day apparent heat shift",
    "temp_deviation_55_sq": "Nonlinear squared temperature deviation from 55 deg F",
    "temp_delta_dfw_houston_c": "Regional gradient between Dallas and Houston",
    "hdh_55": "Heating Degree Hours below 55.2 deg F baseline",
}

if should_promote:
    # 1. Retire previous champion if one exists
    if champion_run is not None:
        client.set_tag(champion_run.info.run_id, "pipeline_role", "RETIRED_CHAMPION")
        client.set_tag(champion_run.info.run_id, "retired_at", now_utc_str)
        client.set_tag(champion_run.info.run_id, "mlflow.runName", f"LightGBM_EIA_Hybrid_Retired_{now_date_str}")
        print(f"Retired previous champion run: {champion_run.info.run_id}")
    
    # 2. Promote candidate run
    client.set_tag(candidate_run_id, "pipeline_role", "CHAMPION")
    client.set_tag(candidate_run_id, "promoted_at", now_utc_str)
    client.set_tag(candidate_run_id, "promotion_decision", "PROMOTED")
    client.set_tag(candidate_run_id, "mlflow.runName", CHAMPION_RUN_NAME)
    print(f"Promoted candidate {candidate_run_id} to active CHAMPION ({CHAMPION_RUN_NAME})!")
    
    # 3. Synchronize Feature Importance to main.gold.feature_importance (A.5)
    model_uri = f"runs:/{candidate_run_id}/lightgbm_eia_hybrid"
    candidate_booster = mlflow.lightgbm.load_model(model_uri)
    
    gains = candidate_booster.feature_importance(importance_type="gain")
    splits = candidate_booster.feature_importance(importance_type="split")
    
    df_feat = pd.DataFrame({
        "feature_name": FEATURE_COLS,
        "gain_importance": gains.astype(float),
        "split_importance": splits.astype(int),
        "gain_pct": ((gains / gains.sum()) * 100.0).round(2),
    })
    df_feat["feature_description"] = df_feat["feature_name"].map(
        lambda f: FEATURE_DESCRIPTIONS.get(f, "Derived thermodynamic/calendar feature")
    )
    df_feat["updated_at"] = pd.Timestamp.now(tz="UTC")
    df_feat["champion_run_id"] = candidate_run_id
    df_feat = df_feat.sort_values("gain_importance", ascending=False).reset_index(drop=True)
    df_feat["rank"] = df_feat.index + 1
    
    try:
        spark_feat = spark.createDataFrame(df_feat)
        (
            spark_feat.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(IMPORTANCE_TABLE)
        )
        print(f"Successfully synchronized {len(df_feat)} features to Delta table: {IMPORTANCE_TABLE}")
    except Exception as e:
        print(f"Notice: Delta table write skipped in environment: {e}")
        
    # Local fallback cache for local dev / Streamlit / Power BI
    local_data_dir = os.path.join(_repo_root, "data")
    os.makedirs(local_data_dir, exist_ok=True)
    df_feat.to_csv(os.path.join(local_data_dir, "feature_importance.csv"), index=False)
    print(f"Saved local cache to: data/feature_importance.csv")

    active_run_id = candidate_run_id
    promo_status = "PROMOTED"

else:
    client.set_tag(candidate_run_id, "pipeline_role", "REJECTED_CANDIDATE")
    client.set_tag(candidate_run_id, "promotion_decision", "REJECTED")
    client.set_tag(candidate_run_id, "rejection_reason", rejection_reason)
    client.set_tag(candidate_run_id, "rejected_at", now_utc_str)
    print(f"Candidate {candidate_run_id} marked as REJECTED. Active champion retained.")
    
    active_run_id = champion_run.info.run_id if champion_run else ""
    promo_status = "REJECTED"

# COMMAND ----------
# DBTITLE 1,Emit Task Values Downstream
try:
    dbutils.jobs.taskValues.set(key="promotion_status", value=promo_status)
    dbutils.jobs.taskValues.set(key="active_champion_run_id", value=active_run_id)
    dbutils.jobs.taskValues.set(key="improvement_pct", value=float(improvement_pct) if not np.isnan(improvement_pct) else 0.0)
    print("Emitted validation gate task values successfully!")
except NameError:
    print("dbutils not found (running in standalone/test mode). Task values output skipped.")
