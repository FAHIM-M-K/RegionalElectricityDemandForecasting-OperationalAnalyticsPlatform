# Databricks notebook source
# MAGIC %md
# MAGIC # Gold Layer: Production Forecast Inference Engine (`gold.forecast_hourly`)
# MAGIC
# MAGIC **Input Table:** `main.gold.features_hourly`  
# MAGIC **Output Table:** `main.gold.forecast_hourly`  
# MAGIC **Model Source:** Databricks MLflow Registry / Experiment (`/Shared/regional_demand_forecasting`)  
# MAGIC
# MAGIC ### Core Production Pipeline Responsibilities:
# MAGIC 1. **Model Retrieval:** Dynamically loads the champion model artifact (`LightGBM_EIA_Hybrid_Champion_H24`) from MLflow.
# MAGIC 2. **Inference Execution:** Generates point forecasts (`predicted_demand_mw`) across the chronological evaluation grid.
# MAGIC 3. **Uncertainty & Risk Quantification:**
# MAGIC    - 95% Confidence Intervals (`prediction_lower_95_mw`, `prediction_upper_95_mw`).
# MAGIC    - Operational Peak Warning (`is_predicted_peak_risk` when $\hat{y} \ge 75,000$ MW).
# MAGIC 4. **Live Benchmarking:** Computes error metrics (`model_error_mw`, `model_abs_error_mw`) comparing our predictions vs. actuals alongside EIA Day-Ahead benchmark metrics.
# MAGIC 5. **Idempotent Delta MERGE:** Null-safe upsert ensuring atomic commits without Parquet file churn.

# COMMAND ----------
# MAGIC %pip install lightgbm --quiet

# COMMAND ----------
# DBTITLE 1,Imports & Setup
import os
import sys
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, BooleanType, TimestampType
from delta.tables import DeltaTable

import lightgbm as lgb
import mlflow
import mlflow.lightgbm

CATALOG = "main"
SOURCE_FEATURES_TABLE = f"{CATALOG}.gold.features_hourly"
TARGET_FORECAST_TABLE = f"{CATALOG}.gold.forecast_hourly"

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
# DBTITLE 1,Retrieve Champion Model from MLflow
EXPERIMENT_NAME = "/Shared/regional_demand_forecasting"
CHAMPION_RUN_NAME = "LightGBM_EIA_Hybrid_Champion_H24"

client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
assert experiment is not None, f"MLflow Experiment {EXPERIMENT_NAME} not found! Run modeling notebook first."

# Find latest successful run for the champion model
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"tags.mlflow.runName = '{CHAMPION_RUN_NAME}'",
    order_by=["start_time DESC"],
    max_results=1
)

assert len(runs) > 0, f"No logged runs found with name '{CHAMPION_RUN_NAME}' in {EXPERIMENT_NAME}."
champion_run = runs[0]
champion_run_id = champion_run.info.run_id
model_uri = f"runs:/{champion_run_id}/lightgbm_eia_hybrid"

print(f"Loading Champion Model from MLflow:")
print(f"  - Experiment: {EXPERIMENT_NAME}")
print(f"  - Run Name:   {CHAMPION_RUN_NAME}")
print(f"  - Run ID:     {champion_run_id}")
print(f"  - Model URI:  {model_uri}")

champion_booster = mlflow.lightgbm.load_model(model_uri)
print("Successfully loaded Champion LightGBM Booster!")

# COMMAND ----------
# DBTITLE 1,Define Feature Extraction Architecture
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

# COMMAND ----------
# DBTITLE 1,Execute Batch Scoring
def generate_forecast_predictions(df_features_spark, booster, feature_cols=FEATURE_COLS, base_cols=BASE_COLS):
    """
    Computes scale-invariant features and generates Day-Ahead model predictions.
    """
    # Pull candidate rows (requires valid 24h lag and day_ahead_forecast_mw)
    df_eligible = (
        df_features_spark
        .filter(F.col("respondent") == BA_CODE)
        .filter(F.col("demand_lag_24h").isNotNull())
        .filter(F.col("day_ahead_forecast_mw").isNotNull())
        .select(*base_cols)
    )
    
    pdf = df_eligible.toPandas()
    pdf["timestamp_utc"] = pd.to_datetime(pdf["timestamp_utc"])
    pdf = pdf.sort_values("timestamp_utc").reset_index(drop=True)
    
    # 1. Weather shifts from yesterday
    pdf["temp_diff_24h"] = (pdf["weighted_temperature_f"] - pdf["weighted_temperature_f"].shift(24)).fillna(0.0)
    pdf["apparent_temp_diff_24h"] = (pdf["weighted_apparent_temperature_f"] - pdf["weighted_apparent_temperature_f"].shift(24)).fillna(0.0)
    pdf["cdh_diff_24h"] = (pdf["cdh_55"] - pdf["cdh_55"].shift(24)).fillna(0.0)
    pdf["hdh_diff_24h"] = (pdf["hdh_55"] - pdf["hdh_55"].shift(24)).fillna(0.0)
    
    # 2. Thermal rolling average & peak interaction
    pdf["temp_rolling_24h"] = pdf["weighted_temperature_f"].rolling(24, min_periods=1).mean().fillna(pdf["weighted_temperature_f"])
    pdf["peak_temp_interaction"] = pdf["hour_local"].between(14, 18).astype(float) * pdf["temp_diff_24h"]
    
    # 3. Calendar shifts
    pdf["weekend_diff_24h"] = (pdf["is_weekend"].astype(float) - pdf["is_weekend"].astype(float).shift(24)).fillna(0.0)
    pdf["holiday_diff_24h"] = (pdf["is_holiday"].astype(float) - pdf["is_holiday"].astype(float).shift(24)).fillna(0.0)
    
    # 4. Momentum ratios
    pdf["demand_ratio_24_48"] = ((pdf["demand_lag_24h"] - pdf["demand_lag_48h"]) / pdf["demand_lag_48h"].clip(lower=1000.0)).fillna(0.0)
    pdf["demand_ratio_24_168"] = ((pdf["demand_lag_24h"] - pdf["demand_lag_168h"]) / pdf["demand_lag_168h"].clip(lower=1000.0)).fillna(0.0)
    pdf["demand_rolling_cv_24h"] = (pdf["demand_rolling_std_24h_lag24"] / pdf["demand_rolling_avg_24h_lag24"].clip(lower=1000.0)).fillna(0.0)
    
    for c in ["is_weekend", "is_holiday"]:
        if c in pdf.columns:
            pdf[c] = pdf[c].astype(float)
            
    X = pdf[feature_cols].copy()
    
    # Predict EIA error correction residual
    predicted_eia_error = booster.predict(X)
    
    # Reconstruct Day-Ahead point prediction: y_pred = EIA_forecast + predicted_eia_error
    pdf["predicted_demand_mw"] = np.round(pdf["day_ahead_forecast_mw"] + predicted_eia_error, 1)
    
    # Empirical 95% Confidence Interval (based on 1.96 * RMSE ~ 1,587 MW)
    HALF_WIDTH_95_MW = 1.96 * 1587.6
    pdf["prediction_lower_95_mw"] = np.round(pdf["predicted_demand_mw"] - HALF_WIDTH_95_MW, 1)
    pdf["prediction_upper_95_mw"] = np.round(pdf["predicted_demand_mw"] + HALF_WIDTH_95_MW, 1)
    
    # Operational Peak Risk Flag (True if predicted demand exceeds operator threshold)
    pdf["is_predicted_peak_risk"] = pdf["predicted_demand_mw"] >= PEAK_THRESHOLD_MW
    
    # Model tracking lineage
    pdf["model_version"] = f"LightGBM_Hybrid_{champion_run_id[:8]}"
    
    # Convert back to Spark
    spark_df = spark.createDataFrame(pdf[[
        "respondent", "timestamp_utc", "timestamp_local", "date_local", "hour_local", "day_of_week_local",
        "actual_demand_mw", "day_ahead_forecast_mw", "predicted_demand_mw",
        "prediction_lower_95_mw", "prediction_upper_95_mw", "is_predicted_peak_risk",
        "model_version"
    ]])
    
    # Attach evaluation metrics for hours where actual demand has been observed
    df_scored = (
        spark_df
        .withColumn("timestamp_utc", F.col("timestamp_utc").cast(TimestampType()))
        .withColumn("timestamp_local", F.col("timestamp_local").cast(TimestampType()))
        .withColumn(
            "model_error_mw",
            F.when(F.col("actual_demand_mw").isNotNull(),
                   F.round(F.col("predicted_demand_mw") - F.col("actual_demand_mw"), 1))
             .otherwise(F.lit(None).cast(DoubleType()))
        )
        .withColumn(
            "model_abs_error_mw",
            F.when(F.col("actual_demand_mw").isNotNull(), F.abs(F.col("model_error_mw")))
             .otherwise(F.lit(None).cast(DoubleType()))
        )
        .withColumn(
            "model_ape_pct",
            F.when(
                (F.col("actual_demand_mw").isNotNull()) & (F.col("actual_demand_mw") > 0.0),
                F.round((F.col("model_abs_error_mw") / F.col("actual_demand_mw")) * 100.0, 3)
            ).otherwise(F.lit(None).cast(DoubleType()))
        )
        .withColumn(
            "eia_error_mw",
            F.when(F.col("actual_demand_mw").isNotNull(),
                   F.round(F.col("day_ahead_forecast_mw") - F.col("actual_demand_mw"), 1))
             .otherwise(F.lit(None).cast(DoubleType()))
        )
        .withColumn(
            "eia_abs_error_mw",
            F.when(F.col("actual_demand_mw").isNotNull(), F.abs(F.col("eia_error_mw")))
             .otherwise(F.lit(None).cast(DoubleType()))
        )
        .withColumn(
            "is_model_better_than_eia",
            F.when(
                F.col("actual_demand_mw").isNotNull(),
                F.col("model_abs_error_mw") < F.col("eia_abs_error_mw")
            ).otherwise(F.lit(None).cast(BooleanType()))
        )
        .withColumn("_scored_at_utc", F.current_timestamp())
    )
    
    return df_scored

# COMMAND ----------
# DBTITLE 1,Pre-Write Quality Gate Assertions
def validate_forecast_predictions(df_scored):
    """
    Enforces data quality contracts before persisting into main.gold.forecast_hourly:
    1. Zero null keys (respondent, timestamp_utc)
    2. Zero duplicate timestamps
    3. Predictions stay within physically plausible grid limits (10k - 110k MW)
    4. Prediction bounds consistency (lower <= predicted <= upper)
    """
    total_rows = df_scored.count()
    distinct_keys = df_scored.select("respondent", "timestamp_utc").distinct().count()
    null_keys = df_scored.filter(F.col("respondent").isNull() | F.col("timestamp_utc").isNull()).count()
    
    assert null_keys == 0, f"Integrity Failure: Found {null_keys} null keys in scored forecast!"
    assert total_rows == distinct_keys, f"Integrity Failure: Found {total_rows - distinct_keys} duplicate timestamp records!"
    
    # Boundary checks (10k to 110k MW)
    out_of_bounds = df_scored.filter(
        (F.col("predicted_demand_mw") < 10000.0) | (F.col("predicted_demand_mw") > 110000.0)
    ).count()
    assert out_of_bounds == 0, f"Physical Boundary Failure: Found {out_of_bounds} predictions outside physical limits!"
    
    # Confidence interval order
    interval_inverted = df_scored.filter(
        (F.col("prediction_lower_95_mw") > F.col("predicted_demand_mw")) |
        (F.col("prediction_upper_95_mw") < F.col("predicted_demand_mw"))
    ).count()
    assert interval_inverted == 0, f"Math Failure: Found {interval_inverted} inverted confidence intervals!"
    
    print(f"Pre-Write Quality Gate Passed: {total_rows:,} verified forecasts.")

# COMMAND ----------
# DBTITLE 1,Idempotent Delta Upsert
def upsert_forecast_table(df_scored, target_table=TARGET_FORECAST_TABLE, spark_session=spark):
    """
    Idempotent MERGE into Gold Forecast table on (respondent, timestamp_utc).
    """
    if not spark_session.catalog.tableExists(target_table):
        df_scored.write \
            .format("delta") \
            .mode("overwrite") \
            .saveAsTable(target_table)
        print(f"Initialized Gold forecast table {target_table} with {df_scored.count():,} rows.")
    else:
        delta_target = DeltaTable.forName(spark_session, target_table)
        
        merge_condition = (
            "target.respondent = source.respondent AND "
            "target.timestamp_utc = source.timestamp_utc"
        )
        
        # Update if prediction, actuals, or error metrics changed
        update_condition = (
            "NOT (target.predicted_demand_mw <=> source.predicted_demand_mw) OR "
            "NOT (target.actual_demand_mw <=> source.actual_demand_mw) OR "
            "NOT (target.model_error_mw <=> source.model_error_mw)"
        )
        
        delta_target.alias("target").merge(
            df_scored.alias("source"),
            merge_condition
        ).whenMatchedUpdateAll(
            condition=update_condition
        ).whenNotMatchedInsertAll(
        ).execute()
        
        print(f"Successfully executed idempotent MERGE on {target_table}.")

# COMMAND ----------
# DBTITLE 1,Post-Write Read-Back Storage Audit
def audit_persisted_forecast_table(target_table, spark_session):
    df_persisted = spark_session.table(target_table)
    persisted_rows = df_persisted.count()
    distinct_keys = df_persisted.select("respondent", "timestamp_utc").distinct().count()
    null_keys = df_persisted.filter(
        F.col("respondent").isNull() | F.col("timestamp_utc").isNull()
    ).count()

    assert null_keys == 0, f"Post-Write Failure: Found {null_keys} null keys in {target_table}!"
    assert persisted_rows == distinct_keys, (
        f"Post-Write Failure: Found {persisted_rows - distinct_keys} duplicate keys in {target_table}!"
    )
    print(f"Post-Write Storage Audit Passed: {persisted_rows:,} verified records in {target_table}.")

# COMMAND ----------
# DBTITLE 1,Execute Forecast Scoring Pipeline
assert spark.catalog.tableExists(SOURCE_FEATURES_TABLE), f"Source {SOURCE_FEATURES_TABLE} does not exist!"
df_features = spark.table(SOURCE_FEATURES_TABLE)

df_scored = generate_forecast_predictions(df_features, champion_booster)
validate_forecast_predictions(df_scored)
upsert_forecast_table(df_scored, TARGET_FORECAST_TABLE, spark)
audit_persisted_forecast_table(TARGET_FORECAST_TABLE, spark)

# COMMAND ----------
# MAGIC %sql
# MAGIC -- Sample Verification of Live Production Predictions vs EIA Benchmark
# MAGIC SELECT 
# MAGIC     date_local,
# MAGIC     hour_local,
# MAGIC     actual_demand_mw,
# MAGIC     day_ahead_forecast_mw as eia_forecast_mw,
# MAGIC     predicted_demand_mw as model_forecast_mw,
# MAGIC     model_abs_error_mw,
# MAGIC     eia_abs_error_mw,
# MAGIC     is_model_better_than_eia,
# MAGIC     is_predicted_peak_risk
# MAGIC FROM gold.forecast_hourly
# MAGIC WHERE actual_demand_mw IS NOT NULL
# MAGIC ORDER BY timestamp_utc DESC
# MAGIC LIMIT 24;
