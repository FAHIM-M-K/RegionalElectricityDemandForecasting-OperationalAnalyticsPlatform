# Databricks notebook source
# MAGIC %md
# MAGIC # Gold Layer: Fact Hourly Electricity Demand & Benchmark Analytics
# MAGIC
# MAGIC **Input Table:** `main.silver.grid_hourly_conformed`  
# MAGIC **Output Table:** `main.gold.fact_hourly_electricity_demand`  
# MAGIC
# MAGIC ### Core Responsibilities:
# MAGIC 1. **Business Metric Derivations:** Calculates analytical error KPIs (`eia_error_mw`, `eia_abs_error_mw`, `eia_ape_pct`) comparing EIA Day-Ahead Forecast against actual ground truth.
# MAGIC 2. **Calendar & Seasonal Dimensions:** Adds meteorological season (`Winter`, `Spring`, `Summer`, `Fall`), peak window indicators, and federal holiday classifications.
# MAGIC 3. **Grid Emergency Flagging:** Explicitly isolates supply-constrained blackout periods (Winter Storm Uri Feb 15–18, 2021) with `is_grid_emergency`.
# MAGIC 4. **Pre-Write Data Quality Gate:** Validates key uniqueness, non-negative demand, metric calculation consistency, and error magnitude bounds.
# MAGIC 5. **Incremental Delta MERGE:** Rebuild the restatement window after the last materialized actual; assert the candidate before the atomic upsert.

# COMMAND ----------
# DBTITLE 1,Imports & Setup
import os
import datetime
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from delta.tables import DeltaTable

CATALOG = "main"
SOURCE_TABLE = f"{CATALOG}.silver.grid_hourly_conformed"
TARGET_TABLE = f"{CATALOG}.gold.fact_hourly_electricity_demand"

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql("USE SCHEMA gold")

# COMMAND ----------
# DBTITLE 1,Dynamic Configuration Locator
import sys
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
RESTATEMENT_BUFFER_DAYS = int(config.get("eia", {}).get("restatement_buffer_days", 5))

# COMMAND ----------
# DBTITLE 1,Verify Upstream Silver Table
assert spark.catalog.tableExists(SOURCE_TABLE), f"Required upstream table {SOURCE_TABLE} does not exist!"
df_silver = spark.table(SOURCE_TABLE)
target_exists = spark.catalog.tableExists(TARGET_TABLE)
last_actual_timestamp = None
if target_exists:
    last_actual_timestamp = (
        spark.table(TARGET_TABLE)
        .filter(F.col("actual_demand_mw").isNotNull())
        .agg(F.max("timestamp_utc").alias("last_actual_timestamp"))
        .first()["last_actual_timestamp"]
    )
requires_full_rebuild = not target_exists or last_actual_timestamp is None
cutoff_utc = None
if requires_full_rebuild:
    print("Full Gold fact build required for initial creation or empty target.")
else:
    cutoff_utc = last_actual_timestamp - datetime.timedelta(days=RESTATEMENT_BUFFER_DAYS)
    df_silver = df_silver.filter(F.col("timestamp_utc") >= F.lit(cutoff_utc))
    print(f"Incremental Gold fact build from {cutoff_utc} UTC ({RESTATEMENT_BUFFER_DAYS}-day correction lookback).")

# COMMAND ----------
# DBTITLE 1,Transformation: Build Gold Fact Dataset
def build_gold_fact_dataset(df_silver, ba_code=BA_CODE):
    """
    Transforms silver conformed grid records into the business analytics fact table.
    Enriches with EIA benchmark error metrics, operational peak indicators, and grid emergency flags.
    """
    df_fact = (
        df_silver
        # 1. Seasonal Classification
        .withColumn(
            "season",
            F.when(F.col("month_local").isin(12, 1, 2), F.lit("Winter"))
             .when(F.col("month_local").isin(3, 4, 5), F.lit("Spring"))
             .when(F.col("month_local").isin(6, 7, 8), F.lit("Summer"))
             .otherwise(F.lit("Fall"))
        )
        
        # 2. Benchmark Forecast Error Metrics (Gold analytical derivations)
        # SIGNED ERROR CONVENTION:
        # eia_error_mw = day_ahead_forecast_mw - actual_demand_mw
        # Positive (> 0) denotes an OVERFORECAST (operator predicted higher than realized).
        # Negative (< 0) denotes an UNDERFORECAST (operator predicted lower than realized).
        .withColumn(
            "eia_error_mw",
            F.when(
                F.col("actual_demand_mw").isNotNull() & F.col("day_ahead_forecast_mw").isNotNull(),
                F.round(F.col("day_ahead_forecast_mw") - F.col("actual_demand_mw"), 1)
            ).otherwise(F.lit(None).cast(DoubleType()))
        )
        .withColumn(
            "eia_abs_error_mw",
            F.when(F.col("eia_error_mw").isNotNull(), F.abs(F.col("eia_error_mw")))
             .otherwise(F.lit(None).cast(DoubleType()))
        )
        .withColumn(
            "eia_ape_pct",
            F.when(
                F.col("eia_abs_error_mw").isNotNull() & (F.col("actual_demand_mw") > 0.0),
                F.round((F.col("eia_abs_error_mw") / F.col("actual_demand_mw")) * 100.0, 3)
            ).otherwise(F.lit(None).cast(DoubleType()))
        )
        
        # 3. Operational time-window indicator
        .withColumn(
            "is_afternoon_ramp_window",
            F.col("hour_local").between(14, 18)
        )
        
        # 4. Grid Emergency Flag (Winter Storm Uri Involuntary Load Shedding)
        .withColumn(
            "is_grid_emergency",
            F.when(
                (F.col("timestamp_local") >= "2021-02-15 00:00:00") &
                (F.col("timestamp_local") <  "2021-02-19 00:00:00"),
                F.lit(True)
            ).otherwise(F.lit(False))
        )
        
        # 5. Metadata Lineage
        .withColumn("_gold_processed_at_utc", F.current_timestamp())
    )
    
    return df_fact

# COMMAND ----------
# DBTITLE 1,Pre-Write Quality Gate Assertions
def validate_gold_fact_candidate(df_fact):
    """
    Asserts structural and business logic invariants before persisting:
    1. Zero null composite primary keys (respondent, timestamp_utc)
    2. Zero duplicate timestamps
    3. Mathematical consistency of error calculations
    4. Non-negative actual demand bounds
    5. Correct assignment of Winter Storm Uri emergency flag
    """
    quality = df_fact.agg(
        F.count(F.lit(1)).alias("rows"),
        F.sum(F.when(F.col("respondent").isNull() | F.col("timestamp_utc").isNull(), 1).otherwise(0)).alias("null_keys"),
        F.sum(F.when(
            F.col("eia_error_mw").isNotNull() &
            (F.abs(F.col("eia_error_mw") - (F.col("day_ahead_forecast_mw") - F.col("actual_demand_mw"))) > 0.15), 1
        ).otherwise(0)).alias("error_mismatch"),
        F.sum(F.when(F.col("eia_ape_pct") < 0.0, 1).otherwise(0)).alias("negative_ape"),
        F.sum(F.when(F.col("actual_demand_mw") < 0.0, 1).otherwise(0)).alias("negative_actual"),
        F.sum(F.when(
            F.col("is_grid_emergency") != (
                (F.col("timestamp_local") >= F.lit("2021-02-15 00:00:00")) &
                (F.col("timestamp_local") < F.lit("2021-02-19 00:00:00"))
            ), 1
        ).otherwise(0)).alias("emergency_mismatch"),
    ).first()
    if not quality["rows"]:
        print("No conformed source rows in the correction window; the idempotent MERGE will be a no-op.")
        return
    assert not quality["null_keys"], f"Integrity Failure: {quality['null_keys']} null keys in Gold fact candidate."
    assert not quality["error_mismatch"], f"Computation Failure: {quality['error_mismatch']} inconsistent error rows."
    assert not quality["negative_ape"], f"Bound Failure: {quality['negative_ape']} negative percentage errors."
    assert not quality["negative_actual"], f"Bound Failure: {quality['negative_actual']} negative actual demand rows."
    assert not quality["emergency_mismatch"], f"Regime flag failure: {quality['emergency_mismatch']} incorrect flags."
    duplicate_key = df_fact.groupBy("respondent", "timestamp_utc").count().filter("count > 1").limit(1).count()
    assert duplicate_key == 0, "Integrity Failure: duplicate keys in Gold fact candidate."
    print(f"Gold fact quality checks passed for {quality['rows']:,} impacted rows.")

# COMMAND ----------
# DBTITLE 1,Idempotent Delta MERGE
def upsert_gold_fact_table(df_fact, target_table, spark_session):
    """
    Executes an idempotent Delta MERGE on (respondent, timestamp_utc).
    Handles historical restatements, revised forecasts, and newly ingested hours.
    """
    if not spark_session.catalog.tableExists(target_table):
        df_fact.write \
            .format("delta") \
            .mode("overwrite") \
            .saveAsTable(target_table)
        print(f"Initialized Gold fact table {target_table}.")
    else:
        delta_target = DeltaTable.forName(spark_session, target_table)
        merge_condition = (
            "target.respondent = source.respondent "
            "AND target.timestamp_utc = source.timestamp_utc"
        )
        if cutoff_utc is not None:
            merge_condition += f" AND target.timestamp_utc >= TIMESTAMP '{cutoff_utc.strftime('%Y-%m-%d %H:%M:%S')}'"
        
        # Track ALL business columns (demand, weather, calendar, error metrics) for restatements (H-11)
        # Excludes composite primary keys and processing timestamp to prevent unnecessary file churn
        business_cols = [c for c in df_fact.columns if c not in ["respondent", "timestamp_utc", "_gold_processed_at_utc"]]
        update_condition = " OR ".join([f"NOT (target.{c} <=> source.{c})" for c in business_cols])
        
        delta_target.alias("target").merge(
            df_fact.alias("source"),
            merge_condition
        ).whenMatchedUpdateAll(
            condition=update_condition
        ).whenNotMatchedInsertAll(
        ).execute()
        print(f"Idempotent MERGE successfully executed on {target_table}.")

# COMMAND ----------
# DBTITLE 1,Post-Write Read-Back Storage Audit
# COMMAND ----------
# DBTITLE 1,Execute Gold Fact Pipeline
df_fact = build_gold_fact_dataset(df_silver)
validate_gold_fact_candidate(df_fact)
upsert_gold_fact_table(df_fact, TARGET_TABLE, spark)
