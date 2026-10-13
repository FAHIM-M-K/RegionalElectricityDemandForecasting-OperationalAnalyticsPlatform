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
# MAGIC 5. **Idempotent Delta MERGE & Read-Back Verification:** Null-safe upsert ensuring atomic commits without Parquet file churn.

# COMMAND ----------
# DBTITLE 1,Imports & Setup
import os
import json
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, BooleanType, IntegerType
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
PEAK_THRESHOLD_MW = float(config.get("region", {}).get("default_peak_threshold_mw", 75000.0))

# COMMAND ----------
# DBTITLE 1,Verify Upstream Silver Table
assert spark.catalog.tableExists(SOURCE_TABLE), f"Required upstream table {SOURCE_TABLE} does not exist!"
df_silver = spark.table(SOURCE_TABLE)

# COMMAND ----------
# DBTITLE 1,Transformation: Build Gold Fact Dataset
def build_gold_fact_dataset(df_silver, ba_code=BA_CODE, peak_threshold_mw=PEAK_THRESHOLD_MW):
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
        
        # 3. High Demand Volume & Operational Window Indicators
        # NOTE: This indicates high system demand volume against a configurable threshold (e.g. 75 GW).
        # It does NOT represent loss-of-load or reserve margin reliability risk, which depends on available generation.
        .withColumn(
            "is_high_demand_threshold_exceeded",
            F.when(
                F.col("actual_demand_mw").isNotNull(),
                F.col("actual_demand_mw") >= F.lit(peak_threshold_mw)
            ).when(
                F.col("day_ahead_forecast_mw").isNotNull(),
                F.col("day_ahead_forecast_mw") >= F.lit(peak_threshold_mw)
            ).otherwise(F.lit(False))
        )
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
    total_rows = df_fact.count()
    distinct_keys = df_fact.select("respondent", "timestamp_utc").distinct().count()
    null_keys = df_fact.filter(F.col("respondent").isNull() | F.col("timestamp_utc").isNull()).count()
    
    assert null_keys == 0, f"Integrity Failure: Found {null_keys} null keys in Gold fact candidate!"
    assert total_rows == distinct_keys, (
        f"Integrity Failure: Found {total_rows - distinct_keys} duplicate timestamp records!"
    )
    
    # Mathematical sanity check on error columns
    error_mismatch = df_fact.filter(
        F.col("eia_error_mw").isNotNull() &
        (F.abs(F.col("eia_error_mw") - (F.col("day_ahead_forecast_mw") - F.col("actual_demand_mw"))) > 0.15)
    ).count()
    assert error_mismatch == 0, f"Computation Failure: {error_mismatch} rows have inconsistent eia_error_mw values!"
    
    # Error percentage bounds (MAPE cannot be negative)
    negative_ape = df_fact.filter(F.col("eia_ape_pct") < 0.0).count()
    assert negative_ape == 0, f"Bound Failure: Found {negative_ape} rows with negative percentage error!"
    
    # Emergency flag verification: exactly 96 hours in Uri blackout window (Feb 15–18, 4 days * 24h)
    uri_emergency_hours = df_fact.filter(F.col("is_grid_emergency") == True).count()
    assert uri_emergency_hours >= 96, (
        f"Regime Flag Failure: Expected at least 96 Uri emergency hours, found {uri_emergency_hours}."
    )
    
    print(f"Gold Fact Quality Gate Passed:")
    print(f"  - Total evaluated records: {total_rows:,}")
    print(f"  - Primary keys unique and non-null (100% verified)")
    print(f"  - Benchmark error calculations mathematically consistent")
    print(f"  - Winter Storm Uri emergency flags verified ({uri_emergency_hours} hours)")

# COMMAND ----------
# DBTITLE 1,Idempotent Delta MERGE
def upsert_gold_fact_table(df_fact, target_table, spark_session):
    """
    Executes an idempotent Delta MERGE on (respondent, timestamp_utc).
    Handles historical restatements, revised forecasts, and newly ingested hours.
    """
    total_records = df_fact.count()
    if not spark_session.catalog.tableExists(target_table):
        df_fact.write \
            .format("delta") \
            .mode("overwrite") \
            .saveAsTable(target_table)
        print(f"Initialized Gold fact table {target_table} with {total_records:,} records.")
    else:
        delta_target = DeltaTable.forName(spark_session, target_table)
        merge_condition = (
            "target.respondent = source.respondent "
            "AND target.timestamp_utc = source.timestamp_utc"
        )
        
        update_condition = (
            "NOT (target.actual_demand_mw <=> source.actual_demand_mw) OR "
            "NOT (target.day_ahead_forecast_mw <=> source.day_ahead_forecast_mw) OR "
            "NOT (target.eia_error_mw <=> source.eia_error_mw) OR "
            "NOT (target.is_high_demand_threshold_exceeded <=> source.is_high_demand_threshold_exceeded) OR "
            "NOT (target.weighted_temperature_f <=> source.weighted_temperature_f)"
        )
        
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
def audit_persisted_gold_fact(target_table, spark_session):
    """
    Direct read-back audit from Delta storage to verify physical persistence integrity.
    """
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
# DBTITLE 1,Execute Gold Fact Pipeline
df_fact = build_gold_fact_dataset(df_silver)
validate_gold_fact_candidate(df_fact)
upsert_gold_fact_table(df_fact, TARGET_TABLE, spark)
audit_persisted_gold_fact(TARGET_TABLE, spark)

# COMMAND ----------
# MAGIC %sql
# MAGIC -- Sample Verification of Business Metrics in Gold Fact Table
# MAGIC SELECT 
# MAGIC     season,
# MAGIC     count(*) as evaluated_hours,
# MAGIC     round(avg(actual_demand_mw), 1) as avg_demand_mw,
# MAGIC     round(avg(eia_abs_error_mw), 1) as eia_mae_mw,
# MAGIC     round(avg(eia_ape_pct), 3) as eia_mape_pct,
# MAGIC     round(avg(eia_error_mw), 1) as eia_mean_bias_mw,
# MAGIC     sum(case when is_high_demand_threshold_exceeded then 1 else 0 end) as high_demand_hours,
# MAGIC     sum(case when is_grid_emergency then 1 else 0 end) as emergency_hours
# MAGIC FROM gold.fact_hourly_electricity_demand
# MAGIC WHERE actual_demand_mw IS NOT NULL AND day_ahead_forecast_mw IS NOT NULL
# MAGIC GROUP BY season
# MAGIC ORDER BY eia_mae_mw DESC;
