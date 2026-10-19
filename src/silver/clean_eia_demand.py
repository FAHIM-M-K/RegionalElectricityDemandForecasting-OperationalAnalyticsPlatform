# Databricks notebook source
# MAGIC %md
# MAGIC # Silver Layer: EIA Electricity Demand Cleaning & Conformance
# MAGIC
# MAGIC **Input Table:** `main.bronze.eia_hourly_raw`  
# MAGIC **Output Table:** `main.silver.electricity_demand_hourly`  
# MAGIC
# MAGIC ### Core Transformations:
# MAGIC 1. **Timezone Normalization:** Parse EIA `period` string (`YYYY-MM-DDTHH`) to exact UTC timestamp (`timestamp_utc`) and local Central Time (`timestamp_local`, `America/Chicago`).
# MAGIC 2. **Pivoting Series:** Restructure long format (`D` = Demand, `DF` = Day-ahead Forecast) into single hourly row: `actual_demand_mw` and `day_ahead_forecast_mw`.
# MAGIC 3. **Physical Domain Screening:** Flag negative demand and physical operating boundary violations without silently dropping rows (`is_valid_demand`, `anomaly_flag`).
# MAGIC 4. **Incremental Idempotency:** Null-safe Delta `MERGE` on `(respondent, timestamp_utc)` updating only when values change or restatements arrive.
# MAGIC
# MAGIC ### Design Decision:
# MAGIC `day_ahead_error_mw` (actual - forecast) is NOT computed here. It is an analytical derivation, not data cleaning.
# MAGIC Silver provides the two clean source values; Gold computes the delta. (Medallion: Silver = clean/conform, Gold = business logic).

# COMMAND ----------
# DBTITLE 1,Imports & Configuration
import os
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, BooleanType
from delta.tables import DeltaTable

CATALOG = "main"
SOURCE_TABLE = f"{CATALOG}.bronze.eia_hourly_raw"
TARGET_TABLE = f"{CATALOG}.silver.electricity_demand_hourly"

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql("USE SCHEMA silver")
spark.conf.set("spark.sql.session.timeZone", "UTC")

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
REGION_CODE = config.get("region", {}).get("code", "ERCO")
REGION_TIMEZONE = config.get("region", {}).get("timezone", "America/Chicago")

# Physical plausible bounds for ERCOT grid (MW)
MIN_PLAUSIBLE_DEMAND_MW = 10000.0  # Lowest historical base load ~20k MW; <10k is a physical telemetry failure
MAX_PLAUSIBLE_DEMAND_MW = 110000.0 # All-time record peak is ~85.5k MW; >110k is impossible for current grid

# COMMAND ----------
# DBTITLE 1,Read Bronze Source
assert spark.catalog.tableExists(SOURCE_TABLE), f"Source table {SOURCE_TABLE} does not exist!"

df_bronze = spark.table(SOURCE_TABLE).filter(F.col("respondent") == REGION_CODE)

# COMMAND ----------
# DBTITLE 1,Parse Timestamps & Assert Source Uniqueness
# EIA period is string formatted as "yyyy-MM-ddTHH" (representing UTC hour)
# Example: "2021-01-01T00" -> 2021-01-01 00:00:00 UTC
df_parsed = df_bronze \
    .withColumn(
        "timestamp_utc",
        F.to_timestamp(F.col("period"), "yyyy-MM-dd'T'HH")
    ) \
    .withColumn(
        "timestamp_local",
        F.from_utc_timestamp(F.col("timestamp_utc"), REGION_TIMEZONE)
    ) \
    .withColumn("value_num", F.col("value").cast(DoubleType()))

# Assert source natural key uniqueness prior to pivoting (prevents silent multi-row collisions)
pre_pivot_dupes = df_parsed.count() - df_parsed.dropDuplicates(["respondent", "period", "type"]).count()
assert pre_pivot_dupes == 0, f"Source Integrity Failure: Found {pre_pivot_dupes} duplicate (respondent, period, type) records in Bronze."

# COMMAND ----------
# DBTITLE 1,Pivot Series: Align Actuals (D) and Day-Ahead Forecast (DF)
# Filter only D and DF, then aggregate by (respondent, timestamp_utc, timestamp_local)
df_pivoted = df_parsed \
    .filter(F.col("type").isin("D", "DF")) \
    .groupBy("respondent", "timestamp_utc", "timestamp_local") \
    .agg(
        F.max(F.when(F.col("type") == "D", F.col("value_num"))).alias("actual_demand_mw"),
        F.max(F.when(F.col("type") == "DF", F.col("value_num"))).alias("day_ahead_forecast_mw"),
        F.max("_ingested_at_utc").alias("_bronze_ingested_at_utc")
    )

# COMMAND ----------
# DBTITLE 1,Physical Anomaly Screening & Audit Flags
# Literature & grid realities: negative values and extreme spikes must be flagged, not dropped.
df_screened = df_pivoted \
    .withColumn(
        "anomaly_flag",
        F.when(F.col("actual_demand_mw") < 0.0, F.lit("NEGATIVE_DEMAND"))
         .when(F.col("actual_demand_mw") < MIN_PLAUSIBLE_DEMAND_MW, F.lit("BELOW_PHYSICAL_MIN"))
         .when(F.col("actual_demand_mw") > MAX_PLAUSIBLE_DEMAND_MW, F.lit("ABOVE_PHYSICAL_MAX"))
         .when(F.col("actual_demand_mw").isNull(), F.lit("MISSING_ACTUAL"))
         .otherwise(F.lit(None).cast(StringType()))
    ) \
    .withColumn(
        "is_valid_demand",
        F.when(F.col("actual_demand_mw").isNull(), F.lit(None).cast(BooleanType()))
         .when(
            (F.col("actual_demand_mw") >= MIN_PLAUSIBLE_DEMAND_MW) &
            (F.col("actual_demand_mw") <= MAX_PLAUSIBLE_DEMAND_MW),
            F.lit(True)
        ).otherwise(F.lit(False))
    ) \
    .withColumn("_transformed_at_utc", F.current_timestamp())

# COMMAND ----------
# DBTITLE 1,Quality & Structural Pre-Merge Assertions
# 1. Zero null keys
null_keys_count = df_screened.filter(
    F.col("respondent").isNull() | F.col("timestamp_utc").isNull()
).count()
assert null_keys_count == 0, f"Integrity Failure: Found {null_keys_count} rows with null composite keys."

# 2. Strict deduplication check on target natural key
dupe_keys_count = df_screened.count() - df_screened.dropDuplicates(["respondent", "timestamp_utc"]).count()
assert dupe_keys_count == 0, f"Integrity Failure: Found {dupe_keys_count} duplicate keys in silver transformation."

total_records = df_screened.count()
valid_count = df_screened.filter(F.col("is_valid_demand") == True).count()
anomalies_count = df_screened.filter(F.col("anomaly_flag").isNotNull()).count()

print(f"Silver EIA Pre-Merge Validation: Total {total_records} hours | Valid: {valid_count} | Flagged Anomalies: {anomalies_count}")

# COMMAND ----------
# DBTITLE 1,Idempotent Delta Upsert with Null-Safe Comparison
if not spark.catalog.tableExists(TARGET_TABLE):
    df_screened.write \
        .format("delta") \
        .mode("overwrite") \
        .saveAsTable(TARGET_TABLE)
    print(f"Initialized Silver table {TARGET_TABLE} with {total_records} records.")
else:
    delta_target = DeltaTable.forName(spark, TARGET_TABLE)
    target_cols = [c.name for c in delta_target.toDF().schema]
    
    # Schema Migration Check:
    # If the target table was initialized in an earlier run and still contains 'day_ahead_error_mw',
    # overwrite the schema once to align with the Medallion rule (error metrics moved to Gold).
    if "day_ahead_error_mw" in target_cols:
        print(f"Migrating Silver schema: dropping legacy 'day_ahead_error_mw' from {TARGET_TABLE}...")
        df_screened.write \
            .format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .saveAsTable(TARGET_TABLE)
        print(f"Successfully migrated and overwritten {TARGET_TABLE} to clean Silver schema.")
    else:
        merge_condition = (
            "target.respondent = source.respondent "
            "AND target.timestamp_utc = source.timestamp_utc"
        )
        
        # Update only if measurements or anomaly flags actually changed
        update_condition = (
            "NOT (target.actual_demand_mw <=> source.actual_demand_mw) OR "
            "NOT (target.day_ahead_forecast_mw <=> source.day_ahead_forecast_mw) OR "
            "NOT (target.anomaly_flag <=> source.anomaly_flag) OR "
            "NOT (target.is_valid_demand <=> source.is_valid_demand)"
        )
        
        # Explicit column mapping guarantees resilience against schema differences
        update_cols = [c for c in df_screened.columns if c not in ["respondent", "timestamp_utc"]]
        update_set = {col: f"source.{col}" for col in update_cols}
        insert_set = {col: f"source.{col}" for col in df_screened.columns}
        
        delta_target.alias("target").merge(
            df_screened.alias("source"),
            merge_condition
        ).whenMatchedUpdate(
            condition=update_condition,
            set=update_set
        ).whenNotMatchedInsert(
            values=insert_set
        ).execute()
        print(f"Idempotent MERGE successfully executed on {TARGET_TABLE}.")

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT 
# MAGIC     count(*) as total_hours,
# MAGIC     sum(case when is_valid_demand then 1 else 0 end) as valid_hours,
# MAGIC     sum(case when anomaly_flag is not null then 1 else 0 end) as anomaly_hours,
# MAGIC     min(timestamp_local) as min_local_time,
# MAGIC     max(timestamp_local) as max_local_time,
# MAGIC     round(avg(actual_demand_mw), 2) as avg_demand_mw,
# MAGIC     round(max(actual_demand_mw), 2) as peak_demand_mw
# MAGIC FROM silver.electricity_demand_hourly;
