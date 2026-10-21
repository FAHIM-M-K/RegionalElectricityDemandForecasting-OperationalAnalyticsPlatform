# Databricks notebook source
# MAGIC %md
# MAGIC # Silver Layer: EIA Electricity Demand Cleaning & Conformance
# MAGIC
# MAGIC **Input Table:** `main.bronze.eia_hourly_raw`  
# MAGIC **Output Table:** `main.silver.electricity_demand_hourly`  
# MAGIC
# MAGIC ### Core Transformations:
# MAGIC 1. **Timezone Normalization:** Bronze requests EIA's `hourly` (UTC) frequency. Parse `period` as the canonical UTC key, then derive Central Time for reporting; do not reinterpret the source period as local wall time.
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
import datetime
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
from src.shared.delta_merge import merge_source_columns

config = load_platform_config("ercot_config.json")
REGION_CODE = config.get("region", {}).get("code", "ERCO")
REGION_TIMEZONE = config.get("region", {}).get("timezone", "America/Chicago")
RESTATEMENT_BUFFER_DAYS = int(config.get("eia", {}).get("restatement_buffer_days", 5))

# Physical plausible bounds for ERCOT grid (MW)
MIN_PLAUSIBLE_DEMAND_MW = 10000.0  # Lowest historical base load ~20k MW; <10k is a physical telemetry failure
MAX_PLAUSIBLE_DEMAND_MW = 110000.0 # All-time record peak is ~85.5k MW; >110k is impossible for current grid

# COMMAND ----------
# DBTITLE 1,Read Bronze Source
assert spark.catalog.tableExists(SOURCE_TABLE), f"Source table {SOURCE_TABLE} does not exist!"

target_exists = spark.catalog.tableExists(TARGET_TABLE)
target_cols = spark.table(TARGET_TABLE).columns if target_exists else []
last_actual_timestamp = None
if target_exists and "day_ahead_error_mw" not in target_cols:
    last_actual_timestamp = (
        spark.table(TARGET_TABLE)
        .filter(F.col("actual_demand_mw").isNotNull())
        .agg(F.max("timestamp_utc").alias("last_actual_timestamp"))
        .first()["last_actual_timestamp"]
    )
requires_full_rebuild = (
    not target_exists or "day_ahead_error_mw" in target_cols or last_actual_timestamp is None
)
df_bronze = spark.table(SOURCE_TABLE).filter(F.col("respondent") == REGION_CODE)

if requires_full_rebuild:
    print("Full Silver build required for initial creation or legacy schema migration.")
else:
    cutoff_date = (
        last_actual_timestamp.date()
        - datetime.timedelta(days=RESTATEMENT_BUFFER_DAYS)
    ).strftime("%Y-%m-%d")
    merge_cutoff_utc = f"{cutoff_date} 00:00:00"
    df_bronze = df_bronze.filter(F.col("period") >= F.lit(f"{cutoff_date}T00"))
    print(
        f"Incremental Silver build: processing EIA periods from {cutoff_date} "
        f"(with a {RESTATEMENT_BUFFER_DAYS}-day restatement lookback)."
    )

# COMMAND ----------
# DBTITLE 1,Parse Timestamps & Assert Source Uniqueness
# Bronze requests EIA's `hourly` frequency (UTC), not `local-hourly`.
# Preserve period as the canonical UTC hour; derive CT only as a display/calendar field.
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
pre_pivot_dupe = df_parsed.groupBy("respondent", "period", "type").count().filter("count > 1").limit(1).count()
assert pre_pivot_dupe == 0, "Source Integrity Failure: duplicate (respondent, period, type) rows in Bronze."

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
quality = df_screened.agg(
    F.count(F.lit(1)).alias("rows"),
    F.sum(F.when(F.col("respondent").isNull() | F.col("timestamp_utc").isNull(), 1).otherwise(0)).alias("null_keys"),
    F.sum(F.when(F.col("is_valid_demand") == True, 1).otherwise(0)).alias("valid_rows"),
    F.sum(F.when(F.col("anomaly_flag").isNotNull(), 1).otherwise(0)).alias("anomaly_rows"),
).first()
assert not quality["null_keys"], f"Integrity Failure: {quality['null_keys']} null composite keys."
duplicate_key = df_screened.groupBy("respondent", "timestamp_utc").count().filter("count > 1").limit(1).count()
assert duplicate_key == 0, "Integrity Failure: duplicate natural keys in Silver transformation."
total_records = quality["rows"] or 0

print(f"Silver EIA pre-merge checks passed for {total_records:,} impacted rows; valid={quality['valid_rows'] or 0:,}, flagged={quality['anomaly_rows'] or 0:,}.")

# COMMAND ----------
# DBTITLE 1,Idempotent Delta Upsert with Null-Safe Comparison
if not target_exists:
    df_screened.write \
        .format("delta") \
        .mode("overwrite") \
        .saveAsTable(TARGET_TABLE)
    print(f"Initialized Silver table {TARGET_TABLE} with {total_records} records.")
else:
    delta_target = DeltaTable.forName(spark, TARGET_TABLE)
    # Schema Migration Check:
    # If the target table was initialized in an earlier run and still contains 'day_ahead_error_mw',
    # overwrite the schema once to align with the Medallion rule (error metrics moved to Gold).
    if requires_full_rebuild:
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
            "AND target.timestamp_utc = source.timestamp_utc "
            f"AND target.timestamp_utc >= TIMESTAMP '{merge_cutoff_utc}'"
        )
        
        # Update only if measurements or anomaly flags actually changed
        update_condition = (
            "NOT (target.actual_demand_mw <=> source.actual_demand_mw) OR "
            "NOT (target.day_ahead_forecast_mw <=> source.day_ahead_forecast_mw) OR "
            "NOT (target.anomaly_flag <=> source.anomaly_flag) OR "
            "NOT (target.is_valid_demand <=> source.is_valid_demand)"
        )
        
        merge_source_columns(
            delta_target,
            df_screened,
            merge_condition,
            update_condition=update_condition,
            update_exclude=("respondent", "timestamp_utc"),
        )
        print(f"Idempotent MERGE successfully executed on {TARGET_TABLE}.")
