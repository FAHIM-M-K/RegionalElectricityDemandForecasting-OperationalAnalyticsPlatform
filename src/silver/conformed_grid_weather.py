# Databricks notebook source
# MAGIC %md
# MAGIC # Silver Layer: Conformed Grid Demand, Weather & Calendar Integration
# MAGIC
# MAGIC **Input Tables:**
# MAGIC - `main.silver.electricity_demand_hourly`
# MAGIC - `main.silver.weather_hourly_conformed`
# MAGIC - `main.bronze.calendar_reference`
# MAGIC
# MAGIC **Output Table:** `main.silver.grid_hourly_conformed`
# MAGIC
# MAGIC ### Core Transformations:
# MAGIC 1. **1:1 Chronological Conformance:** Full outer join on exact `timestamp_utc` grain (preserves weather-only future hours for Gold scoring).
# MAGIC 2. **Calendar & Operational Dimension Join:** Join on date string to attach holiday name, holiday flag, day-of-week, and weekend indicators.
# MAGIC 3. **Time Feature Extraction:** Extract hour of day (local), month, day of year, quarter for temporal seasonality analysis.
# MAGIC 4. **Incremental Processing:** Routine runs rebuild only the correction window anchored to the last actual hour successfully materialized; first builds and schema migrations remain full-history.
# MAGIC 5. **Candidate Quality Gate & Idempotent Delta MERGE:** Validate the changed slice before the atomic write; periodic full reconciliation belongs in a separate maintenance run, not every serving run.
# MAGIC
# MAGIC ### Design Decisions:
# MAGIC - **No business metrics here.** `day_ahead_error_mw` and `eia_day_ahead_ape_pct` belong in Gold (analytical derivations, not conformance).
# MAGIC - **BA code from config.** Weather-only rows get their respondent from `ercot_config.json`, not a hardcoded string.
# MAGIC - **`is_valid_demand` stays NULL for absent rows.** NULL = 'no data', False = 'data exists but is invalid'. Silver doesn't interpret absence.
# MAGIC - **Failure Semantics:** Delta MERGE is atomic. Candidate assertions run before commit; a failed assertion prevents downstream Gold tasks from consuming the candidate.

# COMMAND ----------
# DBTITLE 1,Imports & Configuration
import os
import datetime
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from delta.tables import DeltaTable

CATALOG = "main"
DEMAND_TABLE = f"{CATALOG}.silver.electricity_demand_hourly"
WEATHER_TABLE = f"{CATALOG}.silver.weather_hourly_conformed"
CALENDAR_TABLE = f"{CATALOG}.bronze.calendar_reference"
TARGET_TABLE = f"{CATALOG}.silver.grid_hourly_conformed"

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
BA_CODE = config.get("region", {}).get("code", "ERCO")
RESTATEMENT_BUFFER_DAYS = int(config.get("eia", {}).get("restatement_buffer_days", 5))

# COMMAND ----------
# DBTITLE 1,Verify Source Tables Exist
for tbl in [DEMAND_TABLE, CALENDAR_TABLE]:
    assert spark.catalog.tableExists(tbl), f"Required upstream table {tbl} does not exist!"

df_demand = spark.table(DEMAND_TABLE)
if spark.catalog.tableExists(WEATHER_TABLE):
    df_weather = spark.table(WEATHER_TABLE)
else:
    # Weather is an enrichment source, not a prerequisite for hourly demand
    # serving. An empty, typed frame keeps the canonical conformed schema stable.
    from pyspark.sql.types import StructType, StructField, TimestampType, DoubleType
    weather_schema = StructType([
        StructField("timestamp_utc", TimestampType(), True),
        StructField("timestamp_local", TimestampType(), True),
        *[StructField(name, DoubleType(), True) for name in [
            "weighted_temperature_c", "weighted_temperature_f",
            "weighted_apparent_temperature_c", "weighted_apparent_temperature_f",
            "weighted_dew_point_c", "weighted_relative_humidity_pct",
            "weighted_precipitation_mm", "weighted_wind_speed_ms",
            "weighted_solar_irradiance_wm2", "cdh_65", "hdh_65",
            "temp_houston_c", "temp_dfw_c", "temp_san_antonio_c", "temp_austin_c",
        ]],
    ])
    df_weather = spark.createDataFrame([], weather_schema)
    print(f"Weather table {WEATHER_TABLE} is absent; continuing with demand and calendar inputs only.")
df_calendar = spark.table(CALENDAR_TABLE)
assert df_demand.filter(F.col("actual_demand_mw").isNotNull()).limit(1).count() > 0, (
    f"Required demand table {DEMAND_TABLE} has no usable actual demand rows."
)

target_exists = spark.catalog.tableExists(TARGET_TABLE)
target_cols = spark.table(TARGET_TABLE).columns if target_exists else []
legacy_cols = [c for c in ["eia_day_ahead_ape_pct", "day_ahead_error_mw"] if c in target_cols]
last_actual_timestamp = None
if target_exists and not legacy_cols:
    last_actual_timestamp = (
        spark.table(TARGET_TABLE)
        .filter(F.col("actual_demand_mw").isNotNull())
        .agg(F.max("timestamp_utc").alias("last_actual_timestamp"))
        .first()["last_actual_timestamp"]
    )
requires_full_rebuild = not target_exists or bool(legacy_cols) or last_actual_timestamp is None
cutoff_utc = None
if requires_full_rebuild:
    print("Full conformed Silver build required for initial creation, empty target, or legacy schema migration.")
else:
    cutoff_utc = last_actual_timestamp - datetime.timedelta(days=RESTATEMENT_BUFFER_DAYS)
    df_demand = df_demand.filter(F.col("timestamp_utc") >= F.lit(cutoff_utc))
    df_weather = df_weather.filter(F.col("timestamp_utc") >= F.lit(cutoff_utc))
    print(f"Incremental conformation from {cutoff_utc} UTC ({RESTATEMENT_BUFFER_DAYS}-day correction lookback).")

# COMMAND ----------
# DBTITLE 1,Pipeline Functions: Transformation, Validation & Upsert
def build_conformed_dataset(df_demand, df_weather, df_calendar, ba_code=BA_CODE):
    """
    Integrates clean demand, clean weather, and calendar reference into an
    atomic hourly conformed dataset via a full outer join on timestamp_utc.
    """
    # 1. Full outer join demand and weather on timestamp_utc
    df_grid_weather = df_demand.alias("d").join(
        df_weather.alias("w"),
        on="timestamp_utc",
        how="full"
    ).select(
        # Natural keys & Timestamps
        F.coalesce(F.col("d.respondent"), F.lit(ba_code)).alias("respondent"),
        F.col("timestamp_utc"),
        F.coalesce(F.col("d.timestamp_local"), F.col("w.timestamp_local")).alias("timestamp_local"),
        F.to_date(F.coalesce(F.col("d.timestamp_local"), F.col("w.timestamp_local"))).alias("date_local"),

        # Temporal Components (Based on Texas Local Central Time)
        F.hour(F.coalesce(F.col("d.timestamp_local"), F.col("w.timestamp_local"))).alias("hour_local"),
        F.dayofweek(F.coalesce(F.col("d.timestamp_local"), F.col("w.timestamp_local"))).alias("day_of_week_local"),
        F.dayofmonth(F.coalesce(F.col("d.timestamp_local"), F.col("w.timestamp_local"))).alias("day_of_month_local"),
        F.month(F.coalesce(F.col("d.timestamp_local"), F.col("w.timestamp_local"))).alias("month_local"),
        F.year(F.coalesce(F.col("d.timestamp_local"), F.col("w.timestamp_local"))).alias("year_local"),
        F.dayofyear(F.coalesce(F.col("d.timestamp_local"), F.col("w.timestamp_local"))).alias("day_of_year_local"),

        # Demand Availability State Distinction
        F.when(F.col("d.actual_demand_mw").isNotNull(), F.lit("AVAILABLE"))
         .when(F.col("timestamp_utc") >= (F.current_timestamp() - F.expr("INTERVAL 48 HOURS")), F.lit("NOT_YET_PUBLISHED"))
         .otherwise(F.lit("MISSING_HISTORICAL")).alias("demand_availability_status"),

        # Weather Availability State Distinction
        F.when(F.col("w.weighted_temperature_c").isNotNull(), F.lit("AVAILABLE"))
         .otherwise(F.lit("UNAVAILABLE")).alias("weather_availability_status"),

        # Electricity Metrics & Quality Flags
        F.col("d.actual_demand_mw"),
        F.col("d.day_ahead_forecast_mw"),
        F.when(F.col("d.actual_demand_mw").isNotNull(), F.col("d.is_valid_demand")).alias("is_valid_demand"),
        F.col("d.anomaly_flag"),

        # Regional Weighted Weather Indices
        F.col("w.weighted_temperature_c"),
        F.col("w.weighted_temperature_f"),
        F.col("w.weighted_apparent_temperature_c"),
        F.col("w.weighted_apparent_temperature_f"),
        F.col("w.weighted_dew_point_c"),
        F.col("w.weighted_relative_humidity_pct"),
        F.col("w.weighted_precipitation_mm"),
        F.col("w.weighted_wind_speed_ms"),
        F.col("w.weighted_solar_irradiance_wm2"),
        F.col("w.cdh_65"),
        F.col("w.hdh_65"),

        # Representative Metro Center Temps
        F.col("w.temp_houston_c"),
        F.col("w.temp_dfw_c"),
        F.col("w.temp_san_antonio_c"),
        F.col("w.temp_austin_c")
    )

    # 2. Attach Calendar Reference Dimensions
    df_conformed = df_grid_weather.alias("gw").join(
        df_calendar.alias("c"),
        F.date_format(F.col("gw.date_local"), "yyyy-MM-dd") == F.col("c.date"),
        how="left"
    ).select(
        F.col("gw.*"),
        F.coalesce(F.col("c.is_weekend"), F.col("gw.day_of_week_local").isin(1, 7)).alias("is_weekend"),
        F.coalesce(F.col("c.is_holiday"), F.lit(False)).alias("is_holiday"),
        F.col("c.holiday_name")
    ).withColumn("_conformed_at_utc", F.current_timestamp())

    return df_conformed


def validate_conformed_candidate(df_conformed, df_calendar):
    """
    Enforces row-level semantic checks and one natural-key uniqueness check
    on the impacted candidate. Input tables own their source-level contracts.
    """
    quality = df_conformed.agg(
        F.count(F.lit(1)).alias("rows"),
        F.sum(F.when(F.col("respondent").isNull() | F.col("timestamp_utc").isNull(), 1).otherwise(0)).alias("null_keys"),
        F.sum(F.when(F.col("demand_availability_status").isNull(), 1).otherwise(0)).alias("null_status"),
        F.sum(F.when(F.col("actual_demand_mw").isNull() & F.col("is_valid_demand").isNotNull(), 1).otherwise(0)).alias("invalid_flags"),
        F.sum(F.when(
            (F.col("demand_availability_status") == "NOT_YET_PUBLISHED") &
            (F.col("timestamp_utc") < (F.current_timestamp() - F.expr("INTERVAL 72 HOURS"))), 1
        ).otherwise(0)).alias("old_pending"),
    ).first()
    if not quality["rows"]:
        print("No conformed source rows in the correction window; the idempotent MERGE will be a no-op.")
        return
    assert not quality["null_keys"], f"Integrity Failure: {quality['null_keys']} rows have null keys."
    assert not quality["null_status"], f"Integrity Failure: {quality['null_status']} rows have null availability status."
    assert not quality["invalid_flags"], f"Integrity Failure: {quality['invalid_flags']} absent-demand rows have validity flags."
    assert not quality["old_pending"], f"Semantic Failure: {quality['old_pending']} old rows are marked not-yet-published."

    duplicate_key = df_conformed.groupBy("respondent", "timestamp_utc").count().filter("count > 1").limit(1).count()
    assert duplicate_key == 0, "Integrity Failure: duplicate natural keys in conformed candidate."
    max_data_date = df_conformed.agg(F.max(F.to_date("timestamp_local"))).first()[0]
    max_calendar_date = df_calendar.agg(F.max(F.to_date("date"))).first()[0]
    assert max_calendar_date > max_data_date, f"Calendar ends at {max_calendar_date}; data reaches {max_data_date}."
    print(f"Pre-merge quality checks passed for {quality['rows']:,} impacted rows.")


def upsert_conformed_table(df_conformed, target_table, spark_session):
    """
    Executes idempotent Delta MERGE with null-safe comparison (<=>)
    to handle updates and restatements without unnecessary Parquet file churn.
    """
    if not spark_session.catalog.tableExists(target_table):
        df_conformed.write \
            .format("delta") \
            .mode("overwrite") \
            .saveAsTable(target_table)
        print(f"Initialized Silver conformed table {target_table}.")
    else:
        delta_target = DeltaTable.forName(spark_session, target_table)
        target_cols = [c.name for c in delta_target.toDF().schema]

        # Schema Migration Check:
        # If legacy business metrics (e.g. eia_day_ahead_ape_pct, day_ahead_error_mw) exist in target,
        # overwrite schema to align with Medallion rules (business metrics moved to Gold).
        legacy_cols = [c for c in ["eia_day_ahead_ape_pct", "day_ahead_error_mw"] if c in target_cols]
        if legacy_cols:
            print(f"Migrating Silver conformed schema: removing {legacy_cols} from {target_table}...")
            df_conformed.write \
                .format("delta") \
                .mode("overwrite") \
                .option("overwriteSchema", "true") \
                .saveAsTable(target_table)
            print(f"Successfully migrated and overwritten {target_table} to clean Silver schema.")
        else:
            merge_condition = (
                "target.respondent = source.respondent "
                "AND target.timestamp_utc = source.timestamp_utc"
            )
            if cutoff_utc is not None:
                merge_condition += f" AND target.timestamp_utc >= TIMESTAMP '{cutoff_utc.strftime('%Y-%m-%d %H:%M:%S')}'"

            # Exclude keys and processing timestamp from change detection to prevent unconditional file churn (H-01)
            business_cols = [c for c in df_conformed.columns if c not in ["respondent", "timestamp_utc", "_conformed_at_utc"]]
            update_condition = " OR ".join([f"NOT (target.{c} <=> source.{c})" for c in business_cols])

            # When matched and business columns changed, update all columns including the processing timestamp
            update_set = {col: f"source.{col}" for col in df_conformed.columns if col not in ["respondent", "timestamp_utc"]}
            insert_set = {col: f"source.{col}" for col in df_conformed.columns}

            delta_target.alias("target").merge(
                df_conformed.alias("source"),
                merge_condition
            ).whenMatchedUpdate(
                condition=update_condition,
                set=update_set
            ).whenNotMatchedInsert(
                values=insert_set
            ).execute()
            print(f"Idempotent MERGE successfully executed on {target_table}.")


# COMMAND ----------
# DBTITLE 1,Execute Stage 1: Build Conformed Candidate Dataset
df_conformed = build_conformed_dataset(df_demand, df_weather, df_calendar, ba_code=BA_CODE)

# COMMAND ----------
# DBTITLE 1,Execute Stage 2: Pre-Merge Quality Gate Assertions
validate_conformed_candidate(df_conformed, df_calendar)

# COMMAND ----------
# DBTITLE 1,Execute Stage 3: Idempotent Delta Upsert
upsert_conformed_table(df_conformed, TARGET_TABLE, spark)
