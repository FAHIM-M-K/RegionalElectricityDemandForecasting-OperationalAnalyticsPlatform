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
# MAGIC 4. **Pre-Write Data Quality Gate:** Modular assertions verifying primary keys, set-theoretic 1:1 join cardinality, DST transitions, missingness semantics, and calendar horizon.
# MAGIC 5. **Idempotent Delta MERGE & Post-Write Audit:** Null-safe upsert with read-back verification to protect downstream Gold consumption.
# MAGIC
# MAGIC ### Design Decisions:
# MAGIC - **No business metrics here.** `day_ahead_error_mw` and `eia_day_ahead_ape_pct` belong in Gold (analytical derivations, not conformance).
# MAGIC - **BA code from config.** Weather-only rows get their respondent from `ercot_config.json`, not a hardcoded string.
# MAGIC - **`is_valid_demand` stays NULL for absent rows.** NULL = 'no data', False = 'data exists but is invalid'. Silver doesn't interpret absence.
# MAGIC - **Failure Semantics:** Delta transactions commit upon `execute()`. Post-write verification halts this DAG task to guarantee downstream Gold pipelines never ingest unverified data.

# COMMAND ----------
# DBTITLE 1,Imports & Configuration
import os
import json
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, BooleanType, IntegerType
from delta.tables import DeltaTable

CATALOG = "main"
DEMAND_TABLE = f"{CATALOG}.silver.electricity_demand_hourly"
WEATHER_TABLE = f"{CATALOG}.silver.weather_hourly_conformed"
CALENDAR_TABLE = f"{CATALOG}.bronze.calendar_reference"
TARGET_TABLE = f"{CATALOG}.silver.grid_hourly_conformed"

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql("USE SCHEMA silver")

# COMMAND ----------
# DBTITLE 1,Dynamic Configuration Locator
def load_platform_config(filename="ercot_config.json"):
    curr = os.getcwd()
    for _ in range(5):
        test_path = os.path.join(curr, "configs", filename)
        if os.path.exists(test_path):
            with open(test_path, "r", encoding="utf-8") as f:
                print(f"Loaded platform config from {test_path}")
                return json.load(f)
        parent = os.path.dirname(curr)
        if parent == curr:
            break
        curr = parent
        
    repo_candidate = f"/Workspace/Repos/configs/{filename}"
    if os.path.exists(repo_candidate):
        with open(repo_candidate, "r", encoding="utf-8") as f:
            print(f"Loaded platform config from {repo_candidate}")
            return json.load(f)
            
    raise FileNotFoundError(
        f"Critical Configuration Error: Could not locate 'configs/{filename}'. "
        "Ensure the repository is synced properly in Databricks Repos."
    )

config = load_platform_config("ercot_config.json")
BA_CODE = config.get("region", {}).get("code", "ERCO")

# COMMAND ----------
# DBTITLE 1,Verify Source Tables Exist
for tbl in [DEMAND_TABLE, WEATHER_TABLE, CALENDAR_TABLE]:
    assert spark.catalog.tableExists(tbl), f"Required upstream table {tbl} does not exist!"

df_demand = spark.table(DEMAND_TABLE)
df_weather = spark.table(WEATHER_TABLE)
df_calendar = spark.table(CALENDAR_TABLE)

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
        F.col("d.is_valid_demand"),
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


def validate_conformed_candidate(df_conformed, df_demand, df_weather, df_calendar):
    """
    Enforces pre-write data quality gate invariants:
    1. Null primary key absence
    2. Primary key deduplication
    3. Exact set-theoretic join cardinality (zero Cartesian explosion)
    4. Chronological UTC uniqueness across DST transitions
    5. Missing data semantics completeness
    6. Validity flag leakage prevention (NULL when demand absent)
    7. Calendar forward horizon coverage
    """
    # 1. Zero null primary keys
    null_keys = df_conformed.filter(
        F.col("respondent").isNull() | F.col("timestamp_utc").isNull()
    ).count()
    assert null_keys == 0, f"Integrity Failure: Found {null_keys} records with null primary keys."

    # 2. Strict deduplication check on target natural key
    total_conformed_records = df_conformed.count()
    distinct_keys = df_conformed.select("respondent", "timestamp_utc").distinct().count()
    assert total_conformed_records == distinct_keys, (
        f"Integrity Failure: Found {total_conformed_records - distinct_keys} duplicate timestamp records."
    )

    # 3. Set-Theoretic Join Cardinality Check (Issue A Fix):
    # For a full outer join on unique timestamp_utc keys, the result count must exactly equal
    # the count of the distinct UNION of keys from both sources: |A ∪ B|.
    # (max(|A|, |B|) is only valid if one set is a subset of the other, which is not true in general.)
    expected_union_count = df_demand.select("timestamp_utc") \
        .union(df_weather.select("timestamp_utc")) \
        .distinct().count()
    assert total_conformed_records == expected_union_count, (
        f"Cardinality Explosion Failure: Conformed rows ({total_conformed_records:,}) "
        f"does not match expected set union of source timestamps ({expected_union_count:,}). "
        "Check for non-unique join keys in source tables."
    )

    # 4. DST Transition Safety Check: Ensure no duplicate UTC timestamps
    utc_dupes = df_conformed.groupBy("timestamp_utc").count().filter("count > 1").count()
    assert utc_dupes == 0, f"DST Safety Failure: Found {utc_dupes} duplicate UTC hours."

    # 5. Missing Data Semantics
    null_status = df_conformed.filter(F.col("demand_availability_status").isNull()).count()
    assert null_status == 0, f"Integrity Failure: Found {null_status} rows with null demand_availability_status."
    
    future_misclassified = df_conformed.filter(
        (F.col("demand_availability_status") == "NOT_YET_PUBLISHED") &
        (F.col("timestamp_utc") < (F.current_timestamp() - F.expr("INTERVAL 72 HOURS")))
    ).count()
    assert future_misclassified == 0, (
        f"Semantic Failure: {future_misclassified} rows older than 72h marked NOT_YET_PUBLISHED."
    )

    # 6. Validity Flag Leakage Prevention
    invalid_validity_flags = df_conformed.filter(
        (F.col("actual_demand_mw").isNull()) & (F.col("is_valid_demand").isNotNull())
    ).count()
    assert invalid_validity_flags == 0, (
        f"Integrity Failure: {invalid_validity_flags} rows have is_valid_demand set when actual demand is absent."
    )

    # 7. Calendar Horizon Coverage
    max_data_date = df_conformed.select(F.max(F.to_date("timestamp_local"))).first()[0]
    max_calendar_date = df_calendar.select(F.max(F.to_date("date"))).first()[0]
    assert max_calendar_date > max_data_date, (
        f"Calendar Failure: Calendar ends at {max_calendar_date}, cannot support scoring beyond {max_data_date}."
    )
    days_ahead = (max_calendar_date - max_data_date).days
    if days_ahead < 365:
        print(f"  [WARN] Calendar only extends {days_ahead} days ahead. Recommend >= 365 days.")
    else:
        print(f"  - Calendar horizon verified: extends {days_ahead} days beyond current data.")

    print(f"Pre-Merge Quality Gate Passed:")
    print(f"  - Total conformed hours: {total_conformed_records:,}")
    print(f"  - Exact set-union join cardinality verified (|A ∪ B| = {expected_union_count:,})")
    print(f"  - UTC chronological uniqueness verified (DST safe)")
    print(f"  - Missingness semantics and validity flag invariants verified")


def upsert_conformed_table(df_conformed, target_table, spark_session):
    """
    Executes idempotent Delta MERGE with null-safe comparison (<=>)
    to handle updates and restatements without unnecessary Parquet file churn.
    """
    total_records = df_conformed.count()
    if not spark_session.catalog.tableExists(target_table):
        df_conformed.write \
            .format("delta") \
            .mode("overwrite") \
            .saveAsTable(target_table)
        print(f"Initialized Silver conformed table {target_table} with {total_records:,} records.")
    else:
        delta_target = DeltaTable.forName(spark_session, target_table)
        merge_condition = (
            "target.respondent = source.respondent "
            "AND target.timestamp_utc = source.timestamp_utc"
        )
        
        update_condition = (
            "NOT (target.actual_demand_mw <=> source.actual_demand_mw) OR "
            "NOT (target.day_ahead_forecast_mw <=> source.day_ahead_forecast_mw) OR "
            "NOT (target.weighted_temperature_c <=> source.weighted_temperature_c) OR "
            "NOT (target.is_valid_demand <=> source.is_valid_demand) OR "
            "NOT (target.anomaly_flag <=> source.anomaly_flag)"
        )
        
        delta_target.alias("target").merge(
            df_conformed.alias("source"),
            merge_condition
        ).whenMatchedUpdateAll(
            condition=update_condition
        ).whenNotMatchedInsertAll(
        ).execute()
        print(f"Idempotent MERGE successfully executed on {target_table}.")


def audit_persisted_table(target_table, spark_session):
    """
    Post-write verification audit: reads back directly from Delta storage.
    Failing here halts the DAG task to protect downstream Gold pipelines from consuming invalid data.
    """
    df_persisted = spark_session.table(target_table)
    persisted_rows = df_persisted.count()
    distinct_keys = df_persisted.select("respondent", "timestamp_utc").distinct().count()
    null_keys = df_persisted.filter(
        F.col("respondent").isNull() | F.col("timestamp_utc").isNull()
    ).count()

    assert null_keys == 0, f"Post-MERGE Audit Failure: Found {null_keys} null keys in {target_table}!"
    assert persisted_rows == distinct_keys, (
        f"Post-MERGE Audit Failure: Found {persisted_rows - distinct_keys} duplicate keys in {target_table}!"
    )
    print(f"Post-MERGE Read-Back Audit Passed: {persisted_rows:,} records in {target_table}, 0 null keys, 0 duplicates.")

# COMMAND ----------
# DBTITLE 1,Execute Stage 1: Build Conformed Candidate Dataset
df_conformed = build_conformed_dataset(df_demand, df_weather, df_calendar, ba_code=BA_CODE)

# COMMAND ----------
# DBTITLE 1,Execute Stage 2: Pre-Merge Quality Gate Assertions
validate_conformed_candidate(df_conformed, df_demand, df_weather, df_calendar)

# COMMAND ----------
# DBTITLE 1,Execute Stage 3: Idempotent Delta Upsert
upsert_conformed_table(df_conformed, TARGET_TABLE, spark)

# COMMAND ----------
# DBTITLE 1,Execute Stage 4: Post-Merge Storage Audit
audit_persisted_table(TARGET_TABLE, spark)

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT 
# MAGIC     count(*) as total_hours,
# MAGIC     min(timestamp_local) as min_time,
# MAGIC     max(timestamp_local) as max_time,
# MAGIC     round(avg(actual_demand_mw), 2) as avg_demand_mw,
# MAGIC     round(avg(weighted_temperature_f), 2) as avg_regional_temp_f,
# MAGIC     sum(case when is_holiday then 1 else 0 end) as holiday_hours,
# MAGIC     sum(case when demand_availability_status = 'AVAILABLE' then 1 else 0 end) as demand_available_hours,
# MAGIC     sum(case when demand_availability_status = 'NOT_YET_PUBLISHED' then 1 else 0 end) as pending_hours
# MAGIC FROM silver.grid_hourly_conformed;
