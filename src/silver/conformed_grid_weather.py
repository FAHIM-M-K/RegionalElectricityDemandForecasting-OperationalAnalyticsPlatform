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
# MAGIC 4. **Idempotent Delta MERGE:** Null-safe upsert on `(respondent, timestamp_utc)`.
# MAGIC
# MAGIC ### Design Decisions:
# MAGIC - **No business metrics here.** `day_ahead_error_mw` and `eia_day_ahead_ape_pct` belong in Gold (analytical derivations, not conformance).
# MAGIC - **BA code from config.** Weather-only rows get their respondent from `ercot_config.json`, not a hardcoded string.
# MAGIC - **`is_valid_demand` stays NULL for absent rows.** NULL = 'no data', False = 'data exists but is invalid'. Silver doesn't interpret absence.

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
# DBTITLE 1,Join Demand and Weather on timestamp_utc (Full Outer: Supports Historical Truth & Forecast Horizon)
# Informed by profiling: Weather has 50,640 hours while EIA has 50,626 hours (EIA publication lag ~14h).
# Using a full outer join preserves all historical truth while keeping upcoming weather hours available
# for downstream day-ahead forecasting in Gold.
df_grid_weather = df_demand.alias("d").join(
    df_weather.alias("w"),
    on="timestamp_utc",
    how="full"
).select(
    # Natural keys & Timestamps
    # Weather-only rows (future hours without EIA data) get their BA code from config,
    # not a hardcoded literal. This is the single source of truth for region identity.
    F.coalesce(F.col("d.respondent"), F.lit(BA_CODE)).alias("respondent"),
    F.col("timestamp_utc"),
    F.coalesce(F.col("d.timestamp_local"), F.col("w.timestamp_local")).alias("timestamp_local"),
    F.to_date(F.coalesce(F.col("d.timestamp_local"), F.col("w.timestamp_local"))).alias("date_local"),
    
    # Temporal Components (Based on Texas Local Central Time)
    F.hour(F.coalesce(F.col("d.timestamp_local"), F.col("w.timestamp_local"))).alias("hour_local"),
    F.dayofweek(F.coalesce(F.col("d.timestamp_local"), F.col("w.timestamp_local"))).alias("day_of_week_local"), # 1=Sunday, 7=Saturday
    F.dayofmonth(F.coalesce(F.col("d.timestamp_local"), F.col("w.timestamp_local"))).alias("day_of_month_local"),
    F.month(F.coalesce(F.col("d.timestamp_local"), F.col("w.timestamp_local"))).alias("month_local"),
    F.year(F.coalesce(F.col("d.timestamp_local"), F.col("w.timestamp_local"))).alias("year_local"),
    F.dayofyear(F.coalesce(F.col("d.timestamp_local"), F.col("w.timestamp_local"))).alias("day_of_year_local"),
    
    # Demand Availability State Distinction (Distinguishes actuals available vs publication lag vs historical missing)
    F.when(F.col("d.actual_demand_mw").isNotNull(), F.lit("AVAILABLE"))
     .when(F.col("timestamp_utc") >= (F.current_timestamp() - F.expr("INTERVAL 48 HOURS")), F.lit("NOT_YET_PUBLISHED"))
     .otherwise(F.lit("MISSING_HISTORICAL")).alias("demand_availability_status"),

    # Weather Availability State Distinction
    F.when(F.col("w.weighted_temperature_c").isNotNull(), F.lit("AVAILABLE"))
     .otherwise(F.lit("UNAVAILABLE")).alias("weather_availability_status"),
    
    # Electricity Metrics & Quality Flags
    # is_valid_demand stays NULL for weather-only rows: NULL = 'no data to validate',
    # False = 'data exists but failed physical bounds'. Silver doesn't interpret absence.
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

# COMMAND ----------
# DBTITLE 1,Join Calendar & Holiday Reference Dimension
# Join calendar dimensions using date_local formatted as "yyyy-MM-dd"
df_conformed = df_grid_weather.alias("gw").join(
    df_calendar.alias("c"),
    F.date_format(F.col("gw.date_local"), "yyyy-MM-dd") == F.col("c.date"),
    how="left"
).select(
    F.col("gw.*"),
    F.coalesce(F.col("c.is_weekend"), F.col("gw.day_of_week_local").isin(1, 7)).alias("is_weekend"),
    F.coalesce(F.col("c.is_holiday"), F.lit(False)).alias("is_holiday"),
    F.col("c.holiday_name")
) \
.withColumn("_conformed_at_utc", F.current_timestamp())

# COMMAND ----------
# DBTITLE 1,Pre-Merge Structural & Cardinality Assertions
# 1. Zero null primary keys
null_keys_count = df_conformed.filter(
    F.col("respondent").isNull() | F.col("timestamp_utc").isNull()
).count()
assert null_keys_count == 0, f"Integrity Failure: Found {null_keys_count} records with null primary keys."

# 2. Strict deduplication check on target natural key
dupe_count = df_conformed.count() - df_conformed.dropDuplicates(["respondent", "timestamp_utc"]).count()
assert dupe_count == 0, f"Integrity Failure: Found {dupe_count} duplicate timestamp records."

# 3. Join Cardinality Check: Ensure exact 1:1 match with zero cartesian explosion
# Total conformed rows must exactly equal the number of distinct timestamps across weather and demand
total_conformed_records = df_conformed.count()
demand_utc_count = df_demand.select("timestamp_utc").distinct().count()
weather_utc_count = df_weather.select("timestamp_utc").distinct().count()
expected_max_rows = max(demand_utc_count, weather_utc_count)
assert total_conformed_records == expected_max_rows, (
    f"Cardinality Explosion: Conformed rows ({total_conformed_records}) does not equal expected max ({expected_max_rows}). Check for duplicate timestamps!"
)

# 4. DST Transition Safety Check: Ensure no duplicate UTC timestamps during DST transitions
# Fall back: Local 01:00 or 02:00 repeats, but UTC must remain strictly distinct
dst_check = df_conformed.groupBy("timestamp_utc").count().filter("count > 1").count()
assert dst_check == 0, f"DST Safety Failure: Found {dst_check} duplicate UTC hours."

print(f"Silver Conformed Integration Passed:")
print(f"  - Total conformed hours: {total_conformed_records:,}")
print(f"  - Strict 1:1 join cardinality verified (no many-to-many explosion)")
print(f"  - UTC chronological uniqueness verified (DST transitions safe)")

# COMMAND ----------
# DBTITLE 1,Idempotent Delta Upsert with Null-Safe Comparison
if not spark.catalog.tableExists(TARGET_TABLE):
    df_conformed.write \
        .format("delta") \
        .mode("overwrite") \
        .saveAsTable(TARGET_TABLE)
    print(f"Initialized Silver conformed table {TARGET_TABLE} with {total_conformed_records} records.")
else:
    delta_target = DeltaTable.forName(spark, TARGET_TABLE)
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
    print(f"Idempotent MERGE successfully executed on {TARGET_TABLE}.")

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
