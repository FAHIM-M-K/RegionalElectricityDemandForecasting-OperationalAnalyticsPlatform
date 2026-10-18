# Databricks notebook source
# MAGIC %md
# MAGIC # Silver Layer: Multi-Site Weather Cleaning & Load-Weighted Conformance
# MAGIC
# MAGIC **Input Table:** `main.bronze.weather_hourly_raw`  
# MAGIC **Output Table:** `main.silver.weather_hourly_conformed`  
# MAGIC
# MAGIC ### Core Transformations:
# MAGIC 1. **Timezone Standardization:** Parse Open-Meteo `time_utc` (`yyyy-MM-dd'T'HH:mm`) to timestamp types for UTC (`timestamp_utc`) and local Central Time (`timestamp_local`, `America/Chicago`).
# MAGIC 2. **Fixed Representative-Site Aggregation:** A project-configured weighted average across four metro points. These weights are not measured ERCOT load shares or population estimates:
# MAGIC    - Houston: 35%
# MAGIC    - Dallas-Fort Worth: 35%
# MAGIC    - San Antonio: 15%
# MAGIC    - Austin: 15%
# MAGIC 3. **Thermodynamic Feature Engineering:**
# MAGIC    - Temperature conversions (Celsius -> Fahrenheit).
# MAGIC    - Cooling Degree Hours (`cdh_65` = max(0, temp_f - 65)).
# MAGIC    - Heating Degree Hours (`hdh_65` = max(0, 65 - temp_f)).
# MAGIC 4. **Incremental Idempotency:** Null-safe Delta `MERGE` on `timestamp_utc`.

# COMMAND ----------
# DBTITLE 1,Imports & Configuration
import os
import json
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, TimestampType, StringType
from delta.tables import DeltaTable

CATALOG = "main"
SOURCE_TABLE = f"{CATALOG}.bronze.weather_hourly_raw"
TARGET_TABLE = f"{CATALOG}.silver.weather_hourly_conformed"

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
REGION_TIMEZONE = config.get("region", {}).get("timezone", "America/Chicago")
weather_stations = config.get("weather", {}).get("stations", [])

station_weights = {s["station_id"]: float(s["weight"]) for s in weather_stations}
assert abs(sum(station_weights.values()) - 1.0) < 1e-9, f"Station weights must sum to 1.0, got {sum(station_weights.values())}"
print(f"Loaded station load weights: {station_weights}")

# COMMAND ----------
# DBTITLE 1,Read Bronze Source & Standardize Timestamps
assert spark.catalog.tableExists(SOURCE_TABLE), f"Source table {SOURCE_TABLE} does not exist!"

df_bronze_weather = spark.table(SOURCE_TABLE)

# Open-Meteo time_utc format is "yyyy-MM-dd'T'HH:mm" (e.g., "2021-01-01T00:00")
df_time_std = df_bronze_weather \
    .withColumn("timestamp_utc", F.to_timestamp(F.col("time_utc"), "yyyy-MM-dd'T'HH:mm")) \
    .withColumn("timestamp_local", F.from_utc_timestamp(F.col("timestamp_utc"), REGION_TIMEZONE))

# COMMAND ----------
# DBTITLE 1,Attach Station Load Weights
# Map weights to each station row
weight_expr = F.expr(
    "CASE " + " ".join([f"WHEN station_id = '{st}' THEN {w}" for st, w in station_weights.items()]) + " ELSE 0.0 END"
)

df_weighted_base = df_time_std.withColumn("station_weight", weight_expr.cast(DoubleType()))

# COMMAND ----------
# DBTITLE 1,Compute Regional Population/Load-Weighted Aggregates
# Calculate dynamically normalized metrics across the four configured representative metro points per hour.
# The fixed weights are a project heuristic, not observed load shares.
# Dynamically renormalizes weights if an individual station observation is missing to prevent artificial temperature drops
df_regional = df_weighted_base.groupBy("timestamp_utc", "timestamp_local").agg(
    # Effective weight sum for temperature
    F.sum(F.when(F.col("temperature_2m").isNotNull(), F.col("station_weight")).otherwise(0.0)).alias("temp_weight_sum"),
    
    # Weighted average temperature (°C) normalized by available station weights
    F.round(
        F.sum(F.when(F.col("temperature_2m").isNotNull(), F.col("temperature_2m") * F.col("station_weight"))) /
        F.nullif(F.sum(F.when(F.col("temperature_2m").isNotNull(), F.col("station_weight"))), F.lit(0.0)),
        2
    ).alias("weighted_temperature_c"),
    
    # Weighted apparent temperature (°C, heat index / wind chill)
    F.round(
        F.sum(F.when(F.col("apparent_temperature").isNotNull(), F.col("apparent_temperature") * F.col("station_weight"))) /
        F.nullif(F.sum(F.when(F.col("apparent_temperature").isNotNull(), F.col("station_weight"))), F.lit(0.0)),
        2
    ).alias("weighted_apparent_temperature_c"),
    
    # Weighted dew point (°C)
    F.round(
        F.sum(F.when(F.col("dew_point_2m").isNotNull(), F.col("dew_point_2m") * F.col("station_weight"))) /
        F.nullif(F.sum(F.when(F.col("dew_point_2m").isNotNull(), F.col("station_weight"))), F.lit(0.0)),
        2
    ).alias("weighted_dew_point_c"),
    
    # Weighted relative humidity (%)
    F.round(
        F.sum(F.when(F.col("relative_humidity_2m").isNotNull(), F.col("relative_humidity_2m") * F.col("station_weight"))) /
        F.nullif(F.sum(F.when(F.col("relative_humidity_2m").isNotNull(), F.col("station_weight"))), F.lit(0.0)),
        1
    ).alias("weighted_relative_humidity_pct"),
    
    # Weighted precipitation (mm)
    F.round(
        F.sum(F.when(F.col("precipitation").isNotNull(), F.col("precipitation") * F.col("station_weight"))) /
        F.nullif(F.sum(F.when(F.col("precipitation").isNotNull(), F.col("station_weight"))), F.lit(0.0)),
        2
    ).alias("weighted_precipitation_mm"),
    
    # Weighted wind speed (m/s)
    F.round((
        F.sum(F.when(F.col("wind_speed_10m").isNotNull(), F.col("wind_speed_10m") * F.col("station_weight"))) /
        F.nullif(F.sum(F.when(F.col("wind_speed_10m").isNotNull(), F.col("station_weight"))), F.lit(0.0))
    ) / 3.6, 2
    ).alias("weighted_wind_speed_ms"),
    
    # Weighted direct solar irradiance (W/m²)
    F.round(
        F.sum(F.when(F.col("direct_normal_irradiance").isNotNull(), F.col("direct_normal_irradiance") * F.col("station_weight"))) /
        F.nullif(F.sum(F.when(F.col("direct_normal_irradiance").isNotNull(), F.col("station_weight"))), F.lit(0.0)),
        1
    ).alias("weighted_solar_irradiance_wm2"),
    
    # Station-specific metrics for granular visibility (Houston and DFW are the two giant load centers)
    F.max(F.when(F.col("station_id") == "houston", F.col("temperature_2m"))).alias("temp_houston_c"),
    F.max(F.when(F.col("station_id") == "dfw", F.col("temperature_2m"))).alias("temp_dfw_c"),
    F.max(F.when(F.col("station_id") == "san_antonio", F.col("temperature_2m"))).alias("temp_san_antonio_c"),
    F.max(F.when(F.col("station_id") == "austin", F.col("temperature_2m"))).alias("temp_austin_c"),
    
    # Audit tracking & completeness
    F.count("station_id").alias("reporting_stations_count"),
    F.max("_ingested_at_utc").alias("_bronze_ingested_at_utc")
)

# COMMAND ----------
# DBTITLE 1,Thermodynamic Features: Degree Hours & Conversions
# Degree hours base reference is 65°F (18.33°C) — industry gold standard for Texas thermal balance
df_conformed = df_regional \
    .withColumn(
        "weighted_temperature_f",
        F.round((F.col("weighted_temperature_c") * 9.0 / 5.0) + 32.0, 2)
    ) \
    .withColumn(
        "weighted_apparent_temperature_f",
        F.round((F.col("weighted_apparent_temperature_c") * 9.0 / 5.0) + 32.0, 2)
    ) \
    .withColumn(
        "cdh_65",
        F.when(F.col("weighted_temperature_f") > 65.0, F.round(F.col("weighted_temperature_f") - 65.0, 2)).otherwise(0.0)
    ) \
    .withColumn(
        "hdh_65",
        F.when(F.col("weighted_temperature_f") < 65.0, F.round(65.0 - F.col("weighted_temperature_f"), 2)).otherwise(0.0)
    ) \
    .withColumn(
        "is_complete_weather_reading",
        (F.col("reporting_stations_count") == len(station_weights)) & (F.col("temp_weight_sum") >= 0.99)
    ) \
    .withColumn("_transformed_at_utc", F.current_timestamp())

# COMMAND ----------
# DBTITLE 1,Pre-Merge Integrity Assertions
null_time_count = df_conformed.filter(F.col("timestamp_utc").isNull()).count()
assert null_time_count == 0, f"Integrity Failure: Found {null_time_count} rows with null timestamp_utc."

dupe_time_count = df_conformed.count() - df_conformed.dropDuplicates(["timestamp_utc"]).count()
assert dupe_time_count == 0, f"Integrity Failure: Found {dupe_time_count} duplicate hourly timestamps."

# Verify zero unrepresented hours (at least 1 station must report)
zero_station_hours = df_conformed.filter(F.col("reporting_stations_count") == 0).count()
assert zero_station_hours == 0, f"Completeness Failure: Found {zero_station_hours} hours with 0 reporting stations."

incomplete_station_hours = df_conformed.filter(F.col("is_complete_weather_reading") == False).count()
if incomplete_station_hours > 0:
    print(f"Notice: {incomplete_station_hours:,} hours have partial station coverage (< {len(station_weights)} stations). Dynamic weight normalization preserved valid thermal scaling.")

total_hours = df_conformed.count()
print(f"Silver Weather Pre-Merge Validation: Total {total_hours:,} hourly regional indices verified.")

# COMMAND ----------
# DBTITLE 1,Idempotent Delta Upsert with Full Business Attribute Comparison
if not spark.catalog.tableExists(TARGET_TABLE):
    df_conformed.write \
        .format("delta") \
        .mode("overwrite") \
        .saveAsTable(TARGET_TABLE)
    print(f"Initialized Silver table {TARGET_TABLE} with {total_hours:,} records.")
else:
    delta_target = DeltaTable.forName(spark, TARGET_TABLE)
    merge_condition = "target.timestamp_utc = source.timestamp_utc"
    
    # Check all conformed weather attributes to ensure restatements propagate completely
    weather_business_cols = [
        "weighted_temperature_c", "weighted_apparent_temperature_c", "weighted_dew_point_c",
        "weighted_relative_humidity_pct", "weighted_precipitation_mm", "weighted_wind_speed_ms",
        "weighted_solar_irradiance_wm2", "weighted_temperature_f", "weighted_apparent_temperature_f",
        "cdh_65", "hdh_65", "temp_houston_c", "temp_dfw_c", "temp_san_antonio_c", "temp_austin_c",
        "reporting_stations_count", "is_complete_weather_reading"
    ]
    update_condition = " OR ".join([f"NOT (target.{c} <=> source.{c})" for c in weather_business_cols])
    
    delta_target.alias("target").merge(
        df_conformed.alias("source"),
        merge_condition
    ).whenMatchedUpdateAll(
        condition=update_condition
    ).whenNotMatchedInsertAll(
    ).execute()
    print(f"Idempotent MERGE successfully executed on {TARGET_TABLE} (all business attributes checked).")

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT 
# MAGIC     count(*) as total_hours,
# MAGIC     min(timestamp_local) as min_local_time,
# MAGIC     max(timestamp_local) as max_local_time,
# MAGIC     round(avg(weighted_temperature_f), 2) as avg_regional_temp_f,
# MAGIC     round(min(weighted_temperature_f), 2) as min_regional_temp_f,
# MAGIC     round(max(weighted_temperature_f), 2) as max_regional_temp_f,
# MAGIC     round(sum(cdh_65), 1) as total_cdh,
# MAGIC     round(sum(hdh_65), 1) as total_hdh
# MAGIC FROM silver.weather_hourly_conformed;
