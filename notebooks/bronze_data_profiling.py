# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze Layer Data Profiling & Diagnostic Analysis
# MAGIC
# MAGIC **Purpose:** Perform exploratory data profiling on raw Bronze tables (`eia_hourly_raw` and `weather_hourly_raw`) to inspect real-world distributions, missingness, timestamp formatting, anomalies, and cross-source alignment before finalizing the Silver layer.

# COMMAND ----------
# DBTITLE 1,Environment & Catalog Setup
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, TimestampType
from pyspark.sql.window import Window

CATALOG = "main"
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql("USE SCHEMA bronze")

df_eia = spark.table("main.bronze.eia_hourly_raw")
df_weather = spark.table("main.bronze.weather_hourly_raw")
df_calendar = spark.table("main.bronze.calendar_reference")

print(f"Bronze EIA Raw row count: {df_eia.count():,}")
print(f"Bronze Weather Raw row count: {df_weather.count():,}")
print(f"Bronze Calendar Reference row count: {df_calendar.count():,}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. EIA-930 Demand & Forecast Profiling

# COMMAND ----------
# DBTITLE 1,EIA Schema & Sample Rows
display(df_eia.limit(10))

# COMMAND ----------
# DBTITLE 1,EIA Series Types & Row Counts
# Verify series types present in Bronze (expected: only 'D' and 'DF')
display(
    df_eia.groupBy("type", "type_name", "value_units")
    .agg(
        F.count("*").alias("record_count"),
        F.min("period").alias("earliest_period"),
        F.max("period").alias("latest_period")
    )
)

# COMMAND ----------
# DBTITLE 1,EIA Summary Statistics & Distribution (Actual D vs Forecast DF)
df_eia_stats = df_eia.withColumn("value_num", F.col("value").cast(DoubleType()))

display(
    df_eia_stats.groupBy("type")
    .agg(
        F.count("*").alias("total_rows"),
        F.count("value_num").alias("non_null_rows"),
        F.sum(F.when(F.col("value_num").isNull(), 1).otherwise(0)).alias("null_count"),
        F.round(F.min("value_num"), 2).alias("min_mw"),
        F.round(F.expr("percentile_approx(value_num, 0.01)"), 2).alias("p01_mw"),
        F.round(F.expr("percentile_approx(value_num, 0.50)"), 2).alias("median_mw"),
        F.round(F.avg("value_num"), 2).alias("mean_mw"),
        F.round(F.expr("percentile_approx(value_num, 0.99)"), 2).alias("p99_mw"),
        F.round(F.max("value_num"), 2).alias("max_mw"),
        F.round(F.stddev("value_num"), 2).alias("stddev_mw")
    )
)

# COMMAND ----------
# DBTITLE 1,EIA Anomaly & Boundary Scan (Negatives, Spikes, Zeros)
# Real-world grid boundaries check for ERCOT
display(
    df_eia_stats.filter(
        (F.col("value_num") <= 0.0) |
        (F.col("value_num") < 15000.0) |
        (F.col("value_num") > 95000.0) |
        (F.col("value_num").isNull())
    ).select(
        "respondent", "period", "type", "value_num", "_raw_payload_path"
    ).orderBy("period")
)

# COMMAND ----------
# DBTITLE 1,EIA Timestamp Gap Analysis (Continuous Hourly Check)
# Check if there are skipped hours in the time series grid
df_eia_utc = df_eia.filter(F.col("type") == "D") \
    .withColumn("ts_utc", F.to_timestamp(F.col("period"), "yyyy-MM-dd'T'HH"))

window_spec = Window.orderBy("ts_utc")
df_gaps = df_eia_utc.withColumn("prev_ts", F.lag("ts_utc", 1).over(window_spec)) \
    .withColumn("hours_diff", (F.unix_timestamp("ts_utc") - F.unix_timestamp("prev_ts")) / 3600) \
    .filter(F.col("hours_diff") > 1.0)

gap_count = df_gaps.count()
print(f"Total time gaps (>1 hour) detected in EIA Actual Demand: {gap_count}")
if gap_count > 0:
    display(df_gaps.select("prev_ts", "ts_utc", "hours_diff").limit(20))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Open-Meteo Multi-Site Weather Profiling

# COMMAND ----------
# DBTITLE 1,Weather Schema & Sample Rows
display(df_weather.limit(10))

# COMMAND ----------
# DBTITLE 1,Weather Station Counts & Timestamp Ranges
display(
    df_weather.groupBy("station_id")
    .agg(
        F.count("*").alias("total_hours"),
        F.min("time_utc").alias("min_time"),
        F.max("time_utc").alias("max_time"),
        F.countDistinct("time_utc").alias("distinct_hours")
    )
)

# COMMAND ----------
# DBTITLE 1,Weather Completeness & Null Audit
display(
    df_weather.groupBy("station_id")
    .agg(
        F.sum(F.when(F.col("temperature_2m").isNull(), 1).otherwise(0)).alias("null_temp"),
        F.sum(F.when(F.col("relative_humidity_2m").isNull(), 1).otherwise(0)).alias("null_humidity"),
        F.sum(F.when(F.col("apparent_temperature").isNull(), 1).otherwise(0)).alias("null_apparent_temp"),
        F.sum(F.when(F.col("wind_speed_10m").isNull(), 1).otherwise(0)).alias("null_wind"),
        F.sum(F.when(F.col("direct_normal_irradiance").isNull(), 1).otherwise(0)).alias("null_solar"),
        F.sum(F.when(F.col("precipitation").isNull(), 1).otherwise(0)).alias("null_precip")
    )
)

# COMMAND ----------
# DBTITLE 1,Meteorological Statistics Across Texas Stations
display(
    df_weather.groupBy("station_id")
    .agg(
        F.round(F.min("temperature_2m"), 2).alias("min_temp_c"),
        F.round(F.avg("temperature_2m"), 2).alias("avg_temp_c"),
        F.round(F.max("temperature_2m"), 2).alias("max_temp_c"),
        F.round(F.min("apparent_temperature"), 2).alias("min_apparent_temp_c"),
        F.round(F.max("apparent_temperature"), 2).alias("max_apparent_temp_c"),
        F.round(F.max("wind_speed_10m"), 2).alias("max_wind_ms"),
        F.round(F.max("direct_normal_irradiance"), 2).alias("max_solar_wm2")
    )
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Cross-Source Join & Alignment Analysis (EIA vs Weather)

# COMMAND ----------
# DBTITLE 1,Hourly Join Coverage Check (EIA Demand vs Open-Meteo)
# Standardize both to timestamp_utc and test inner, left, and right join alignment
df_eia_ts = df_eia.filter(F.col("type") == "D") \
    .select(F.to_timestamp(F.col("period"), "yyyy-MM-dd'T'HH").alias("ts_utc"), "value")

df_w_houston_ts = df_weather.filter(F.col("station_id") == "houston") \
    .select(F.to_timestamp(F.col("time_utc"), "yyyy-MM-dd'T'HH:mm").alias("ts_utc"))

eia_only = df_eia_ts.join(df_w_houston_ts, on="ts_utc", how="left_anti").count()
weather_only = df_w_houston_ts.join(df_eia_ts, on="ts_utc", how="left_anti").count()
matched = df_eia_ts.join(df_w_houston_ts, on="ts_utc", how="inner").count()

print("--- Cross-Source Join Coverage ---")
print(f"Matched Hours (Both Present): {matched:,}")
print(f"Hours in EIA only (Missing Weather): {eia_only:,}")
print(f"Hours in Weather only (Missing EIA): {weather_only:,}")

# COMMAND ----------
# DBTITLE 1,Correlation Sample: Temperature vs Electricity Demand
df_paired = df_eia_ts.join(
    df_weather.filter(F.col("station_id") == "houston"),
    on="ts_utc",
    how="inner"
).select(
    "ts_utc",
    F.col("value").cast(DoubleType()).alias("demand_mw"),
    F.col("temperature_2m").alias("houston_temp_c")
)

# Display a sample to visually inspect the relationship
display(df_paired.orderBy("ts_utc").limit(100))
