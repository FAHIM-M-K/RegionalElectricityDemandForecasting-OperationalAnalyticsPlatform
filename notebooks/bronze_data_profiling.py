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
df_weather_houston = df_weather.filter(F.col("station_id") == "houston") \
    .withColumn("ts_utc", F.to_timestamp(F.col("time_utc"), "yyyy-MM-dd'T'HH:mm"))

df_paired = df_eia_ts.join(
    df_weather_houston,
    on="ts_utc",
    how="inner"
).select(
    "ts_utc",
    F.col("value").cast(DoubleType()).alias("demand_mw"),
    F.col("temperature_2m").alias("houston_temp_c")
)

# Display a sample to visually inspect the relationship
display(df_paired.orderBy("ts_utc").limit(100))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Programmatic Visualizations (Matplotlib & Seaborn)

# COMMAND ----------
# DBTITLE 1,Matplotlib Setup
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

# Apply clean visualization style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11

# COMMAND ----------
# DBTITLE 1,Figure 1: The Thermodynamic U-Curve (Temperature vs. Electricity Demand)
# Sample 10,000 paired rows for lightweight, crisp scatter rendering
sample_pd = df_paired.sample(withReplacement=False, fraction=0.25, seed=42).toPandas()

fig, ax = plt.subplots(figsize=(12, 6))
scatter = ax.scatter(
    sample_pd["houston_temp_c"], 
    sample_pd["demand_mw"], 
    alpha=0.15, 
    color="#1f77b4", 
    edgecolors="none",
    s=20,
    label="Hourly Observations (Sample)"
)

# Add polynomial fit trendline to highlight non-linear AC/heating inflection
if len(sample_pd.dropna()) > 0:
    clean_sample = sample_pd.dropna()
    z = np.polyfit(clean_sample["houston_temp_c"], clean_sample["demand_mw"], 2)
    p = np.poly1d(z)
    x_range = np.linspace(clean_sample["houston_temp_c"].min(), clean_sample["houston_temp_c"].max(), 200)
    ax.plot(x_range, p(x_range), color="#d62728", linewidth=2.5, label="2nd Degree Polynomial Fit")

ax.axvline(18.33, color="#2ca02c", linestyle="--", linewidth=1.5, label="65°F (18.3°C) Thermal Balance Point")
ax.set_title("ERCOT Electricity Demand vs. Temperature (The U-Curve Profile)", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Houston Temperature (°C)", fontsize=12)
ax.set_ylabel("ERCOT Grid Demand (MW)", fontsize=12)
ax.legend(loc="upper center", frameon=True)
ax.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# COMMAND ----------
# DBTITLE 1,Figure 2: Multi-Year Macro Demand Trend & Seasonality (2021–2026)
# Aggregate to daily average, min, and max for clean macroscopic timeline
df_daily = df_eia.filter(F.col("type") == "D") \
    .withColumn("ts_utc", F.to_timestamp(F.col("period"), "yyyy-MM-dd'T'HH")) \
    .withColumn("date", F.to_date("ts_utc")) \
    .withColumn("demand_mw", F.col("value").cast(DoubleType())) \
    .groupBy("date") \
    .agg(
        F.round(F.avg("demand_mw"), 1).alias("avg_demand"),
        F.round(F.min("demand_mw"), 1).alias("min_demand"),
        F.round(F.max("demand_mw"), 1).alias("max_demand")
    ).orderBy("date")

daily_pd = df_daily.toPandas()
daily_pd["date"] = pd.to_datetime(daily_pd["date"])

fig, ax = plt.subplots(figsize=(14, 6))
ax.fill_between(daily_pd["date"], daily_pd["min_demand"], daily_pd["max_demand"], color="#aec7e8", alpha=0.5, label="Daily Min-Max Range")
ax.plot(daily_pd["date"], daily_pd["avg_demand"], color="#1f77b4", linewidth=1.2, label="Daily Mean Demand")

# Mark Texas historical peaks threshold
ax.axhline(85000, color="#d62728", linestyle=":", linewidth=1.5, label="85 GW Summer Peak Level")

ax.set_title("ERCOT System Demand Timeline (2021–2026)", fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("Demand (MW)", fontsize=12)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
ax.legend(loc="upper left", frameon=True)
ax.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# COMMAND ----------
# DBTITLE 1,Figure 3: Diurnal Load Cycle by Season (Hourly 0-23 Profile)
# Extract local Central Time hour and season to visualize the classic double-peak profile
df_diurnal = df_eia.filter(F.col("type") == "D") \
    .withColumn("ts_utc", F.to_timestamp(F.col("period"), "yyyy-MM-dd'T'HH")) \
    .withColumn("ts_local", F.from_utc_timestamp("ts_utc", "America/Chicago")) \
    .withColumn("hour_local", F.hour("ts_local")) \
    .withColumn("month", F.month("ts_local")) \
    .withColumn(
        "season",
        F.when(F.col("month").isin(6, 7, 8), "Summer (Jun-Aug)")
         .when(F.col("month").isin(12, 1, 2), "Winter (Dec-Feb)")
         .otherwise("Shoulder (Spring/Fall)")
    ) \
    .withColumn("demand_mw", F.col("value").cast(DoubleType())) \
    .groupBy("season", "hour_local") \
    .agg(F.round(F.avg("demand_mw"), 1).alias("mean_demand_mw")) \
    .orderBy("season", "hour_local")

diurnal_pd = df_diurnal.toPandas()

fig, ax = plt.subplots(figsize=(12, 6))
colors = {"Summer (Jun-Aug)": "#d62728", "Winter (Dec-Feb)": "#1f77b4", "Shoulder (Spring/Fall)": "#2ca02c"}

for season, group in diurnal_pd.groupby("season"):
    ax.plot(group["hour_local"], group["mean_demand_mw"], marker="o", linewidth=2.2, label=season, color=colors.get(season, "#333333"))

ax.set_title("ERCOT 24-Hour Diurnal Load Profiles by Season (Central Time)", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Hour of Day (Local Central Time)", fontsize=12)
ax.set_ylabel("Average Demand (MW)", fontsize=12)
ax.set_xticks(range(0, 24))
ax.legend(loc="upper left", frameon=True)
ax.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# COMMAND ----------
# DBTITLE 1,Figure 4: Temperature Distribution Across 4 Texas Metropolitan Stations
# Group temperatures to visualize regional climate variance (Houston vs DFW vs San Antonio vs Austin)
weather_sample_pd = df_weather.sample(withReplacement=False, fraction=0.2, seed=42) \
    .select("station_id", F.col("temperature_2m").cast(DoubleType())) \
    .dropna() \
    .toPandas()

fig, ax = plt.subplots(figsize=(12, 5))
station_order = ["dfw", "austin", "san_antonio", "houston"]
station_labels = ["Dallas-Fort Worth", "Austin", "San Antonio", "Houston"]

data_by_station = [weather_sample_pd[weather_sample_pd["station_id"] == st]["temperature_2m"].values for st in station_order]

bplot = ax.boxplot(data_by_station, patch_artist=True, tick_labels=station_labels, medianprops=dict(color="black", linewidth=1.5))

box_colors = ["#4e79a7", "#59a14f", "#edc949", "#e15759"]
for patch, color in zip(bplot["boxes"], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_title("Temperature Distribution Comparison Across Texas Load Centers (2021–2026)", fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("Temperature (°C)", fontsize=12)
ax.axhline(0, color="#1f77b4", linestyle="--", alpha=0.5, label="Freezing Mark (0°C)")
ax.axhline(35, color="#d62728", linestyle="--", alpha=0.5, label="Extreme Heat Mark (35°C / 95°F)")
ax.legend(loc="lower right", frameon=True)
ax.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
