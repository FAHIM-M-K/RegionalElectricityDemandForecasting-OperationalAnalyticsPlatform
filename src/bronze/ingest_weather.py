# Databricks notebook source
# MAGIC %md
# MAGIC # 02. Open-Meteo Multi-Site Weather Ingestion
# MAGIC
# MAGIC **Source:** Open-Meteo Historical Archive & Forecast API (Keyless)  
# MAGIC **Target:** `bronze.weather_hourly_raw`  
# MAGIC **Panel:** 4 Representative Texas Metropolitan Centers (Houston, DFW, San Antonio, Austin)  
# MAGIC **Key Features:** Dry-bulb temp, apparent temp, dew point, humidity, wind, solar irradiance.

# COMMAND ----------
# DBTITLE 1,Imports & Configuration
import os
import json
import time
import datetime
import urllib.request
import urllib.parse
from pyspark.sql import functions as F
from delta.tables import DeltaTable

CATALOG = "main"
TABLE_NAME = f"{CATALOG}.bronze.weather_hourly_raw"
FULL_HISTORY_START = "2021-01-01"  # Earliest date for initial full load
RESTATEMENT_BUFFER_DAYS = 3       # Overlap to catch weather observation corrections

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql("USE SCHEMA bronze")

# COMMAND ----------
# DBTITLE 1,High-Water Mark: Auto-Detect Full Load vs Incremental
now_utc = datetime.datetime.now(datetime.timezone.utc)
end_date_str = now_utc.strftime("%Y-%m-%d")

if spark.catalog.tableExists(TABLE_NAME) and spark.table(TABLE_NAME).count() > 0:
    # Table exists with data: incremental from high-water mark minus buffer
    max_time = spark.sql(f"SELECT max(time_utc) FROM {TABLE_NAME}").first()[0]
    hwm_dt = datetime.datetime.strptime(max_time[:10], "%Y-%m-%d") - datetime.timedelta(days=RESTATEMENT_BUFFER_DAYS)
    start_date_str = hwm_dt.strftime("%Y-%m-%d")
    print(f"[INCREMENTAL] High-water mark: {max_time}. Fetching from {start_date_str} (with {RESTATEMENT_BUFFER_DAYS}-day buffer).")
else:
    # First run or empty table: full historical load
    start_date_str = FULL_HISTORY_START
    print(f"[FULL LOAD] No existing data found. Loading full history from {start_date_str}.")

# COMMAND ----------
# DBTITLE 1,Weather Stations Panel Definition
weather_stations = [
    {"station_id": "houston", "name": "Houston", "latitude": 29.7604, "longitude": -95.3698, "weight": 0.35},
    {"station_id": "dfw", "name": "Dallas-Fort Worth", "latitude": 32.7767, "longitude": -96.7970, "weight": 0.35},
    {"station_id": "san_antonio", "name": "San Antonio", "latitude": 29.4241, "longitude": -98.4936, "weight": 0.15},
    {"station_id": "austin", "name": "Austin", "latitude": 30.2672, "longitude": -97.7431, "weight": 0.15}
]

weather_vars = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "apparent_temperature", "precipitation", "wind_speed_10m",
    "direct_normal_irradiance"
]
vars_param = ",".join(weather_vars)

# COMMAND ----------
# DBTITLE 1,Fetch Multi-Site Weather
all_weather_rows = []

for st in weather_stations:
    station_id = st["station_id"]
    lat = st["latitude"]
    lon = st["longitude"]
    print(f"Fetching weather for {st['name']} ({station_id})...")
    
    archive_url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&start_date={start_date_str}&end_date={end_date_str}&"
        f"hourly={vars_param}&timezone=UTC"
    )
    
    req = urllib.request.Request(archive_url, headers={"User-Agent": "ERCOT-Platform/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        w_data = json.loads(resp.read().decode("utf-8"))
        
    hourly = w_data.get("hourly", {})
    timestamps = hourly.get("time", [])
    
    for i, t in enumerate(timestamps):
        row = {
            "station_id": station_id,
            "latitude": lat,
            "longitude": lon,
            "time_utc": t,
            "temperature_2m": hourly.get("temperature_2m", [])[i] if i < len(hourly.get("temperature_2m", [])) else None,
            "relative_humidity_2m": hourly.get("relative_humidity_2m", [])[i] if i < len(hourly.get("relative_humidity_2m", [])) else None,
            "dew_point_2m": hourly.get("dew_point_2m", [])[i] if i < len(hourly.get("dew_point_2m", [])) else None,
            "apparent_temperature": hourly.get("apparent_temperature", [])[i] if i < len(hourly.get("apparent_temperature", [])) else None,
            "precipitation": hourly.get("precipitation", [])[i] if i < len(hourly.get("precipitation", [])) else None,
            "wind_speed_10m": hourly.get("wind_speed_10m", [])[i] if i < len(hourly.get("wind_speed_10m", [])) else None,
            "direct_normal_irradiance": hourly.get("direct_normal_irradiance", [])[i] if i < len(hourly.get("direct_normal_irradiance", [])) else None,
        }
        all_weather_rows.append(row)
    time.sleep(0.5)

print(f"Total weather station hourly records fetched: {len(all_weather_rows)}")

# COMMAND ----------
# DBTITLE 1,Land Raw to Volume & Append to Bronze Delta Table
if all_weather_rows:
    # 1. Production Landing: Save raw multi-station weather payload to Volume
    raw_weather_dir = f"/Volumes/{CATALOG}/bronze/raw_landing/weather"
    os.makedirs(raw_weather_dir, exist_ok=True)
    timestamp_epoch = int(time.time())
    raw_weather_file = f"weather_ercot_panel_{start_date_str}_{end_date_str}_{timestamp_epoch}.json"
    raw_weather_path = f"{raw_weather_dir}/{raw_weather_file}"
    
    with open(raw_weather_path, "w", encoding="utf-8") as f:
        json.dump(all_weather_rows, f)
    print(f"Archived raw weather payload to Volume: {raw_weather_path}")

    # 2. Convert to Spark DataFrame & attach provenance metadata
    df_weather = spark.createDataFrame(all_weather_rows)
    df_bronze_weather = df_weather \
        .withColumn("_ingested_at_utc", F.current_timestamp()) \
        .withColumn("_source", F.lit("Open-Meteo-Archive")) \
        .withColumn("_raw_payload_path", F.lit(raw_weather_path))
        
    # 3. Idempotent Upsert into Bronze Delta table
    source_deduped = df_bronze_weather.dropDuplicates(["station_id", "time_utc"])
    
    if not spark.catalog.tableExists(TABLE_NAME):
        source_deduped.write \
            .format("delta") \
            .mode("overwrite") \
            .saveAsTable(TABLE_NAME)
        print(f"Initialized table {TABLE_NAME} with {source_deduped.count()} records.")
    else:
        delta_table = DeltaTable.forName(spark, TABLE_NAME)
        merge_condition = (
            "target.station_id = source.station_id "
            "AND target.time_utc = source.time_utc"
        )
        delta_table.alias("target").merge(
            source_deduped.alias("source"),
            merge_condition
        ).whenMatchedUpdateAll(
        ).whenNotMatchedInsertAll(
        ).execute()
        print(f"Idempotent MERGE completed on {TABLE_NAME}.")

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT station_id, count(*) as count, min(time_utc) as min_time, max(time_utc) as max_time
# MAGIC FROM bronze.weather_hourly_raw
# MAGIC GROUP BY station_id;
