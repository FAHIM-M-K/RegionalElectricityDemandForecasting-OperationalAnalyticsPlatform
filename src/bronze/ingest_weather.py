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
# DBTITLE 1,Load Configuration & Establish Boundaries
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
weather_cfg = config.get("weather", {})
weather_stations = weather_cfg.get("stations", [])
weather_vars = weather_cfg.get("hourly_variables", [])
ARCHIVE_BASE_URL = weather_cfg.get("archive_base_url", "https://archive-api.open-meteo.com/v1/archive")
vars_param = ",".join(weather_vars)

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
# DBTITLE 1,Fetch Multi-Site Weather with Retries & Raw Response Preservation
all_weather_rows = []
raw_api_responses = {}

for st in weather_stations:
    station_id = st["station_id"]
    lat = st["latitude"]
    lon = st["longitude"]
    print(f"Fetching weather for {st['name']} ({station_id})...")
    
    archive_url = (
        f"{ARCHIVE_BASE_URL}?"
        f"latitude={lat}&longitude={lon}&start_date={start_date_str}&end_date={end_date_str}&"
        f"hourly={vars_param}&timezone=UTC"
    )
    
    req = urllib.request.Request(archive_url, headers={"User-Agent": "ERCOT-Platform/1.0"})
    
    w_data = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(req, timeout=40) as resp:
                raw_text = resp.read().decode("utf-8")
                w_data = json.loads(raw_text)
                break
        except Exception as e:
            if attempt == 3:
                raise RuntimeError(f"Open-Meteo request failed for {station_id} after 3 attempts: {e}")
            wait_time = 2 * attempt
            print(f"  Attempt {attempt} failed for {station_id} ({e}). Retrying in {wait_time}s...")
            time.sleep(wait_time)
            
    # Preserve full, authentic API response envelope and request parameters
    raw_api_responses[station_id] = {
        "station_id": station_id,
        "latitude": lat,
        "longitude": lon,
        "request_url": archive_url,
        "fetched_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "response_envelope": w_data
    }
        
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

print(f"Total weather station hourly records extracted: {len(all_weather_rows)} across {len(raw_api_responses)} stations.")

# COMMAND ----------
# DBTITLE 1,Land Raw to Volume & Append to Bronze Delta Table
if all_weather_rows:
    # 1. Production Landing: Save full original API responses to Volume
    raw_weather_dir = f"/Volumes/{CATALOG}/bronze/raw_landing/weather"
    os.makedirs(raw_weather_dir, exist_ok=True)
    timestamp_epoch = int(time.time())
    raw_weather_file = f"weather_ercot_raw_responses_{start_date_str}_{end_date_str}_{timestamp_epoch}.json"
    raw_weather_path = f"{raw_weather_dir}/{raw_weather_file}"
    
    with open(raw_weather_path, "w", encoding="utf-8") as f:
        json.dump(raw_api_responses, f)
    print(f"Archived authentic raw API responses to Volume: {raw_weather_path}")

    # 2. Convert to Spark DataFrame & attach provenance metadata
    df_weather = spark.createDataFrame(all_weather_rows)
    df_bronze_weather = df_weather \
        .withColumn("_ingested_at_utc", F.current_timestamp()) \
        .withColumn("_source", F.lit("Open-Meteo-Archive")) \
        .withColumn("_raw_payload_path", F.lit(raw_weather_path))
        
    # 3. Quality & Completeness Assertions
    source_deduped = df_bronze_weather.dropDuplicates(["station_id", "time_utc"])
    
    # Assert zero null primary keys
    null_keys = source_deduped.filter(F.col("station_id").isNull() | F.col("time_utc").isNull()).count()
    assert null_keys == 0, f"Integrity Failure: Found {null_keys} weather records with null primary keys."
    
    # Assert all expected stations are present
    ingested_stations = [r["station_id"] for r in source_deduped.select("station_id").distinct().collect()]
    expected_stations = [s["station_id"] for s in weather_stations]
    for exp_st in expected_stations:
        assert exp_st in ingested_stations, f"Completeness Failure: Expected station {exp_st} not found in ingested batch."
        
    # Assert timestamp matrix coverage: every station must have the exact same count of timestamps
    station_counts = (
        source_deduped.groupBy("station_id")
        .count()
        .collect()
    )
    counts_map = {row["station_id"]: row["count"] for row in station_counts}
    unique_counts = set(counts_map.values())
    assert len(unique_counts) == 1, (
        f"Alignment Failure: Stations have unequal hourly timestamps! Counts: {counts_map}"
    )
        
    print(f"Quality Assertions Passed: 0 null keys. All {len(expected_stations)} stations present with identical count ({list(unique_counts)[0]} hrs).")
    
    # 4. Idempotent Upsert into Bronze Delta table with Null-Safe Equality (<=>)
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
        
        # Null-safe inequality: NOT (target <=> source) correctly detects null -> val, val -> null, val -> new_val
        update_condition = (
            "NOT (target.temperature_2m <=> source.temperature_2m) OR "
            "NOT (target.relative_humidity_2m <=> source.relative_humidity_2m) OR "
            "NOT (target.dew_point_2m <=> source.dew_point_2m) OR "
            "NOT (target.apparent_temperature <=> source.apparent_temperature) OR "
            "NOT (target.precipitation <=> source.precipitation) OR "
            "NOT (target.wind_speed_10m <=> source.wind_speed_10m) OR "
            "NOT (target.direct_normal_irradiance <=> source.direct_normal_irradiance)"
        )
        
        delta_table.alias("target").merge(
            source_deduped.alias("source"),
            merge_condition
        ).whenMatchedUpdateAll(
            condition=update_condition
        ).whenNotMatchedInsertAll(
        ).execute()
        print(f"Idempotent MERGE completed on {TABLE_NAME} (null-safe equality enabled).")

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT station_id, count(*) as count, min(time_utc) as min_time, max(time_utc) as max_time
# MAGIC FROM bronze.weather_hourly_raw
# MAGIC GROUP BY station_id;
