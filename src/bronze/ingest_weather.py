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
import urllib.error
from pyspark.sql import functions as F
from delta.tables import DeltaTable

CATALOG = "main"
TABLE_NAME = f"{CATALOG}.bronze.weather_hourly_raw"
FULL_HISTORY_START = "2021-01-01"  # Earliest date for initial full load
ARCHIVE_DELAY_DAYS = 5            # Open-Meteo ERA5 archive is published with a five-day delay

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql("USE SCHEMA bronze")

# COMMAND ----------
# DBTITLE 1,Load Configuration & Establish Boundaries
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
weather_cfg = config.get("weather", {})
RESTATEMENT_BUFFER_DAYS = int(weather_cfg.get("restatement_buffer_days", 3))
weather_stations = weather_cfg.get("stations", [])
weather_vars = weather_cfg.get("hourly_variables", [])
ARCHIVE_BASE_URL = weather_cfg.get("archive_base_url", "https://archive-api.open-meteo.com/v1/archive")
vars_param = ",".join(weather_vars)


def date_chunks(start_date: str, end_date: str, days: int = 30):
    """Split archive backfills into small requests so one bad range cannot sink a full load."""
    current, end = datetime.date.fromisoformat(start_date), datetime.date.fromisoformat(end_date)
    while current <= end:
        chunk_end = min(current + datetime.timedelta(days=days - 1), end)
        yield current.isoformat(), chunk_end.isoformat()
        current = chunk_end + datetime.timedelta(days=1)


def fetch_weather_chunk(url: str, params: dict, station_id: str) -> dict:
    request_url = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(request_url, headers={"User-Agent": "ERCOT-Platform/1.0"})
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request, timeout=40) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")[:1000]
            if error.code < 500 and error.code != 429:
                raise RuntimeError(f"Open-Meteo rejected {station_id} request (HTTP {error.code}): {body}") from error
            if attempt == 3:
                raise RuntimeError(f"Open-Meteo request failed for {station_id} after 3 attempts (HTTP {error.code}): {body}") from error
        except Exception as error:
            if attempt == 3:
                raise RuntimeError(f"Open-Meteo request failed for {station_id} after 3 attempts: {error}") from error
        wait_time = 2 * attempt
        print(f"  Attempt {attempt} failed for {station_id}; retrying in {wait_time}s...")
        time.sleep(wait_time)
    raise RuntimeError(f"Open-Meteo request failed for {station_id}.")

# COMMAND ----------
# DBTITLE 1,High-Water Mark: Auto-Detect Full Load vs Incremental
now_utc = datetime.datetime.now(datetime.timezone.utc)
end_date_str = (now_utc.date() - datetime.timedelta(days=ARCHIVE_DELAY_DAYS)).strftime("%Y-%m-%d")

if spark.catalog.tableExists(TABLE_NAME):
    # One aggregate determines both whether data exists and its high-water mark.
    max_time = spark.table(TABLE_NAME).agg(F.max("time_utc").alias("max_time")).first()["max_time"]
else:
    max_time = None

if max_time is not None:
    hwm_dt = datetime.datetime.strptime(max_time[:10], "%Y-%m-%d") - datetime.timedelta(days=RESTATEMENT_BUFFER_DAYS)
    start_date_str = hwm_dt.strftime("%Y-%m-%d")
    if start_date_str > end_date_str:
        # The lake can be ahead of the archive's publication edge. Re-read the
        # latest available archive day so the daily job remains a safe no-op/upsert.
        print(
            f"[ARCHIVE CURRENT] High-water mark {max_time} is newer than the archive "
            f"boundary {end_date_str}. Rechecking the latest available day only."
        )
        start_date_str = end_date_str
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
    
    raw_api_responses[station_id] = []
    for chunk_start, chunk_end in date_chunks(start_date_str, end_date_str):
        params = {"latitude": lat, "longitude": lon, "start_date": chunk_start, "end_date": chunk_end,
                  "hourly": vars_param, "timezone": "UTC"}
        w_data = fetch_weather_chunk(ARCHIVE_BASE_URL, params, station_id)
        raw_api_responses[station_id].append({
            "request_params": params,
            "fetched_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "response_envelope": w_data,
        })
        if w_data.get("error"):
            raise RuntimeError(f"Open-Meteo rejected {station_id} request: {w_data.get('reason', 'unknown API error')}")
        hourly = w_data.get("hourly", {})
        timestamps = hourly.get("time", [])
        for i, timestamp in enumerate(timestamps):
            all_weather_rows.append({
                "station_id": station_id, "latitude": lat, "longitude": lon, "time_utc": timestamp,
                "temperature_2m": hourly.get("temperature_2m", [])[i] if i < len(hourly.get("temperature_2m", [])) else None,
                "relative_humidity_2m": hourly.get("relative_humidity_2m", [])[i] if i < len(hourly.get("relative_humidity_2m", [])) else None,
                "dew_point_2m": hourly.get("dew_point_2m", [])[i] if i < len(hourly.get("dew_point_2m", [])) else None,
                "apparent_temperature": hourly.get("apparent_temperature", [])[i] if i < len(hourly.get("apparent_temperature", [])) else None,
                "precipitation": hourly.get("precipitation", [])[i] if i < len(hourly.get("precipitation", [])) else None,
                "wind_speed_10m": hourly.get("wind_speed_10m", [])[i] if i < len(hourly.get("wind_speed_10m", [])) else None,
                "direct_normal_irradiance": hourly.get("direct_normal_irradiance", [])[i] if i < len(hourly.get("direct_normal_irradiance", [])) else None,
            })
        time.sleep(0.5)

print(f"Total weather station hourly records extracted: {len(all_weather_rows)} across {len(raw_api_responses)} stations.")

# COMMAND ----------
# DBTITLE 1,Land Raw to Volume & Append to Bronze Delta Table
if not all_weather_rows:
    raise RuntimeError(
        f"Bronze Ingestion Failure: 0 weather records retrieved across {len(weather_stations)} stations "
        f"for window {start_date_str} to {end_date_str}. Pipeline failed closed to prevent downstream silent stale tables (C-03)."
    )

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
duplicate_keys = (df_bronze_weather.groupBy("station_id", "time_utc")
                  .count().filter(F.col("count") > 1).count())
assert duplicate_keys == 0, f"Integrity Failure: Found {duplicate_keys} duplicate weather source keys in this batch."
source_deduped = df_bronze_weather

# Assert zero null primary keys
null_keys = source_deduped.filter(F.col("station_id").isNull() | F.col("time_utc").isNull()).count()
assert null_keys == 0, f"Integrity Failure: Found {null_keys} weather records with null primary keys."

# Assert all expected stations are present
ingested_stations = [r["station_id"] for r in source_deduped.select("station_id").distinct().collect()]
expected_stations = [s["station_id"] for s in weather_stations]
for exp_st in expected_stations:
    assert exp_st in ingested_stations, f"Completeness Failure: Expected station {exp_st} not found in ingested batch."
    
# Assert exact timestamp matrix equality: every single timestamp must have ALL stations reporting
missing_station_timestamps = (
    source_deduped.groupBy("time_utc")
    .agg(F.countDistinct("station_id").alias("stations_reporting"))
    .filter(F.col("stations_reporting") != len(expected_stations))
    .count()
)
assert missing_station_timestamps == 0, (
    f"Matrix Alignment Failure: Found {missing_station_timestamps} timestamps where not all "
    f"{len(expected_stations)} stations reported simultaneously."
)

total_distinct_hours = source_deduped.select("time_utc").distinct().count()
print(f"Quality Assertions Passed: 0 null keys. All {len(expected_stations)} stations aligned across exactly {total_distinct_hours:,} synchronized hours.")

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
        "AND target.time_utc = source.time_utc "
        f"AND target.time_utc >= '{start_date_str}T00:00'"
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
    
    merge_source_columns(
        delta_table,
        source_deduped,
        merge_condition,
        update_condition=update_condition,
    )
    print(f"Idempotent MERGE completed on {TABLE_NAME} (null-safe equality enabled).")
