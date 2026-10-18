# Databricks notebook source
"""Capture issue-time Open-Meteo forecasts for leakage-safe future evaluation.

The archive weather table contains retrospective observations only. This
notebook stores the forecast values returned at each run together with the
time each station was queried. That retrieval time is the operational
availability timestamp; Open-Meteo does not expose a model initialization
timestamp through this endpoint.
"""

import json
import os
import sys
import time
import uuid
import datetime as dt
import urllib.error
import urllib.parse
import urllib.request

from delta.tables import DeltaTable
from pyspark.sql import functions as F

CATALOG = "main"
TARGET_TABLE = f"{CATALOG}.bronze.weather_forecast_snapshots"
RAW_DIR = f"/Volumes/{CATALOG}/bronze/raw_landing/weather_forecasts"
HORIZON_HOURS = 168
FORECAST_DAYS = 8

spark.conf.set("spark.sql.session.timeZone", "UTC")

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
weather_cfg = config["weather"]
stations = weather_cfg["stations"]
variables = weather_cfg["hourly_variables"]
forecast_url = weather_cfg.get("forecast_base_url", "https://api.open-meteo.com/v1/forecast")
snapshot_batch_id = str(uuid.uuid4())


def fetch_forecast(params: dict, station_id: str) -> dict:
    """Fetch a station forecast, retrying transient API/server errors only."""
    request_url = f"{forecast_url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(request_url, headers={"User-Agent": "ERCOT-Platform/1.0"})
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("error"):
                raise RuntimeError(
                    f"Open-Meteo rejected {station_id}: {payload.get('reason', 'unknown API error')}"
                )
            return payload
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:1000]
            if error.code < 500 and error.code != 429:
                raise RuntimeError(
                    f"Open-Meteo rejected {station_id} (HTTP {error.code}): {detail}"
                ) from error
            if attempt == 3:
                raise RuntimeError(
                    f"Open-Meteo failed for {station_id} after {attempt} attempts (HTTP {error.code})."
                ) from error
        except RuntimeError:
            raise
        except Exception as error:
            if attempt == 3:
                raise RuntimeError(
                    f"Open-Meteo failed for {station_id} after {attempt} attempts: {error}"
                ) from error
        wait_seconds = 2 * attempt
        print(f"Transient weather forecast request error for {station_id}; retrying in {wait_seconds}s.")
        time.sleep(wait_seconds)


batch_rows = []
raw_payloads = {}
for station in stations:
    station_id = station["station_id"]
    params = {
        "latitude": station["latitude"],
        "longitude": station["longitude"],
        "hourly": ",".join(variables),
        "timezone": "UTC",
        "forecast_days": FORECAST_DAYS,
    }
    queried_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    payload = fetch_forecast(params, station_id)
    retrieved_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    raw_payloads[station_id] = {
        "request_params": params,
        "queried_at_utc": queried_at.isoformat(),
        "retrieved_at_utc": retrieved_at.isoformat(),
        "response_envelope": payload,
    }
    hourly = payload.get("hourly", {})
    target_times = hourly.get("time", [])
    if not target_times:
        raise RuntimeError(f"Open-Meteo returned no hourly forecast times for {station_id}.")
    arrays = {name: hourly.get(name, []) for name in variables}
    malformed = [name for name, values in arrays.items() if len(values) != len(target_times)]
    if malformed:
        raise RuntimeError(f"Forecast arrays for {station_id} do not align: {malformed}.")

    station_rows = 0
    for index, target_text in enumerate(target_times):
        target = dt.datetime.fromisoformat(target_text).replace(tzinfo=dt.timezone.utc)
        row = {
            "snapshot_batch_id": snapshot_batch_id,
            "station_id": station_id,
            "snapshot_requested_at_utc": queried_at,
            "snapshot_available_at_utc": retrieved_at,
            "target_timestamp_utc": target,
            "latitude": float(station["latitude"]),
            "longitude": float(station["longitude"]),
        }
        for name, values in arrays.items():
            row[name] = None if values[index] is None else float(values[index])
        batch_rows.append(row)
        station_rows += 1
    if station_rows < HORIZON_HOURS - 2:
        raise RuntimeError(
            f"Only {station_rows} future hourly values were returned for {station_id}; "
            f"expected at least {HORIZON_HOURS - 2}. No snapshot batch was written."
        )
    print(f"Captured {station_rows} future hourly forecast rows for {station_id}.")

# Use one conservative availability cutoff for the full multi-station snapshot.
# This avoids claiming the first station's forecasts were available before the
# last station response had arrived, and keeps all stations aligned by target.
snapshot_available_at = max(row["snapshot_available_at_utc"] for row in batch_rows)
first_valid_target = snapshot_available_at.replace(minute=0, second=0, microsecond=0) + dt.timedelta(hours=1)
last_valid_target = snapshot_available_at + dt.timedelta(hours=HORIZON_HOURS)
batch_rows = [
    row for row in batch_rows
    if first_valid_target <= row["target_timestamp_utc"] <= last_valid_target
]
for row in batch_rows:
    row["snapshot_available_at_utc"] = snapshot_available_at

expected_stations = {station["station_id"] for station in stations}
captured_stations = {row["station_id"] for row in batch_rows}
if captured_stations != expected_stations:
    raise RuntimeError(
        f"Incomplete station forecast batch: expected {sorted(expected_stations)}, "
        f"received {sorted(captured_stations)}. No snapshot batch was written."
    )
station_counts = {}
for row in batch_rows:
    station_counts[row["station_id"]] = station_counts.get(row["station_id"], 0) + 1
short_stations = {name: count for name, count in station_counts.items() if count < HORIZON_HOURS - 2}
if short_stations:
    raise RuntimeError(
        f"Common aligned snapshot contains too few future hours by station: {short_stations}. "
        "No snapshot batch was written."
    )

# Preserve original provider payloads for audit/replay before materializing rows.
os.makedirs(RAW_DIR, exist_ok=True)
raw_path = f"{RAW_DIR}/open_meteo_forecast_{snapshot_batch_id}.json"
with open(raw_path, "w", encoding="utf-8") as raw_file:
    json.dump(raw_payloads, raw_file)

incoming = spark.createDataFrame(batch_rows)
incoming = (
    incoming
    .withColumn("_source", F.lit("Open-Meteo-Forecast"))
    .withColumn("_raw_payload_path", F.lit(raw_path))
    .withColumn("_ingested_at_utc", F.current_timestamp())
)

duplicate_count = (
    incoming.groupBy("snapshot_batch_id", "station_id", "target_timestamp_utc")
    .count().filter(F.col("count") > 1).count()
)
if duplicate_count:
    raise RuntimeError(f"Snapshot batch contains {duplicate_count} duplicate station/target keys.")

if not spark.catalog.tableExists(TARGET_TABLE):
    incoming.write.format("delta").mode("overwrite").saveAsTable(TARGET_TABLE)
else:
    DeltaTable.forName(spark, TARGET_TABLE).alias("target").merge(
        incoming.alias("source"),
        "target.snapshot_batch_id = source.snapshot_batch_id "
        "AND target.station_id = source.station_id "
        "AND target.target_timestamp_utc = source.target_timestamp_utc",
    ).whenNotMatchedInsertAll().execute()

summary = incoming.agg(
    F.count("*").alias("rows"),
    F.countDistinct("target_timestamp_utc").alias("target_hours"),
    F.min("target_timestamp_utc").alias("first_target_utc"),
    F.max("target_timestamp_utc").alias("last_target_utc"),
).first()
print(
    f"Stored weather snapshot {snapshot_batch_id}: {summary['rows']:,} station-hours, "
    f"{summary['target_hours']:,} target hours, {summary['first_target_utc']} to "
    f"{summary['last_target_utc']} in {TARGET_TABLE}. Availability time is the as-of time; "
    "the provider's internal model initialization time is not exposed by this endpoint."
)
