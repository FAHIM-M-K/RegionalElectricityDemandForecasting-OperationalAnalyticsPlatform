# Databricks notebook source
"""Capture one issue-time ECMWF IFS forecast snapshot for the four ERCOT sites.

The four coordinates are requested together in one HTTP call. Each successful
hourly run is retained as an immutable Bronze snapshot with a conservative
availability timestamp taken after the provider response is received.
"""

import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from delta.tables import DeltaTable
from pyspark.sql import functions as F

CATALOG = "main"
TARGET_TABLE = f"{CATALOG}.bronze.weather_forecast_snapshots"
RAW_DIR = f"/Volumes/{CATALOG}/bronze/raw_landing/weather_forecasts"
SNAPSHOT_HOURS = 24
FORECAST_DAYS = 2  # Request enough provider coverage to retain the next 24 complete target hours.
MAX_ATTEMPTS = 2  # Retry a transient transport/server failure once; never retry 429.

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
from src.shared.delta_merge import merge_source_columns

config = load_platform_config("ercot_config.json")
weather_cfg = config["weather"]
stations = weather_cfg["stations"]
variables = weather_cfg["hourly_variables"]
forecast_url = weather_cfg.get(
    "forecast_base_url", "https://api.open-meteo.com/v1/ecmwf"
)
if len(stations) != 4 or len({station["station_id"] for station in stations}) != 4:
    raise ValueError("Expected four uniquely named representative weather stations.")
if not variables or len(variables) > 10:
    raise ValueError("Request a non-empty set of at most 10 weather variables.")

snapshot_batch_id = str(uuid.uuid4())
station_ids = [station["station_id"] for station in stations]


def request_weather_forecast(params: dict) -> tuple[list[dict], dict, dt.datetime, dt.datetime]:
    """Make one batched request; only retry transient failures, never quota errors."""
    request_url = f"{forecast_url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        request_url,
        headers={"User-Agent": "ERCOT-OperationalAnalytics/1.0 (weather snapshot)"},
    )
    requested_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if isinstance(payload, dict) and payload.get("error"):
                raise RuntimeError(
                    f"Open-Meteo rejected the forecast request: "
                    f"{payload.get('reason', 'unknown API error')}"
                )
            locations = payload if isinstance(payload, list) else [payload]
            if len(locations) != len(stations):
                raise RuntimeError(
                    f"Open-Meteo returned {len(locations)} location payloads; "
                    f"expected {len(stations)}."
                )
            available_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
            return locations, payload, requested_at, available_at
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:1000]
            # Especially do not retry 429: that would spend more quota while limited.
            if error.code == 429 or error.code < 500 or attempt == MAX_ATTEMPTS:
                raise RuntimeError(
                    f"Open-Meteo forecast request failed (HTTP {error.code}): {detail}"
                ) from error
        except RuntimeError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(
                    f"Open-Meteo forecast request failed after {attempt} attempts: {error}"
                ) from error
        wait_seconds = 2
        print(f"Transient weather API failure; retrying the single batched call once in {wait_seconds}s.")
        time.sleep(wait_seconds)


params = {
    "latitude": ",".join(str(station["latitude"]) for station in stations),
    "longitude": ",".join(str(station["longitude"]) for station in stations),
    "hourly": ",".join(variables),
    "timezone": "UTC",
    "forecast_days": FORECAST_DAYS,
}
location_payloads, raw_payload, requested_at, available_at = request_weather_forecast(params)

# Keep exactly the next 24 complete target hours after this snapshot became
# available. This preserves the forecast vintage and excludes already-started hours.
first_target = available_at.replace(minute=0, second=0, microsecond=0) + dt.timedelta(hours=1)
last_target = available_at + dt.timedelta(hours=SNAPSHOT_HOURS)
batch_rows = []
target_sets = {}

for station, location in zip(stations, location_payloads):
    station_id = station["station_id"]
    hourly = location.get("hourly", {})
    target_times = hourly.get("time", [])
    arrays = {name: hourly.get(name, []) for name in variables}
    malformed = [name for name, values in arrays.items() if len(values) != len(target_times)]
    if not target_times or malformed:
        raise RuntimeError(
            f"Malformed Open-Meteo hourly response for {station_id}; "
            f"missing times or misaligned variables: {malformed}. No snapshot written."
        )

    station_targets = set()
    for index, target_text in enumerate(target_times):
        target = dt.datetime.fromisoformat(target_text)
        if target.tzinfo is None:
            target = target.replace(tzinfo=dt.timezone.utc)
        else:
            target = target.astimezone(dt.timezone.utc)
        if not first_target <= target <= last_target:
            continue
        station_targets.add(target)
        row = {
            "snapshot_batch_id": snapshot_batch_id,
            "station_id": station_id,
            "snapshot_requested_at_utc": requested_at,
            "snapshot_available_at_utc": available_at,
            "target_timestamp_utc": target,
            "latitude": float(station["latitude"]),
            "longitude": float(station["longitude"]),
        }
        for name, values in arrays.items():
            row[name] = None if values[index] is None else float(values[index])
        batch_rows.append(row)
    target_sets[station_id] = station_targets

expected_targets = set.intersection(*(target_sets[sid] for sid in station_ids)) if target_sets else set()
if len(expected_targets) != SNAPSHOT_HOURS or any(
    target_sets[sid] != expected_targets for sid in station_ids
):
    counts = {station_id: len(targets) for station_id, targets in target_sets.items()}
    raise RuntimeError(
        f"Expected the same {SNAPSHOT_HOURS} future hours for all stations; got {counts}. "
        "No partial snapshot was written."
    )
if any(row["temperature_2m"] is None for row in batch_rows):
    raise RuntimeError("ECMWF returned a missing temperature value; no snapshot was written.")

os.makedirs(RAW_DIR, exist_ok=True)
raw_path = f"{RAW_DIR}/open_meteo_ecmwf_{snapshot_batch_id}.json"
with open(raw_path, "w", encoding="utf-8") as raw_file:
    json.dump(
        {
            "provider": "Open-Meteo",
            "weather_model": "ECMWF IFS HRES",
            "request_params": params,
            "requested_at_utc": requested_at.isoformat(),
            "available_at_utc": available_at.isoformat(),
            "response": raw_payload,
        },
        raw_file,
    )

schema_fields = [
    StructField("snapshot_batch_id", StringType(), False),
    StructField("station_id", StringType(), False),
    StructField("snapshot_requested_at_utc", TimestampType(), False),
    StructField("snapshot_available_at_utc", TimestampType(), False),
    StructField("target_timestamp_utc", TimestampType(), False),
    StructField("latitude", DoubleType(), False),
    StructField("longitude", DoubleType(), False),
]
schema_fields.extend(StructField(name, DoubleType(), True) for name in variables)
schema = StructType(schema_fields)
incoming = spark.createDataFrame(batch_rows, schema=schema)
incoming = (
    incoming.withColumn("_source", F.lit("Open-Meteo-ECMWF-IFS-HRES"))
    .withColumn("_raw_payload_path", F.lit(raw_path))
    .withColumn("_ingested_at_utc", F.current_timestamp())
)

duplicate_count = (
    incoming.groupBy("snapshot_batch_id", "station_id", "target_timestamp_utc")
    .count().filter(F.col("count") > 1).limit(1).count()
)
if duplicate_count:
    raise RuntimeError("Batched forecast snapshot has duplicate station/target keys.")

if not spark.catalog.tableExists(TARGET_TABLE):
    incoming.write.format("delta").mode("overwrite").saveAsTable(TARGET_TABLE)
else:
    merge_source_columns(
        DeltaTable.forName(spark, TARGET_TABLE),
        incoming,
        "target.snapshot_batch_id = source.snapshot_batch_id "
        "AND target.station_id = source.station_id "
        "AND target.target_timestamp_utc = source.target_timestamp_utc",
        update_matched=False,
    )

print(
    f"Stored ECMWF IFS forecast snapshot {snapshot_batch_id}: "
    f"{len(batch_rows):,} station-hours, {len(expected_targets)} target hours, "
    f"{min(expected_targets)} to {max(expected_targets)} UTC. "
    "All four sites were retrieved in one API request; availability is stamped "
    "after the response because the provider does not expose its model-run issue time."
)
