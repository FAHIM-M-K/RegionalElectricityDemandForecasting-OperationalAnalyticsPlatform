# Databricks notebook source
"""Conform issue-time station forecasts into regional hourly features.

This Silver table preserves each forecast snapshot's availability timestamp so
future model experiments can join only weather information available at the
demand forecast origin. The configured metro weights are project heuristics,
not measured ERCOT load shares.
"""

import os
import sys

from delta.tables import DeltaTable
from pyspark.sql import functions as F

CATALOG = "main"
SOURCE_TABLE = f"{CATALOG}.bronze.weather_forecast_snapshots"
TARGET_TABLE = f"{CATALOG}.silver.weather_forecast_conformed"

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
station_weights = {
    station["station_id"]: float(station["weight"])
    for station in config["weather"]["stations"]
}
expected_station_count = len(station_weights)
if abs(sum(station_weights.values()) - 1.0) > 1e-9:
    raise ValueError(f"Weather station weights must sum to 1.0: {station_weights}")
if not spark.catalog.tableExists(SOURCE_TABLE):
    raise RuntimeError(f"Weather forecast snapshot table does not exist: {SOURCE_TABLE}")

raw = spark.table(SOURCE_TABLE)
key_columns = ["snapshot_batch_id", "station_id", "target_timestamp_utc"]
duplicate_keys = raw.groupBy(*key_columns).count().filter(F.col("count") > 1).limit(1).count()
if duplicate_keys:
    raise RuntimeError("Bronze weather forecast snapshots contain duplicate station/target keys.")

invalid_asof = raw.filter(
    F.col("snapshot_available_at_utc").isNull()
    | F.col("target_timestamp_utc").isNull()
    | (F.col("snapshot_available_at_utc") >= F.col("target_timestamp_utc"))
).limit(1).count()
if invalid_asof:
    raise RuntimeError("Weather snapshot has a missing or non-future target relative to its availability time.")

weight_entries = []
for station_id, weight in station_weights.items():
    weight_entries.extend([F.lit(station_id), F.lit(weight)])
weighted = raw.withColumn("station_weight", F.element_at(F.create_map(*weight_entries), F.col("station_id")))
unknown_station = weighted.filter(F.col("station_weight").isNull()).limit(1).count()
if unknown_station:
    raise RuntimeError("Bronze weather forecast snapshots contain an unconfigured station_id.")

group_columns = [
    "snapshot_batch_id",
    "snapshot_available_at_utc",
    "target_timestamp_utc",
]

def weighted_mean(column_name: str):
    value = F.col(column_name)
    usable_weight = F.when(value.isNotNull(), F.col("station_weight"))
    return (
        F.sum(F.when(value.isNotNull(), value * F.col("station_weight")))
        / F.nullif(F.sum(usable_weight), F.lit(0.0))
    )


regional = weighted.groupBy(*group_columns).agg(
    F.min("snapshot_requested_at_utc").alias("first_station_requested_at_utc"),
    F.max("snapshot_requested_at_utc").alias("last_station_requested_at_utc"),
    F.countDistinct("station_id").alias("reporting_stations_count"),
    F.countDistinct(F.when(F.col("temperature_2m").isNotNull(), F.col("station_id")))
     .alias("temperature_reporting_stations_count"),
    F.sum(F.when(F.col("temperature_2m").isNotNull(), F.col("station_weight")))
     .alias("temperature_weight_coverage"),
    F.round(weighted_mean("temperature_2m") * F.lit(9.0 / 5.0) + F.lit(32.0), 2)
     .alias("weighted_temperature_f"),
    F.round(weighted_mean("apparent_temperature") * F.lit(9.0 / 5.0) + F.lit(32.0), 2)
     .alias("weighted_apparent_temperature_f"),
    F.round(weighted_mean("dew_point_2m") * F.lit(9.0 / 5.0) + F.lit(32.0), 2)
     .alias("weighted_dew_point_f"),
    F.round(weighted_mean("relative_humidity_2m"), 1).alias("weighted_relative_humidity_pct"),
    F.round(weighted_mean("precipitation"), 2).alias("weighted_precipitation_mm"),
    F.round(weighted_mean("wind_speed_10m") / F.lit(3.6), 2).alias("weighted_wind_speed_ms"),
    F.round(weighted_mean("direct_normal_irradiance"), 1).alias("weighted_solar_irradiance_wm2"),
    F.max("_raw_payload_path").alias("_raw_payload_path"),
    F.max("_ingested_at_utc").alias("_bronze_ingested_at_utc"),
).withColumn(
    "cdh_65_forecast", F.greatest(F.col("weighted_temperature_f") - F.lit(65.0), F.lit(0.0))
).withColumn(
    "hdh_65_forecast", F.greatest(F.lit(65.0) - F.col("weighted_temperature_f"), F.lit(0.0))
).withColumn(
    "is_complete_weather_forecast",
    F.col("reporting_stations_count") == F.lit(expected_station_count),
).withColumn("_conformed_at_utc", F.current_timestamp())

duplicate_targets = (
    regional.groupBy("snapshot_batch_id", "target_timestamp_utc")
    .count().filter(F.col("count") > 1).limit(1).count()
)
if duplicate_targets:
    raise RuntimeError("Silver weather forecast rows are not unique by snapshot and target hour.")
if regional.filter(F.col("weighted_temperature_f").isNull()).limit(1).count():
    raise RuntimeError("A regional weather forecast row has no usable temperature values.")

if not spark.catalog.tableExists(TARGET_TABLE):
    regional.write.format("delta").mode("overwrite").saveAsTable(TARGET_TABLE)
else:
    DeltaTable.forName(spark, TARGET_TABLE).alias("target").merge(
        regional.alias("source"),
        "target.snapshot_batch_id = source.snapshot_batch_id "
        "AND target.target_timestamp_utc = source.target_timestamp_utc",
    ).whenNotMatchedInsertAll().execute()

summary = spark.table(TARGET_TABLE).filter(
    F.col("snapshot_batch_id") == F.lit(regional.select("snapshot_batch_id").first()[0])
).agg(
    F.count("*").alias("target_hours"),
    F.sum(F.col("is_complete_weather_forecast").cast("int")).alias("complete_hours"),
    F.min("target_timestamp_utc").alias("first_target_utc"),
    F.max("target_timestamp_utc").alias("last_target_utc"),
).first()
print(
    f"Materialized {summary['target_hours']:,} regional weather hours in {TARGET_TABLE}; "
    f"{summary['complete_hours']:,} have all {expected_station_count} stations, "
    f"targets {summary['first_target_utc']} to {summary['last_target_utc']}."
)
