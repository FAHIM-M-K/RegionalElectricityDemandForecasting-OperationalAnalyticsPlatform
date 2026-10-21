# Databricks notebook source
"""Build issue-time training examples for the hourly rolling H24 forecast.

Writes the active rolling H24 training examples. Demand history is loaded in
bounded 30-day chunks with the exact 336-hour historical lookback and a
24-hour future label halo.
"""

import os
import sys

import pandas as pd
from pyspark.sql import functions as F

CATALOG = "main"
SOURCE_TABLE = f"{CATALOG}.gold.fact_hourly_electricity_demand"
TARGET_TABLE = f"{CATALOG}.gold.forecast_examples_hourly"
RESPONDENT = "ERCO"
CHUNK_DAYS = 30
HISTORY_HOURS = 336
MAX_HORIZON_HOURS = 24
CORRECTION_LOOKBACK_DAYS = 7  # EIA restatements plus 24h labels and margin

spark.conf.set("spark.sql.session.timeZone", "UTC")
_root = os.getcwd()
for _ in range(5):
    if os.path.isdir(os.path.join(_root, "configs")):
        if _root not in sys.path:
            sys.path.insert(0, _root)
        break
    _root = os.path.dirname(_root)

from src.gold.forecast_example_features import generate_forecast_examples_df
from src.shared.forecast_contract import FEATURE_SET_DEMAND_FOUNDATION


def _as_utc(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


required_columns = {
    "respondent", "timestamp_utc", "actual_demand_mw", "day_ahead_forecast_mw",
    "is_holiday", "is_grid_emergency",
}
available_columns = set(spark.table(SOURCE_TABLE).columns)
missing_columns = required_columns - available_columns
if missing_columns:
    raise RuntimeError(f"Fact table is missing required hourly-example inputs: {sorted(missing_columns)}")

source = spark.table(SOURCE_TABLE).filter(F.col("respondent") == RESPONDENT)
bounds = source.agg(
    F.min("timestamp_utc").alias("first_timestamp"),
    F.max("timestamp_utc").alias("last_timestamp"),
).first()
if bounds["first_timestamp"] is None or bounds["last_timestamp"] is None:
    raise RuntimeError(f"No source demand rows found for {RESPONDENT} in {SOURCE_TABLE}.")

first_origin = _as_utc(bounds["first_timestamp"]) + pd.Timedelta(hours=HISTORY_HOURS)
last_origin = _as_utc(bounds["last_timestamp"]) - pd.Timedelta(hours=MAX_HORIZON_HOURS)
if first_origin > last_origin:
    raise RuntimeError(
        f"Insufficient continuous demand history for rolling examples: "
        f"first eligible origin {first_origin}, latest eligible origin {last_origin}."
    )

source_columns = sorted(required_columns)
total_written = 0
chunk_number = 0
table_exists = spark.catalog.tableExists(TARGET_TABLE)
last_materialized_origin = None
if table_exists:
    last_materialized_origin = (
        spark.table(TARGET_TABLE)
        .filter(
            (F.col("respondent") == RESPONDENT)
            & (F.col("horizon_hours") == HORIZON_H24)
            & (F.col("feature_set") == FEATURE_SET_DEMAND_FOUNDATION)
        )
        .agg(F.max("forecast_origin_utc").alias("last_origin"))
        .first()["last_origin"]
    )

chunk_start = first_origin
if last_materialized_origin is not None:
    stored_origin = _as_utc(last_materialized_origin)
    rebuild_anchor = min(stored_origin, last_origin)
    chunk_start = max(
        first_origin,
        rebuild_anchor - pd.Timedelta(days=CORRECTION_LOOKBACK_DAYS),
    )
    print(
        f"Incremental example rebuild from {chunk_start} through {last_origin}; "
        f"the {CORRECTION_LOOKBACK_DAYS}-day overlap refreshes recent corrections and a missed run catches up from its stored watermark."
    )
else:
    print(f"Initial example build from {first_origin} through {last_origin}.")
rebuild_start = chunk_start

while chunk_start <= last_origin:
    chunk_number += 1
    chunk_end = min(chunk_start + pd.Timedelta(days=CHUNK_DAYS), last_origin + pd.Timedelta(hours=1))
    input_start = chunk_start - pd.Timedelta(hours=HISTORY_HOURS)
    input_end = chunk_end + pd.Timedelta(hours=MAX_HORIZON_HOURS)
    demand_pdf = (
        source.filter(
            (F.col("timestamp_utc") >= F.lit(input_start.to_pydatetime()))
            & (F.col("timestamp_utc") < F.lit(input_end.to_pydatetime()))
        )
        .select(*source_columns)
        .orderBy("timestamp_utc")
        .toPandas()
    )
    examples = generate_forecast_examples_df(demand_pdf, step_hours=1)
    if not examples.empty:
        origins = pd.to_datetime(examples["forecast_origin_utc"], utc=True)
        examples = examples.loc[(origins >= chunk_start) & (origins < chunk_end)].copy()
    if not examples.empty or table_exists:
        incoming = (
            spark.createDataFrame(examples)
            if not examples.empty
            else spark.table(TARGET_TABLE).limit(0)
        )
        if not table_exists:
            incoming.write.format("delta").mode("overwrite").partitionBy("horizon_hours").saveAsTable(TARGET_TABLE)
            table_exists = True
        else:
            # Origin ranges are disjoint across chunks. Replacing only this
            # chunk's respondent/feature/time slice is idempotent on reruns and
            # avoids a growing full-table MERGE scan for every 30-day batch.
            replace_predicate = (
                f"respondent = '{RESPONDENT}' "
                f"AND feature_set = '{FEATURE_SET_DEMAND_FOUNDATION}' "
                f"AND horizon_hours = {HORIZON_H24} "
                f"AND forecast_origin_utc >= TIMESTAMP '{chunk_start:%Y-%m-%d %H:%M:%S}' "
                f"AND forecast_origin_utc < TIMESTAMP '{chunk_end:%Y-%m-%d %H:%M:%S}'"
            )
            incoming.write.format("delta").mode("overwrite").option(
                "replaceWhere", replace_predicate
            ).saveAsTable(TARGET_TABLE)
        total_written += len(examples)
    print(
        f"Hourly example chunk {chunk_number}: origins {chunk_start} to "
        f"{chunk_end - pd.Timedelta(hours=1)}; {len(examples):,} rows materialized."
    )
    chunk_start = chunk_end

if not table_exists:
    raise RuntimeError("No rolling examples were generated; inspect history continuity and data quality.")
print(
    f"Rolling hourly examples updated in {TARGET_TABLE}: {total_written:,} rows rewritten "
    f"across {chunk_number:,} chunks from {rebuild_start} through {last_origin}."
)
