# Databricks notebook source
"""Build issue-time training examples for hourly rolling forecast research.

Writes a separate Gold table so the existing daily-origin training table and
the production model workflow keep their current contract. Demand history is
loaded in bounded 30-day chunks with the exact 336-hour lookback and 168-hour
future label halo needed to build each chunk's hourly H24/H168 examples.
"""

import os
import sys
from datetime import timedelta

import pandas as pd
from delta.tables import DeltaTable
from pyspark.sql import functions as F

CATALOG = "main"
SOURCE_TABLE = f"{CATALOG}.gold.fact_hourly_electricity_demand"
TARGET_TABLE = f"{CATALOG}.gold.forecast_examples_hourly"
RESPONDENT = "ERCO"
CHUNK_DAYS = 30
HISTORY_HOURS = 336
MAX_HORIZON_HOURS = 168

spark.conf.set("spark.sql.session.timeZone", "UTC")
_root = os.getcwd()
for _ in range(5):
    if os.path.isdir(os.path.join(_root, "configs")):
        if _root not in sys.path:
            sys.path.insert(0, _root)
        break
    _root = os.path.dirname(_root)

from src.gold.build_forecast_examples import generate_forecast_examples_df
from src.shared.forecast_contract import EXAMPLE_GRAIN_KEYS


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
grain_condition = " AND ".join(f"t.`{key}` = s.`{key}`" for key in EXAMPLE_GRAIN_KEYS)
chunk_start = first_origin
total_written = 0
chunk_number = 0
table_exists = spark.catalog.tableExists(TARGET_TABLE)

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
    if not examples.empty:
        incoming = spark.createDataFrame(examples)
        if not table_exists:
            incoming.write.format("delta").mode("overwrite").partitionBy("horizon_hours").saveAsTable(TARGET_TABLE)
            table_exists = True
        else:
            DeltaTable.forName(spark, TARGET_TABLE).alias("t").merge(
                incoming.alias("s"), grain_condition
            ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
        total_written += len(examples)
    print(
        f"Hourly example chunk {chunk_number}: origins {chunk_start} to "
        f"{chunk_end - pd.Timedelta(hours=1)}; {len(examples):,} rows merged."
    )
    chunk_start = chunk_end

if not table_exists:
    raise RuntimeError("No rolling examples were generated; inspect history continuity and data quality.")

summary = spark.table(TARGET_TABLE).agg(
    F.count("*").alias("rows"),
    F.countDistinct("forecast_origin_utc").alias("origins"),
    F.min("forecast_origin_utc").alias("first_origin"),
    F.max("forecast_origin_utc").alias("last_origin"),
    F.sum(F.col("is_valid_training_example").cast("int")).alias("valid_rows"),
).first()
print(
    f"Rolling hourly examples are ready in {TARGET_TABLE}: "
    f"{summary['rows']:,} rows, {summary['origins']:,} hourly origins, "
    f"{summary['valid_rows']:,} valid training rows, "
    f"{summary['first_origin']} through {summary['last_origin']}. "
    f"Rows merged in this run: {total_written:,}."
)
