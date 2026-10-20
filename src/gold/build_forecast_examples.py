# Databricks notebook source
"""Build point-in-time training examples for the two forecast products.

This replaces the former target-timestamp feature table as the ML source.
The builder uses exact timestamps, not row offsets: missing hours invalidate an
example instead of silently changing the meaning of a lag or rolling window.
"""

import os
import sys

_root = os.getcwd()
for _ in range(5):
    if os.path.isdir(os.path.join(_root, "configs")):
        sys.path.insert(0, _root)
        break
    _root = os.path.dirname(_root)

from src.shared.forecast_contract import (
    EXAMPLE_GRAIN_KEYS,
)
from src.gold.forecast_example_features import generate_forecast_examples_df


if __name__ == "__main__" and "spark" in globals():
    from delta.tables import DeltaTable
    from pyspark.sql import functions as F

    spark.conf.set("spark.sql.session.timeZone", "UTC")
    source, target = "main.gold.fact_hourly_electricity_demand", "main.gold.forecast_examples"
    pdf = spark.table(source).filter(F.col("respondent") == "ERCO").select(
        "respondent", "timestamp_utc", "actual_demand_mw", "day_ahead_forecast_mw", "is_holiday", "is_grid_emergency"
    ).toPandas()
    examples = generate_forecast_examples_df(pdf)
    if examples.empty:
        raise ValueError("No valid forecast examples: inspect source continuity and demand quality.")
    incoming = spark.createDataFrame(examples)
    if spark.catalog.tableExists(target):
        existing_columns = set(spark.table(target).columns)
        if not set(incoming.columns).issubset(existing_columns):
            # Feature-contract changes require a one-time full rebuild from the
            # Gold fact table; a Delta MERGE cannot populate a newly added field
            # for old rows safely without schema evolution being enabled.
            incoming.write.format("delta").mode("overwrite").option("overwriteSchema", "true").partitionBy(
                "horizon_hours"
            ).saveAsTable(target)
            print("Forecast-example schema changed; rebuilt the table from the current Gold fact history.")
        else:
            DeltaTable.forName(spark, target).alias("t").merge(
                incoming.alias("s"), " AND ".join(f"t.{key} = s.{key}" for key in EXAMPLE_GRAIN_KEYS)
            ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
    else:
        incoming.write.format("delta").mode("overwrite").partitionBy("horizon_hours").saveAsTable(target)
    print(f"Built {len(examples):,} point-in-time forecast examples in {target}.")
