# Databricks notebook source
"""Attach archived, issue-time weather vintages to H24 demand examples.

Only complete weather paths whose conservative publication time is no later
than the simulated issue cutoff are eligible. This produces a separate model
training source and leaves the demand-only foundation untouched.
"""

from pyspark.sql import functions as F

from src.shared.forecast_contract import (
    FEATURE_SCHEMA_VERSION, FEATURE_SET_DEMAND_FOUNDATION,
    FEATURE_SET_OPERATIONAL, WEATHER_FORECAST_FEATURES,
)

CATALOG = "main"
EXAMPLES_TABLE = f"{CATALOG}.gold.forecast_examples_hourly"
WEATHER_TABLE = f"{CATALOG}.silver.weather_backtest_features"
TARGET_TABLE = f"{CATALOG}.gold.forecast_examples_hourly_weather"
EXCLUSIONS_TABLE = f"{CATALOG}.silver.weather_backtest_exclusions"
HORIZON_HOURS = 24
ARCHIVE_START_UTC = "2025-01-15 00:00:00"

spark.conf.set("spark.sql.session.timeZone", "UTC")
if not spark.catalog.tableExists(EXAMPLES_TABLE):
    raise RuntimeError(f"Rolling demand examples are missing: {EXAMPLES_TABLE}")
if not spark.catalog.tableExists(WEATHER_TABLE):
    raise RuntimeError(
        f"Archived issue-time weather is missing: {WEATHER_TABLE}. "
        "Run/resume the bounded weather vintage backfill before building weather-trained models."
    )

examples = (
    spark.table(EXAMPLES_TABLE)
    .filter((F.col("respondent") == "ERCO") & (F.col("horizon_hours") == HORIZON_HOURS))
    .filter(F.col("feature_set") == FEATURE_SET_DEMAND_FOUNDATION)
    .filter(F.col("forecast_origin_utc") >= F.lit(ARCHIVE_START_UTC).cast("timestamp"))
)
weather = spark.table(WEATHER_TABLE).filter(
    F.col("target_timestamp_utc") <= F.col("forecast_origin_utc") + F.expr("INTERVAL 24 HOURS")
)
weather_keys = ["respondent", "forecast_origin_utc", "target_timestamp_utc"]
if weather.groupBy(*weather_keys).count().filter(F.col("count") > 1).limit(1).count():
    raise RuntimeError("Archived weather has duplicate respondent/origin/target keys.")
required_weather = set(weather_keys) | {
    "weather_available_at_utc", "weather_issue_time_proxy_utc", *WEATHER_FORECAST_FEATURES,
}
missing_weather = required_weather - set(weather.columns)
if missing_weather:
    raise RuntimeError(f"Archived weather table lacks required fields: {sorted(missing_weather)}")

weather = weather.select(*sorted(required_weather)).filter(
    F.col("weather_available_at_utc") <= F.col("weather_issue_time_proxy_utc")
)
weather_paths = weather.groupBy("respondent", "forecast_origin_utc").agg(
    F.countDistinct("target_timestamp_utc").alias("weather_target_hours"),
    F.sum(F.when(
        F.col("weighted_temperature_f").isNull()
        | F.col("cdh_65_forecast").isNull()
        | F.col("hdh_65_forecast").isNull(), 1
    ).otherwise(0)).alias("weather_null_feature_rows"),
)
incomplete_weather = weather_paths.filter(
    (F.col("weather_target_hours") != HORIZON_HOURS) | (F.col("weather_null_feature_rows") != 0)
)
if incomplete_weather.limit(1).count():
    first_incomplete = incomplete_weather.orderBy("forecast_origin_utc").first()
    raise RuntimeError(
        f"Archived weather path for {first_incomplete['forecast_origin_utc']} is incomplete "
        f"({first_incomplete['weather_target_hours']} target hours, "
        f"{first_incomplete['weather_null_feature_rows']} rows with null features); "
        "repair/resume the vintage backfill instead of dropping the affected origin."
    )
expected_origins = examples.select("respondent", "forecast_origin_utc").distinct()
covered_origins = weather.select("respondent", "forecast_origin_utc").distinct()
missing_origins = expected_origins.join(covered_origins, ["respondent", "forecast_origin_utc"], "left_anti")
missing_origin_count = missing_origins.count()
excluded_origin_count = 0
if missing_origin_count and not spark.catalog.tableExists(EXCLUSIONS_TABLE):
    first_missing = missing_origins.orderBy("forecast_origin_utc").first()["forecast_origin_utc"]
    raise RuntimeError(
        f"{missing_origin_count:,} rolling origins have no weather and are not explained by "
        f"{EXCLUSIONS_TABLE}; first missing origin is {first_missing}. Resume the backfill."
    )
if spark.catalog.tableExists(EXCLUSIONS_TABLE):
    excluded = spark.table(EXCLUSIONS_TABLE).select(
        "respondent", "forecast_origin_utc", "exclusion_reason"
    ).distinct()
    unclassified_missing = missing_origins.join(
        excluded.select("respondent", "forecast_origin_utc"),
        ["respondent", "forecast_origin_utc"], "left_anti"
    )
    if unclassified_missing.limit(1).count():
        first_missing = unclassified_missing.orderBy("forecast_origin_utc").first()["forecast_origin_utc"]
        raise RuntimeError(
            f"Missing weather origin {first_missing} is not classified in {EXCLUSIONS_TABLE}; "
            "the backfill must resolve or explicitly record every origin."
        )
    excluded_origin_count = missing_origins.count()
    unexpected_exclusions = excluded.join(expected_origins, ["respondent", "forecast_origin_utc"], "left_anti")
    if unexpected_exclusions.limit(1).count():
        first_extra = unexpected_exclusions.orderBy("forecast_origin_utc").first()["forecast_origin_utc"]
        print(f"Ignoring {first_extra} archived exclusion record outside the current example range.")

print(
    f"Weather coverage: {covered_origins.count():,}/{expected_origins.count():,} origins complete; "
    f"{excluded_origin_count:,} origins explicitly excluded for unavailable/incomplete issue-time weather. "
    "Weather candidates and no-weather controls will use this same matched origin set."
)

eligible_examples = examples.join(covered_origins, ["respondent", "forecast_origin_utc"], "inner")
joined = eligible_examples.join(weather, on=weather_keys, how="inner")
expected_row_counts = eligible_examples.groupBy("respondent", "forecast_origin_utc", "horizon_hours").agg(
    F.countDistinct("target_timestamp_utc").alias("expected_example_rows")
)
matched_row_counts = joined.groupBy("respondent", "forecast_origin_utc", "horizon_hours").agg(
    F.countDistinct("target_timestamp_utc").alias("matched_weather_rows"),
    F.sum(F.when(F.col("weighted_temperature_f").isNotNull(), 1).otherwise(0)).alias("temperature_rows"),
    F.sum(F.when(F.col("cdh_65_forecast").isNotNull() & F.col("hdh_65_forecast").isNotNull(), 1).otherwise(0)).alias("degree_hour_rows"),
)
coverage = expected_row_counts.join(
    matched_row_counts, ["respondent", "forecast_origin_utc", "horizon_hours"], "left"
)
incomplete_examples = coverage.filter(
    F.col("matched_weather_rows").isNull()
    | (F.col("expected_example_rows") != F.col("matched_weather_rows"))
    | (F.col("matched_weather_rows") != F.col("temperature_rows"))
    | (F.col("matched_weather_rows") != F.col("degree_hour_rows"))
)
if incomplete_examples.limit(1).count():
    first_incomplete = incomplete_examples.orderBy("forecast_origin_utc").first()
    raise RuntimeError(
        f"Weather did not match every demand example for {first_incomplete['forecast_origin_utc']} "
        f"H{first_incomplete['horizon_hours']}: expected {first_incomplete['expected_example_rows']}, "
        f"matched {first_incomplete['matched_weather_rows']}. No examples were dropped silently."
    )

target = (
    joined.drop("weather_issue_time_proxy_utc")
    .withColumn("feature_set", F.lit(FEATURE_SET_OPERATIONAL))
    .withColumn("feature_schema_version", F.lit(FEATURE_SCHEMA_VERSION))
)
if target.groupBy(
    "respondent", "forecast_origin_utc", "target_timestamp_utc", "horizon_hours", "feature_set"
).count().filter(F.col("count") > 1).limit(1).count():
    raise RuntimeError("Weather training source violates the forecast-example natural key.")

if not target.limit(1).count():
    raise RuntimeError(
        "No complete point-in-time weather training examples exist. Resume the archived "
        "weather backfill; no no-weather training fallback is allowed."
    )
# Replace only the active H24 partition. Retired H168 training rows are
# historical data and must not disappear as a side effect of this refactor.
if spark.catalog.tableExists(TARGET_TABLE):
    existing_columns = set(spark.table(TARGET_TABLE).columns)
    if existing_columns != set(target.columns):
        raise RuntimeError(
            f"Training table schema mismatch for {TARGET_TABLE}; refusing to overwrite it. "
            "Handle any schema migration explicitly so historical partitions are preserved."
        )
    target.write.format("delta").mode("overwrite").option(
        "replaceWhere", f"horizon_hours = {HORIZON_HOURS}"
    ).saveAsTable(TARGET_TABLE)
else:
    target.write.format("delta").mode("overwrite").partitionBy("horizon_hours").saveAsTable(TARGET_TABLE)

summary = spark.table(TARGET_TABLE).groupBy("horizon_hours").agg(
    F.count("*").alias("rows"), F.countDistinct("forecast_origin_utc").alias("complete_origins"),
    F.min("forecast_origin_utc").alias("first_origin"), F.max("forecast_origin_utc").alias("last_origin"),
).orderBy("horizon_hours")
for row in summary.collect():
    print(
        f"H{row['horizon_hours']} point-in-time weather training source: "
        f"{row['rows']:,} rows across {row['complete_origins']:,} complete origins, "
        f"{row['first_origin']} through {row['last_origin']} UTC."
    )
