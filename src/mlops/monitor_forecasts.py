# Databricks notebook source
"""Materialize post-issue forecast errors after actual demand arrives.

This separates forecast issuance from evaluation.  It prevents the dashboard
from implying that a forecast was evaluated before the target hour occurred.
"""

from pyspark.sql import functions as F

CATALOG = "main"
FORECASTS = f"{CATALOG}.gold.forecast_evaluation"
FACT = f"{CATALOG}.gold.fact_hourly_electricity_demand"
METRICS = f"{CATALOG}.gold.forecast_monitoring_daily"


def _assert_unique(df, keys, label):
    duplicate = (
        df.groupBy(*keys)
        .count()
        .filter(F.col("count") > 1)
        .limit(1)
        .collect()
    )
    if duplicate:
        key_values = {key: duplicate[0][key] for key in keys}
        raise RuntimeError(f"Duplicate {label} key found: {key_values}")


if __name__ == "__main__" and "spark" in globals():
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    issued = spark.table(FORECASTS)
    # Older stored issues may predate forecast_issue_id. Fall back to the full
    # identity available in that schema while preserving distinct re-issues.
    if "forecast_issue_id" in issued.columns:
        issued = issued.withColumn(
            "_monitor_issue_key",
            F.coalesce(
                F.col("forecast_issue_id"),
                F.concat_ws(
                    "|", F.col("model_run_id"), F.col("horizon_hours").cast("string"),
                    F.col("forecast_origin_utc").cast("string"), F.col("issued_at_utc").cast("string"),
                ),
            ),
        )
    else:
        issued = issued.withColumn(
            "_monitor_issue_key",
            F.concat_ws(
                "|", F.col("model_run_id"), F.col("horizon_hours").cast("string"),
                F.col("forecast_origin_utc").cast("string"), F.col("issued_at_utc").cast("string"),
            ),
        )
    _assert_unique(
        issued,
        [
            "respondent",
            "_monitor_issue_key",
            "horizon_hours",
            "target_timestamp_utc",
        ],
        "forecast evaluation",
    )
    actuals = (
        spark.table(FACT)
        .select("respondent", "timestamp_utc", "actual_demand_mw")
        .filter(F.col("actual_demand_mw").isNotNull())
        # Compare hour instants in UTC rather than relying on implicit casts
        # between timestamp and timestamp_ntz columns from different tables.
        .withColumn("_actual_epoch_seconds", F.unix_timestamp("timestamp_utc"))
    )
    _assert_unique(actuals, ["respondent", "timestamp_utc"], "actual demand")
    issued_with_key = issued.withColumn(
        "_target_epoch_seconds", F.unix_timestamp("target_timestamp_utc")
    )
    evaluated = (
        issued_with_key.join(
            actuals,
            (issued_with_key.respondent == actuals.respondent)
            & (issued_with_key._target_epoch_seconds == actuals._actual_epoch_seconds),
            "inner",
        )
        .select(issued_with_key["*"], actuals.actual_demand_mw.alias("actual_demand_mw"))
        .drop("_target_epoch_seconds")
        .filter(F.col("predicted_demand_mw").isNotNull())
        .withColumn(
            "absolute_error_mw",
            F.abs(F.col("predicted_demand_mw") - F.col("actual_demand_mw")),
        )
    )
    if evaluated.limit(1).count() == 0:
        raise RuntimeError(
            f"No issued forecasts matched non-null actual demand in {FACT}; "
            f"leaving {METRICS} unchanged. Check UTC hourly timestamps and respondent keys."
        )
    daily = evaluated.groupBy(
        "model_run_id",
        "horizon_hours",
        "lead_hours",
        F.to_date("target_timestamp_utc").alias("target_date_utc"),
    ).agg(
        F.count("*").alias("evaluated_rows"),
        F.countDistinct("target_timestamp_utc").alias("unique_targets"),
        F.countDistinct("forecast_origin_utc").alias("issue_origins_count"),
        F.avg("absolute_error_mw").alias("mae_mw"),
        F.sqrt(
            F.avg(F.pow(F.col("predicted_demand_mw") - F.col("actual_demand_mw"), 2))
        ).alias("rmse_mw"),
        F.avg(F.col("predicted_demand_mw") - F.col("actual_demand_mw")).alias("bias_mw"),
    ).withColumn("computed_at_utc", F.current_timestamp())
    daily.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(METRICS)

    # Read counts from the small materialized metrics table; counting the source
    # DataFrames here would replay the full forecast/actual join and aggregation.
    totals = spark.table(METRICS).agg(
        F.count("*").alias("daily_rows"),
        F.sum("evaluated_rows").alias("evaluated_forecasts"),
    ).first()
    print(
        f"Materialized {totals['daily_rows']:,} daily forecast-monitoring rows from "
        f"{totals['evaluated_forecasts'] or 0:,} forecasts with non-null actual demand in {METRICS}."
    )
