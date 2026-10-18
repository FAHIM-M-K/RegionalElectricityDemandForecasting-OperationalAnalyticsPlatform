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


if __name__ == "__main__" and "spark" in globals():
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    issued = spark.table(FORECASTS)
    actuals = spark.table(FACT).select("respondent", "timestamp_utc", "actual_demand_mw")
    evaluated = (
        issued.join(
            actuals,
            (issued.respondent == actuals.respondent)
            & (issued.target_timestamp_utc == actuals.timestamp_utc),
            "inner",
        )
        .select(issued["*"], actuals.actual_demand_mw.alias("actual_demand_mw"))
        # A timestamp match is not enough: EIA can publish an hourly row before
        # its actual value is available. Keep those forecasts pending.
        .filter(
            F.col("actual_demand_mw").isNotNull()
            & F.col("predicted_demand_mw").isNotNull()
        )
        .withColumn(
            "absolute_error_mw",
            F.abs(F.col("predicted_demand_mw") - F.col("actual_demand_mw")),
        )
    )
    daily = evaluated.groupBy("model_run_id", "horizon_hours", "lead_hours",
                              F.to_date("target_timestamp_utc").alias("target_date_utc")).agg(
        F.count("*").alias("evaluated_rows"), F.avg("absolute_error_mw").alias("mae_mw"),
        F.sqrt(F.avg(F.pow(F.col("predicted_demand_mw") - F.col("actual_demand_mw"), 2))).alias("rmse_mw"),
        F.avg(F.col("predicted_demand_mw") - F.col("actual_demand_mw")).alias("bias_mw"),
    ).withColumn("computed_at_utc", F.current_timestamp())
    daily.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(METRICS)
    evaluated_count = evaluated.count()
    print(
        f"Materialized {daily.count():,} daily forecast-monitoring rows from "
        f"{evaluated_count:,} forecasts with non-null actual demand in {METRICS}."
    )
