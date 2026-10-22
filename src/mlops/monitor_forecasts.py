# Databricks notebook source
"""Materialize post-issue forecast errors after actual demand arrives.

This separates forecast issuance from evaluation.  It prevents the dashboard
from implying that a forecast was evaluated before the target hour occurred.
"""

from pyspark.sql import functions as F
from delta.tables import DeltaTable
import os
import sys

_repo_root = os.getcwd()
for _ in range(5):
    if os.path.isdir(os.path.join(_repo_root, "src")):
        if _repo_root not in sys.path:
            sys.path.insert(0, _repo_root)
        break
    _parent = os.path.dirname(_repo_root)
    if _parent == _repo_root:
        break
    _repo_root = _parent

from src.shared.delta_merge import merge_source_columns

CATALOG = "main"
FORECASTS = f"{CATALOG}.gold.forecast_evaluation"
FACT = f"{CATALOG}.gold.fact_hourly_electricity_demand"
METRICS = f"{CATALOG}.gold.forecast_monitoring_daily"
MONITOR_LOOKBACK_DAYS = 7  # Actual-demand correction window plus H24 lead and catch-up margin
INITIAL_MONITOR_LOOKBACK_DAYS = 180  # Bounded first materialization when no metrics watermark exists


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
    metrics_exist = spark.catalog.tableExists(METRICS)
    last_metric_date = None
    if metrics_exist:
        last_metric_date = (
            spark.table(METRICS)
            .agg(F.max("target_date_utc").alias("last_target_date"))
            .first()["last_target_date"]
        )
    if last_metric_date is None:
        target_cutoff = F.current_timestamp() - F.expr(
            f"INTERVAL {INITIAL_MONITOR_LOOKBACK_DAYS} DAYS"
        )
    else:
        # Use the older of the saved watermark and now. A long outage widens
        # the replay range instead of silently skipping matured forecasts.
        target_cutoff = F.least(
            F.to_timestamp(F.lit(last_metric_date)), F.current_timestamp()
        ) - F.expr(f"INTERVAL {MONITOR_LOOKBACK_DAYS} DAYS")
    origin_cutoff = target_cutoff - F.expr("INTERVAL 24 HOURS")
    issued = (
        spark.table(FORECASTS)
        .filter(
            (F.col("horizon_hours") == 24)
            & (F.col("forecast_origin_utc") >= origin_cutoff)
        )
    )
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
        .filter(
            F.col("actual_demand_mw").isNotNull()
            & (F.col("timestamp_utc") >= target_cutoff)
        )
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
    if not metrics_exist:
        daily.write.format("delta").mode("overwrite").saveAsTable(METRICS)
        print(f"Initialized {METRICS} with the recent matured forecast evaluation window.")
    else:
        target = DeltaTable.forName(spark, METRICS)
        match = (
            "t.model_run_id = s.model_run_id AND "
            "t.horizon_hours = s.horizon_hours AND "
            "t.lead_hours = s.lead_hours AND "
            "t.target_date_utc = s.target_date_utc"
        )
        merge_source_columns(
            target,
            daily,
            match,
            target_alias="t",
            source_alias="s",
        )
        print(f"Updated {METRICS} for the recent matured H24 evaluation window; older results were retained.")
