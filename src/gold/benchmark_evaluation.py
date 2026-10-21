"""Fair issue-time-matched evaluation of the active rolling H24 forecast against EIA."""

import json
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.shared.forecast_contract import HORIZON_H24


def _require_unique(frame: pd.DataFrame, keys: List[str], label: str) -> None:
    """Reject duplicate evaluation keys instead of silently overweighting them."""
    if frame.empty:
        return
    duplicates = frame.duplicated(keys, keep=False)
    if duplicates.any():
        sample = frame.loc[duplicates, keys].head(3).to_dict("records")
        raise ValueError(f"{label} contains duplicate rows at key {keys}: {sample}")


def select_as_of_benchmark_vintages(
    df_vintages: pd.DataFrame,
    as_of_time_utc: pd.Timestamp,
    provider: str = "EIA",
    respondent: str = "ERCO",
) -> pd.DataFrame:
    """Select each target's latest benchmark vintage actually available at issue time."""
    if df_vintages.empty:
        return pd.DataFrame(columns=df_vintages.columns)

    frame = df_vintages.copy()
    frame["target_timestamp_utc"] = pd.to_datetime(frame["target_timestamp_utc"], utc=True)
    frame["first_observed_at_utc"] = pd.to_datetime(frame["first_observed_at_utc"], utc=True)
    as_of = pd.to_datetime(as_of_time_utc, utc=True)
    eligible = frame[
        (frame["benchmark_provider"] == provider)
        & (frame["respondent"] == respondent)
        & (frame["first_observed_at_utc"] <= as_of)
        & frame["forecast_mw"].notna()
    ].copy()
    if "published_at_utc" in eligible:
        published = pd.to_datetime(eligible["published_at_utc"], utc=True)
        eligible = eligible[published.isna() | (published <= as_of)]
    if eligible.empty:
        return pd.DataFrame(columns=frame.columns)
    return eligible.sort_values(
        ["target_timestamp_utc", "first_observed_at_utc", "vintage_id"]
    ).drop_duplicates("target_timestamp_utc", keep="last")


def match_h24_and_eia(
    df_h24_forecasts: pd.DataFrame,
    df_eia_vintages: pd.DataFrame,
    df_actuals: pd.DataFrame,
    headline_policy: str = "lead_24",
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Match model, EIA and matured actuals on target hour and issue-time availability."""
    forecasts = df_h24_forecasts[
        df_h24_forecasts["horizon_hours"] == HORIZON_H24
    ].copy()
    forecasts["target_timestamp_utc"] = pd.to_datetime(forecasts["target_timestamp_utc"], utc=True)
    forecasts["issued_at_utc"] = pd.to_datetime(forecasts["issued_at_utc"], utc=True)
    issue_keys = ["respondent", "horizon_hours", "forecast_origin_utc", "issued_at_utc", "target_timestamp_utc"]
    if "forecast_issue_id" in forecasts:
        issue_keys.insert(1, "forecast_issue_id")
    _require_unique(forecasts, issue_keys, "H24 forecasts")

    actuals = df_actuals[["respondent", "timestamp_utc", "actual_demand_mw"]].copy()
    actuals["timestamp_utc"] = pd.to_datetime(actuals["timestamp_utc"], utc=True)
    actuals = actuals[actuals["actual_demand_mw"].notna()]
    _require_unique(actuals, ["respondent", "timestamp_utc"], "Actual demand")

    matched_issues = []
    issue_keys = ["forecast_origin_utc", "issued_at_utc"]
    if "forecast_issue_id" in forecasts:
        issue_keys.append("forecast_issue_id")
    for _, issue in forecasts.groupby(issue_keys, dropna=False):
        vintages = select_as_of_benchmark_vintages(df_eia_vintages, issue["issued_at_utc"].iloc[0])
        if vintages.empty:
            continue
        matched = issue.merge(
            vintages[["target_timestamp_utc", "forecast_mw"]].rename(
                columns={"forecast_mw": "eia_forecast_mw"}
            ),
            on="target_timestamp_utc", how="inner",
        ).merge(
            actuals.rename(columns={"timestamp_utc": "target_timestamp_utc"}),
            on=["respondent", "target_timestamp_utc"], how="inner",
        )
        if not matched.empty:
            matched_issues.append(matched)

    pending = {
        "status": "PROSPECTIVE_PENDING_MATURED_DATA",
        "matched_rows": 0,
        "unique_targets": 0,
        "model_mae_mw": None,
        "eia_mae_mw": None,
        "model_rmse_mw": None,
        "eia_rmse_mw": None,
        "model_bias_mw": None,
        "eia_bias_mw": None,
        "notice": "No H24 targets have both an issue-time-available EIA vintage and a matured actual; metrics are pending, not zero.",
    }
    if not matched_issues:
        return pd.DataFrame(), pending

    matched = pd.concat(matched_issues, ignore_index=True)
    matched["model_error_mw"] = matched["predicted_demand_mw"] - matched["actual_demand_mw"]
    matched["model_abs_error_mw"] = matched["model_error_mw"].abs()
    matched["eia_error_mw"] = matched["eia_forecast_mw"] - matched["actual_demand_mw"]
    matched["eia_abs_error_mw"] = matched["eia_error_mw"].abs()

    if headline_policy == "lead_24":
        headline = matched[matched["lead_hours"] == 24]
    elif headline_policy == "all_leads":
        sort_by = ["target_timestamp_utc", "issued_at_utc", "forecast_origin_utc"]
        if "forecast_issue_id" in matched:
            sort_by.append("forecast_issue_id")
        headline = matched.sort_values(sort_by).drop_duplicates("target_timestamp_utc", keep="last")
    else:
        raise ValueError("headline_policy must be 'lead_24' or 'all_leads'.")

    if headline.empty:
        pending["status"] = "PENDING_LEAD_24_MATCHES"
        pending["notice"] = "Other matched leads exist, but no matured lead-24 targets are available for the H24 headline."
        return matched, pending

    return matched, {
        "status": "EVALUATED",
        "matched_rows": int(len(headline)),
        "unique_targets": int(headline["target_timestamp_utc"].nunique()),
        "model_mae_mw": round(float(headline["model_abs_error_mw"].mean()), 2),
        "eia_mae_mw": round(float(headline["eia_abs_error_mw"].mean()), 2),
        "model_rmse_mw": round(float(np.sqrt((headline["model_error_mw"] ** 2).mean())), 2),
        "eia_rmse_mw": round(float(np.sqrt((headline["eia_error_mw"] ** 2).mean())), 2),
        "model_bias_mw": round(float(headline["model_error_mw"].mean()), 2),
        "eia_bias_mw": round(float(headline["eia_error_mw"].mean()), 2),
    }


def run_benchmark_evaluation(spark_session) -> dict:
    """Materialize H24/EIA metrics for the latest served model run."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        DoubleType, IntegerType, LongType, StringType, StructField, StructType, TimestampType,
    )

    catalog = "main"
    forecasts_table = f"{catalog}.gold.forecast_evaluation"
    serving_table = f"{catalog}.gold.forecast_latest"
    actuals_table = f"{catalog}.gold.fact_hourly_electricity_demand"
    eia_table = f"{catalog}.bronze.eia_forecast_vintages"
    summary_table = f"{catalog}.gold.forecast_benchmark_metrics"
    detail_table = f"{catalog}.gold.forecast_benchmark_evaluation"
    for table in (forecasts_table, serving_table, actuals_table):
        if not spark_session.catalog.tableExists(table):
            raise RuntimeError(f"Benchmark evaluation requires table {table}.")

    latest = (
        spark_session.table(serving_table)
        .filter(F.col("horizon_hours") == HORIZON_H24)
        .select("model_run_id").distinct().limit(2).collect()
    )
    if len(latest) != 1:
        raise RuntimeError("Benchmark evaluation requires exactly one active H24 served model run.")
    run_id = str(latest[0]["model_run_id"])
    window_start = F.current_timestamp() - F.expr("INTERVAL 180 DAYS")
    forecasts = (
        spark_session.table(forecasts_table)
        .filter(
            (F.col("horizon_hours") == HORIZON_H24)
            & (F.col("model_run_id") == run_id)
            & (F.col("forecast_origin_utc") >= window_start)
        )
        .select("respondent", "forecast_issue_id", "forecast_origin_utc", "issued_at_utc",
                "target_timestamp_utc", "horizon_hours", "lead_hours", "predicted_demand_mw")
        .toPandas()
    )
    actuals = (
        spark_session.table(actuals_table)
        .filter(F.col("actual_demand_mw").isNotNull() & (F.col("timestamp_utc") >= window_start))
        .select("respondent", "timestamp_utc", "actual_demand_mw")
        .toPandas()
    )
    for frame, columns in ((forecasts, ("forecast_origin_utc", "target_timestamp_utc", "issued_at_utc")),
                           (actuals, ("timestamp_utc",))):
        for column in columns:
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], utc=True)

    if spark_session.catalog.tableExists(eia_table):
        eia = (
            spark_session.table(eia_table)
            .filter(F.col("target_timestamp_utc") >= window_start)
            .select("benchmark_provider", "respondent", "vintage_id", "target_timestamp_utc",
                    "forecast_mw", "published_at_utc", "first_observed_at_utc")
            .toPandas()
        )
    else:
        eia = pd.DataFrame(columns=[
            "benchmark_provider", "respondent", "vintage_id", "target_timestamp_utc",
            "forecast_mw", "published_at_utc", "first_observed_at_utc",
        ])
    for column in ("target_timestamp_utc", "published_at_utc", "first_observed_at_utc"):
        if column in eia:
            eia[column] = pd.to_datetime(eia[column], utc=True)

    matched, metrics = match_h24_and_eia(forecasts, eia, actuals, headline_policy="lead_24")
    now = datetime.now(timezone.utc)
    summary = {
        "benchmark_provider": "EIA",
        "horizon_hours": HORIZON_H24,
        "model_run_id": run_id,
        "status": metrics["status"],
        "matched_rows": metrics["matched_rows"],
        "unique_targets": metrics["unique_targets"],
        "model_mae_mw": metrics["model_mae_mw"],
        "benchmark_mae_mw": metrics["eia_mae_mw"],
        "model_rmse_mw": metrics["model_rmse_mw"],
        "benchmark_rmse_mw": metrics["eia_rmse_mw"],
        "model_bias_mw": metrics["model_bias_mw"],
        "benchmark_bias_mw": metrics["eia_bias_mw"],
        "metrics_json": json.dumps(metrics, default=str, sort_keys=True),
        "computed_at_utc": now,
    }
    summary_schema = StructType([
        StructField("benchmark_provider", StringType(), False),
        StructField("horizon_hours", IntegerType(), False),
        StructField("model_run_id", StringType(), False),
        StructField("status", StringType(), False),
        StructField("matched_rows", LongType(), False),
        StructField("unique_targets", LongType(), False),
        StructField("model_mae_mw", DoubleType(), True),
        StructField("benchmark_mae_mw", DoubleType(), True),
        StructField("model_rmse_mw", DoubleType(), True),
        StructField("benchmark_rmse_mw", DoubleType(), True),
        StructField("model_bias_mw", DoubleType(), True),
        StructField("benchmark_bias_mw", DoubleType(), True),
        StructField("metrics_json", StringType(), False),
        StructField("computed_at_utc", TimestampType(), False),
    ])
    spark_session.createDataFrame([summary], schema=summary_schema).write.format("delta").mode(
        "overwrite"
    ).option("overwriteSchema", "true").saveAsTable(summary_table)

    if not matched.empty:
        matched.insert(0, "benchmark_provider", "EIA")
        matched.insert(1, "horizon_hours_evaluated", HORIZON_H24)
        for column in matched.columns:
            if pd.api.types.is_datetime64_any_dtype(matched[column]):
                matched[column] = pd.to_datetime(matched[column], utc=True).dt.tz_localize(None)
        spark_session.createDataFrame(matched).write.format("delta").mode("overwrite").option(
            "overwriteSchema", "true"
        ).saveAsTable(detail_table)
    elif spark_session.catalog.tableExists(detail_table):
        from delta.tables import DeltaTable
        DeltaTable.forName(spark_session, detail_table).delete()

    print(
        f"Materialized H24/EIA benchmark status in {summary_table}: {metrics['status']}; "
        f"matched detail rows={len(matched):,}."
    )
    return {"summary_table": summary_table, "detail_table": detail_table, "rows": [summary]}


if __name__ == "__main__" and "spark" in globals():
    run_benchmark_evaluation(spark)
