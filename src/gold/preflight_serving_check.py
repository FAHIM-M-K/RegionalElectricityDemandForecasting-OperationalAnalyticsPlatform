# Databricks notebook source
"""Read-only preflight for the hourly forecast pipeline.

Checks the data products consumed by the hourly conformation and scorer. This
task runs after EIA Bronze/Silver refresh, before Silver conformation and Gold
materialization, so a stale Gold table cannot mask a healthy fresh input.
"""

import os
import sys
from datetime import datetime, timezone

_root = os.getcwd()
for _ in range(5):
    if os.path.isdir(os.path.join(_root, "configs")):
        sys.path.insert(0, _root)
        break
    _root = os.path.dirname(_root)

from src.shared.forecast_contract import (
    DEMAND_MAX_VALID_MW,
    DEMAND_MIN_VALID_MW,
    MAX_FAIL_ORIGIN_HOURS,
    MAX_STALE_ORIGIN_HOURS,
)

CATALOG = "main"
DEMAND_TABLE = f"{CATALOG}.silver.electricity_demand_hourly"
CALENDAR_TABLE = f"{CATALOG}.bronze.calendar_reference"
EXPERIMENT_NAME = "/Shared/ERCOT_MultiHorizon_Forecast_Platform"
REQUIRED_HISTORY_POINTS = 337  # origin plus the exact lag_336 timestamp


def check_preflight(spark_session, mlflow_client=None, experiment_name: str = EXPERIMENT_NAME) -> dict:
    """Fail with actionable diagnostics when hourly serving prerequisites fail."""
    report = {"status": "HEALTHY", "checks": []}
    for table in (DEMAND_TABLE, CALENDAR_TABLE):
        exists = spark_session.catalog.tableExists(table)
        report["checks"].append({"check": f"table_exists:{table}", "passed": exists})
        if not exists:
            raise RuntimeError(f"Preflight failure: Required source table {table} does not exist.")

    from pyspark.sql import functions as F

    demand = spark_session.table(DEMAND_TABLE)
    latest = (
        demand.filter((F.col("respondent") == "ERCO") & F.col("actual_demand_mw").isNotNull())
        .orderBy(F.col("timestamp_utc").desc())
        .limit(1)
        .select("timestamp_utc", "actual_demand_mw")
        .first()
    )
    if latest is None:
        raise RuntimeError(f"Preflight failure: {DEMAND_TABLE} has no non-null ERCO actual demand.")

    latest_ts = latest["timestamp_utc"]
    if latest_ts.tzinfo is None:
        latest_ts = latest_ts.replace(tzinfo=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - latest_ts).total_seconds() / 3600.0
    report["latest_observation_utc"] = latest_ts.isoformat()
    report["observation_age_hours"] = round(age_hours, 2)
    if age_hours < 0:
        raise RuntimeError(f"Preflight failure: Latest actual demand is {abs(age_hours):.1f}h in the future.")
    if age_hours > MAX_FAIL_ORIGIN_HOURS:
        raise RuntimeError(
            f"Preflight failure: Latest actual demand is {age_hours:.1f}h old; "
            f"maximum permitted age is {MAX_FAIL_ORIGIN_HOURS}h."
        )
    report["checks"].append({
        "check": "source_freshness",
        "passed": True,
        "warning": (f"Input is {age_hours:.1f}h old; forecast will be marked stale."
                    if age_hours > MAX_STALE_ORIGIN_HOURS else None),
    })

    # Validate the exact history shape the scorer consumes, rather than relying
    # on a row count over a broad interval that can hide gaps or duplicates.
    recent = (
        demand.filter((F.col("respondent") == "ERCO") & F.col("actual_demand_mw").isNotNull())
        .orderBy(F.col("timestamp_utc").desc())
        .limit(REQUIRED_HISTORY_POINTS)
        .select("timestamp_utc", "actual_demand_mw")
        .collect()
    )
    if len(recent) != REQUIRED_HISTORY_POINTS:
        raise RuntimeError(
            f"Preflight failure: Scorer requires {REQUIRED_HISTORY_POINTS} recent actuals "
            f"(origin through origin-336h); found {len(recent)}."
        )
    chronological = list(reversed(recent))
    for previous, current in zip(chronological, chronological[1:]):
        gap_hours = (current["timestamp_utc"] - previous["timestamp_utc"]).total_seconds() / 3600.0
        if gap_hours != 1:
            raise RuntimeError(
                f"Preflight failure: Demand history is not hourly-contiguous near "
                f"{current['timestamp_utc']} (gap={gap_hours:g}h)."
            )
    invalid = [
        row for row in chronological
        if row["actual_demand_mw"] is None
        or not DEMAND_MIN_VALID_MW <= float(row["actual_demand_mw"]) <= DEMAND_MAX_VALID_MW
    ]
    if invalid:
        raise RuntimeError(
            f"Preflight failure: {len(invalid)} of the latest {REQUIRED_HISTORY_POINTS} "
            "actuals fall outside configured physical demand bounds."
        )
    report["checks"].append({"check": "exact_contiguous_337_point_history", "passed": True})

    calendar = spark_session.table(CALENDAR_TABLE)
    calendar_end = calendar.select(F.max(F.to_date("date")).alias("max_date")).first()["max_date"]
    # Convert on the driver to avoid Spark/session timezone coercion.
    from datetime import timedelta
    from zoneinfo import ZoneInfo
    if latest_ts.tzinfo is None:
        latest_ts = latest_ts.replace(tzinfo=timezone.utc)
    required_local_date = (
        latest_ts.astimezone(ZoneInfo("America/Chicago")) + timedelta(hours=168)
    ).date()
    if calendar_end is None or calendar_end < required_local_date:
        raise RuntimeError(
            f"Preflight failure: Calendar ends {calendar_end}; it must cover through "
            f"{required_local_date} for the H168 forecast."
        )
    report["checks"].append({"check": "calendar_covers_h168", "passed": True})

    if mlflow_client is None:
        try:
            import mlflow
            mlflow_client = mlflow.tracking.MlflowClient()
            experiment = mlflow.get_experiment_by_name(experiment_name)
        except Exception as exc:
            raise RuntimeError(f"Preflight failure: Cannot access MLflow production models: {exc}") from exc
        if experiment is None:
            raise RuntimeError(f"Preflight failure: MLflow experiment {experiment_name!r} does not exist.")
        experiment_id = experiment.experiment_id
    else:
        experiment = mlflow_client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise RuntimeError(f"Preflight failure: MLflow experiment {experiment_name!r} does not exist.")
        experiment_id = experiment.experiment_id

    production = {}
    for horizon in (24, 168):
        runs = mlflow_client.search_runs(
            [experiment_id],
            filter_string=f"tags.pipeline_role = 'PRODUCTION' AND tags.target_horizon = 'H{horizon}'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        production[horizon] = bool(runs)
    # H24 has no safe fallback; H168's explicit weekly-naive fallback is supported.
    if not production[24]:
        raise RuntimeError("Preflight failure: No promoted H24 production model is available.")
    report["checks"].append({"check": "production_h24_model", "passed": True})
    report["checks"].append({
        "check": "production_h168_model_or_documented_baseline",
        "passed": True,
        "warning": None if production[168] else "No H168 production model; scorer will use its labeled weekly-naive baseline.",
    })
    return report


if __name__ == "__main__" and "spark" in globals():
    status = check_preflight(spark)
    print("Hourly Forecast Preflight Passed:")
    for check in status["checks"]:
        print(f"  - {check['check']}: {check.get('warning') or 'PASSED'}")
