# Databricks notebook source
# COMMAND ----------
# MAGIC %pip install lightgbm==4.7.0
# COMMAND ----------
"""Issue H24/H168 forecasts, using an explicit weekly-naive fallback for H168.

H24 requires a promoted ML model. H168 uses its promoted ML model when one is
available; otherwise, it serves the weekly seasonal-naive benchmark with clear
baseline metadata until an H168 candidate passes validation.
"""

import math
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

_root = os.getcwd()
for _ in range(5):
    if os.path.isdir(os.path.join(_root, "configs")):
        sys.path.insert(0, _root)
        break
    _root = os.path.dirname(_root)

from src.shared.forecast_contract import (
    DEMAND_MAX_VALID_MW, DEMAND_MIN_VALID_MW, FEATURE_SCHEMA_VERSION,
    FEATURE_SET_OPERATIONAL, HORIZONS, features_for_horizon,
    generate_forecast_issue_id, validate_forecast_issue_frame,
    MAX_STALE_ORIGIN_HOURS, MAX_FAIL_ORIGIN_HOURS,
)

try:
    import mlflow
    import mlflow.lightgbm
    from delta.tables import DeltaTable
    from pyspark.sql import functions as F
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

CATALOG = "main"
EXPERIMENT_NAME = "/Shared/ERCOT_MultiHorizon_Forecast_Platform"
SOURCE_FACT_TABLE = f"{CATALOG}.gold.fact_hourly_electricity_demand"
TARGET_LATEST_TABLE = f"{CATALOG}.gold.forecast_latest"
TARGET_EVALUATION_TABLE = f"{CATALOG}.gold.forecast_evaluation"
H168_BASELINE_RUN_ID = "baseline:weekly-seasonal-naive:v1"



def _calendar(timestamp: pd.Timestamp, holiday_dates: set) -> dict:
    local = timestamp.tz_convert("America/Chicago")
    return {"hour_local": local.hour, "day_of_week_local": local.dayofweek,
            "is_weekend": int(local.dayofweek >= 5), "is_holiday": int(local.date() in holiday_dates),
            "sin_hour": round(math.sin(2 * math.pi * local.hour / 24), 6),
            "cos_hour": round(math.cos(2 * math.pi * local.hour / 24), 6),
            "sin_dow": round(math.sin(2 * math.pi * local.dayofweek / 7), 6),
            "cos_dow": round(math.cos(2 * math.pi * local.dayofweek / 7), 6)}


def _snapshot(frame: pd.DataFrame, origin: pd.Timestamp) -> dict:
    values = dict(zip(frame["timestamp_utc"], frame["actual_demand_mw"]))
    required = [origin, origin - pd.Timedelta(hours=24), origin - pd.Timedelta(hours=48),
                origin - pd.Timedelta(hours=168), origin - pd.Timedelta(hours=336)]
    missing = [value for value in required if pd.isna(values.get(value, np.nan))]
    if missing:
        raise ValueError(f"Cannot forecast from {origin}: required exact demand history is incomplete.")
    window24 = [values.get(origin - pd.Timedelta(hours=i), np.nan) for i in range(24)]
    window168 = [values.get(origin - pd.Timedelta(hours=i), np.nan) for i in range(168)]
    if any(pd.isna(window24)) or any(pd.isna(window168)):
        raise ValueError("Cannot forecast: trailing demand window contains missing hours.")
    if any(not DEMAND_MIN_VALID_MW <= float(value) <= DEMAND_MAX_VALID_MW for value in window168):
        raise ValueError("Cannot forecast: trailing demand window contains an invalid load value.")
    if any(not DEMAND_MIN_VALID_MW <= float(values[ts]) <= DEMAND_MAX_VALID_MW for ts in required):
        raise ValueError("Cannot forecast: required demand history contains an invalid load value.")
    return {"demand_origin": values[origin], "demand_lag_24h_origin": values[required[1]],
            "demand_lag_48h_origin": values[required[2]], "demand_lag_168h_origin": values[required[3]],
            "demand_lag_336h_origin": values[required[4]], "demand_rolling_avg_24h_origin": float(np.mean(window24)),
            "demand_rolling_std_24h_origin": float(np.std(window24)), "demand_rolling_avg_168h_origin": float(np.mean(window168))}


def generate_independent_forecasts(df_history: pd.DataFrame, models: dict, model_runs: dict, origin_utc=None,
                                  holiday_dates: set | None = None, calendar_dates: set | None = None,
                                  high_demand_threshold_mw: float = 75_000.0,
                                  issued_at_utc=None) -> pd.DataFrame:
    """Return direct predictions for every lead in H24 and H168 with issue identity and validation."""
    frame = df_history.copy()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    duplicated = frame.loc[
        frame.duplicated(["respondent", "timestamp_utc"], keep=False),
        ["respondent", "timestamp_utc"],
    ]
    if not duplicated.empty:
        examples = duplicated.drop_duplicates().head(5).astype(str).to_dict("records")
        raise ValueError(f"Duplicate demand keys in scoring history: {examples}")
    frame = frame.sort_values("timestamp_utc")
    if frame["respondent"].nunique() != 1:
        raise ValueError("Score one respondent per invocation.")
    origin = frame.timestamp_utc.max() if origin_utc is None else pd.to_datetime(origin_utc, utc=True)
    snapshot = _snapshot(frame, origin)
    demand_values = dict(zip(frame["timestamp_utc"], frame["actual_demand_mw"]))
    if holiday_dates is None or calendar_dates is None:
        raise ValueError("Forward calendar dates and holiday dates are required for scoring.")
    forecast_dates = {(origin + pd.Timedelta(hours=lead)).tz_convert("America/Chicago").date()
                       for horizon in HORIZONS for lead in range(1, horizon + 1)}
    missing_calendar = forecast_dates - set(calendar_dates)
    if missing_calendar:
        raise ValueError(f"Calendar reference does not cover forecast dates: {sorted(missing_calendar)}")
    respondent = str(frame.respondent.iloc[0])
    issued_at = datetime.now(timezone.utc) if issued_at_utc is None else pd.to_datetime(issued_at_utc, utc=True)

    # Calculate source data age and freshness status
    source_age_hours = round((issued_at - origin).total_seconds() / 3600.0, 2)
    freshness_status = "OPERATIONAL_FRESH" if source_age_hours <= MAX_STALE_ORIGIN_HOURS else "STALE_INPUT"

    rows = []
    for horizon in HORIZONS:
        model = models.get(horizon)
        run_id = model_runs.get(horizon)
        use_weekly_baseline = horizon == 168 and model is None and run_id == H168_BASELINE_RUN_ID
        if (model is None and not use_weekly_baseline) or not run_id:
            raise ValueError(f"No active production H{horizon} model is available; scoring stopped.")
        expected_features = features_for_horizon(horizon)
        if model is not None and hasattr(model, "feature_name") and model.feature_name() != expected_features:
            raise ValueError(f"Production H{horizon} model feature schema does not match {FEATURE_SCHEMA_VERSION}.")
        leads = list(range(1, horizon + 1))
        seasonal_lag_hours = 24 if horizon == 24 else 168
        feature_rows = []
        for lead in leads:
            target = origin + pd.Timedelta(hours=lead)
            seasonal_timestamp = target - pd.Timedelta(hours=seasonal_lag_hours)
            seasonal_value = demand_values.get(seasonal_timestamp, np.nan)
            if pd.isna(seasonal_value):
                raise ValueError(f"H{horizon} seasonal baseline is unavailable at lead {lead}; required history is missing.")
            feature_rows.append({"lead_hours": lead, "seasonal_naive_mw": float(seasonal_value),
                                 **_calendar(target, holiday_dates), **snapshot})
        residuals = (np.zeros(len(feature_rows), dtype=float) if use_weekly_baseline
                     else model.predict(pd.DataFrame(feature_rows)[expected_features]))
        predictions = np.asarray([row["seasonal_naive_mw"] for row in feature_rows], dtype=float) + residuals

        issue_id = generate_forecast_issue_id(respondent, horizon, origin, issued_at, run_id)

        for lead, prediction, feature in zip(leads, predictions, feature_rows):
            target = origin + pd.Timedelta(hours=lead)
            predicted = float(prediction)
            if not DEMAND_MIN_VALID_MW <= predicted <= DEMAND_MAX_VALID_MW:
                raise ValueError(f"H{horizon} prediction {predicted:.1f} MW is outside plausibility bounds.")
            local = target.tz_convert("America/Chicago")
            rows.append({
                "forecast_issue_id": issue_id,
                "respondent": respondent, "forecast_origin_utc": origin, "target_timestamp_utc": target,
                "issued_at_utc": issued_at, "horizon_hours": horizon, "lead_hours": lead,
                "timestamp_local": local.tz_localize(None), "date_local": local.date(),
                "hour_local": feature["hour_local"], "predicted_demand_mw": round(predicted, 1),
                "is_predicted_high_demand": bool(predicted >= high_demand_threshold_mw),
                "high_demand_threshold_mw": high_demand_threshold_mw,
                "freshness_status": freshness_status,
                "source_data_age_hours": source_age_hours,
                "feature_set": "weekly_seasonal_naive_baseline" if use_weekly_baseline else FEATURE_SET_OPERATIONAL,
                "feature_schema_version": "not_applicable_baseline" if use_weekly_baseline else FEATURE_SCHEMA_VERSION,
                "forecast_source": ("weekly_seasonal_naive_baseline" if use_weekly_baseline
                                    else "seasonal_residual_lightgbm"),
                "model_version": ("WeeklySeasonalNaive_Baseline_H168" if use_weekly_baseline
                                  else f"SeasonalResidual_LightGBM_H{horizon}"),
                "model_run_id": run_id,
            })

    result_df = pd.DataFrame(rows)
    validate_forecast_issue_frame(result_df)
    return result_df



def _production_models():
    if not MLFLOW_AVAILABLE:
        raise RuntimeError("MLflow is required for production scoring.")
    client, experiment = mlflow.tracking.MlflowClient(), mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment {EXPERIMENT_NAME!r} does not exist. "
            "Run the monthly model workflow (build_forecast_examples → retrain_model → validate_model) "
            "once before the daily scorer. Validation must promote production models for both H24 and H168."
        )
    models, runs = {}, {}
    for horizon in HORIZONS:
        found = client.search_runs([experiment.experiment_id],
                                   filter_string=f"tags.pipeline_role = 'PRODUCTION' AND tags.target_horizon = 'H{horizon}'",
                                   order_by=["start_time DESC"], max_results=1)
        if not found:
            if horizon == 168:
                models[horizon] = None
                runs[horizon] = H168_BASELINE_RUN_ID
                print("No promoted H168 ML model is available; issuing the explicitly labeled weekly seasonal-naive baseline.")
                continue
            raise RuntimeError(
                f"No production H{horizon} model found in MLflow experiment {EXPERIMENT_NAME!r}. "
                "The daily scorer requires a validated H24 model. Check validate_model output."
            )
        runs[horizon] = found[0].info.run_id
        trained_schema = found[0].data.params.get("feature_schema_version")
        if trained_schema != FEATURE_SCHEMA_VERSION:
            raise RuntimeError(
                f"Production H{horizon} model uses feature schema {trained_schema!r}, "
                f"but this scorer requires {FEATURE_SCHEMA_VERSION!r}. Run monthly retraining "
                "and promote a model trained with the current schema before daily scoring."
            )
        models[horizon] = mlflow.lightgbm.load_model(f"runs:/{runs[horizon]}/model_seasonal_residual_h{horizon}")
    return models, runs


if __name__ == "__main__" and "spark" in globals():
    from configs.loader import load_platform_config

    spark.conf.set("spark.sql.session.timeZone", "UTC")
    high_demand_threshold_mw = float(load_platform_config("ercot_config.json")["region"]["default_peak_threshold_mw"])
    history = (spark.table(SOURCE_FACT_TABLE)
               .filter((F.col("respondent") == "ERCO") & F.col("actual_demand_mw").isNotNull())
               .orderBy(F.col("timestamp_utc").desc()).limit(600).toPandas())
    latest_observation = pd.to_datetime(history["timestamp_utc"], utc=True).max()
    now_utc = datetime.now(timezone.utc)
    age_hours = (now_utc - latest_observation.to_pydatetime()).total_seconds() / 3600
    if age_hours < 0 or age_hours > MAX_FAIL_ORIGIN_HOURS:
        raise RuntimeError(f"Latest actual demand is {age_hours:.1f} hours old (exceeds {MAX_FAIL_ORIGIN_HOURS}h threshold); refusing to issue predictions.")
    if age_hours > MAX_STALE_ORIGIN_HOURS:
        print(f"Warning: Input actual demand is {age_hours:.1f} hours old; issuing with STALE_INPUT status.")

    calendar = spark.table(f"{CATALOG}.bronze.calendar_reference").select("date", "is_holiday").toPandas()
    holidays = set(pd.to_datetime(calendar.loc[calendar.is_holiday.astype(bool), "date"]).dt.date)
    calendar_dates = set(pd.to_datetime(calendar["date"]).dt.date)
    models, run_ids = _production_models()
    forecast = generate_independent_forecasts(
        history, models, run_ids, holiday_dates=holidays, calendar_dates=calendar_dates,
        high_demand_threshold_mw=high_demand_threshold_mw,
    )
    incoming = spark.createDataFrame(forecast)
    # Validate before persisting: pre-publish gate enforces complete curves (24 + 168 rows)
    assert len(forecast) == 192, f"Expected 192 forecast rows, found {len(forecast)}"

    # Serving table is atomically replaced: it contains only the newest issue.
    incoming.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(TARGET_LATEST_TABLE)

    # Evaluation table is append-only at issued-forecast grain.
    if spark.catalog.tableExists(TARGET_EVALUATION_TABLE):
        # Migrate additive issue/freshness fields in place before MERGE. Without
        # this, an older evaluation table rejects the new scorer's insert schema.
        existing_columns = set(spark.table(TARGET_EVALUATION_TABLE).columns)
        missing_columns = [field for field in incoming.schema.fields if field.name not in existing_columns]
        if missing_columns:
            ddl_columns = ", ".join(
                f"`{field.name}` {field.dataType.simpleString()}" for field in missing_columns
            )
            spark.sql(f"ALTER TABLE {TARGET_EVALUATION_TABLE} ADD COLUMNS ({ddl_columns})")
        DeltaTable.forName(spark, TARGET_EVALUATION_TABLE).alias("t").merge(
            incoming.alias("s"),
            "t.respondent=s.respondent AND t.forecast_issue_id=s.forecast_issue_id "
            "AND t.horizon_hours=s.horizon_hours AND t.target_timestamp_utc=s.target_timestamp_utc"
        ).whenNotMatchedInsertAll().execute()
    else:
        incoming.write.format("delta").mode("overwrite").saveAsTable(TARGET_EVALUATION_TABLE)
    print(f"Issued {len(forecast):,} direct forecast rows from {forecast.forecast_origin_utc.iloc[0]} (Issue: {forecast.forecast_issue_id.iloc[0]}).")
