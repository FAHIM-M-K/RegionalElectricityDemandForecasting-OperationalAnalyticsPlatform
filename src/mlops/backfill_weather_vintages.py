# Databricks notebook source
# COMMAND ----------
# MAGIC %pip install lightgbm==4.7.0
# COMMAND ----------
"""One-time, isolated backfill of issue-time ECMWF weather for model research.

This is deliberately not part of the daily production job. It reconstructs a
conservative historical issue cutoff (forecast origin + 65 minutes), selects
the latest 00Z/12Z ECMWF IFS run available by that cutoff (init + 6 hours),
and stores weighted four-station temperature for each H168 example target.
The issue-time offset is an explicit proxy based on the observed daily run;
the later comparison must report that assumption. If the selected archived
run has a gap, only the immediately previous eligible cycle is considered as
a fallback; older runs are not chased because they are less representative.
"""

import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd
import lightgbm as lgb
from delta.tables import DeltaTable
from pyspark.sql import functions as F

CATALOG = "main"
# This historical backfill now follows the hourly issue cadence used in service.
EXAMPLES_TABLE = f"{CATALOG}.gold.forecast_examples_hourly"
TARGET_TABLE = f"{CATALOG}.silver.weather_backtest_features"
API_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
MODEL = "ecmwf_ifs"
ARCHIVE_START_UTC = dt.datetime(2025, 1, 15, tzinfo=dt.timezone.utc)
ISSUE_DELAY_MINUTES = 65
CONSERVATIVE_PUBLICATION_DELAY_HOURS = 6
HORIZON_HOURS = 168
FORECAST_DAYS = 9
REQUEST_PAUSE_SECONDS = 0.15
CHECKPOINT_EVERY_RUNS = 20
# Preview is the default. After reviewing the exact request count in Databricks,
# set RUN_API_BACKFILL=True to fetch at most this many distinct ECMWF vintages.
RUN_API_BACKFILL = False
MAX_NEW_RUNS_PER_EXECUTION = 250


class DailyApiQuotaExceeded(RuntimeError):
    """Open-Meteo's daily request quota is exhausted; retrying cannot help today."""

spark.conf.set("spark.sql.session.timeZone", "UTC")

_root = os.getcwd()
for _ in range(5):
    if os.path.isdir(os.path.join(_root, "configs")):
        if _root not in sys.path:
            sys.path.insert(0, _root)
        break
    _parent = os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent

from configs.loader import load_platform_config
from src.mlops.backtest_multihorizon import (
    compute_metrics,
    evaluate_promotion_gate,
    generate_chronological_splits,
)
from src.mlops.weather_training_data import attach_weather_asof_issue
from src.shared.forecast_contract import features_for_horizon

config = load_platform_config("ercot_config.json")
stations = config["weather"]["stations"]
weights = {item["station_id"]: float(item["weight"]) for item in stations}
if len(stations) != 4 or abs(sum(weights.values()) - 1.0) > 1e-6:
    raise ValueError("Expected the four configured ERCOT weather stations with weights summing to 1.")


def eligible_run(origin: dt.datetime) -> tuple[dt.datetime, dt.datetime, dt.datetime]:
    """Choose the latest long-horizon 00Z/12Z run available by issue cutoff."""
    issue_proxy = origin + dt.timedelta(minutes=ISSUE_DELAY_MINUTES)
    latest_eligible_init = issue_proxy - dt.timedelta(hours=CONSERVATIVE_PUBLICATION_DELAY_HOURS)
    cycle_hour = 12 if latest_eligible_init.hour >= 12 else 0
    run_init = latest_eligible_init.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
    if run_init > latest_eligible_init:
        run_init -= dt.timedelta(hours=12)
    available_at = run_init + dt.timedelta(hours=CONSERVATIVE_PUBLICATION_DELAY_HOURS)
    if available_at > issue_proxy:
        raise RuntimeError(f"Selected weather run is not available by issue proxy for {origin}.")
    return run_init, available_at, issue_proxy


def request_run(run_init: dt.datetime) -> list[dict]:
    """Fetch one ECMWF vintage for all four configured stations."""
    params = {
        "latitude": ",".join(str(item["latitude"]) for item in stations),
        "longitude": ",".join(str(item["longitude"]) for item in stations),
        "hourly": "temperature_2m",
        "timezone": "UTC",
        "forecast_days": FORECAST_DAYS,
        "models": MODEL,
        # The Single Runs API requires ISO minutes, with no seconds or offset.
        "run": run_init.strftime("%Y-%m-%dT%H:%M"),
    }
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "ERCOT-weather-backtest/1.0"})
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")[:1500]
            if error.code == 429 and "daily api request limit exceeded" in body.lower():
                raise DailyApiQuotaExceeded(
                    f"Open-Meteo daily request quota is exhausted at ECMWF run "
                    f"{run_init:%Y-%m-%dT%H:%M}."
                ) from error
            if error.code < 500 and error.code != 429:
                raise RuntimeError(
                    f"Single Runs API rejected ECMWF vintage {run_init:%Y-%m-%dT%H:%M} "
                    f"(HTTP {error.code}): {body}"
                ) from error
            if attempt == 3:
                raise RuntimeError(
                    f"Single Runs API failed for {run_init:%Y-%m-%dT%H:%M} "
                    f"after {attempt} attempts (HTTP {error.code}): {body}"
                ) from error
            time.sleep(2 * attempt)
        except Exception as error:
            if attempt == 3:
                raise RuntimeError(
                    f"Single Runs API failed for {run_init:%Y-%m-%dT%H:%M} "
                    f"after {attempt} attempts: {error}"
                ) from error
            time.sleep(2 * attempt)
    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(f"Single Runs API rejected vintage {run_init}: {payload.get('reason')}")
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or len(payload) != len(stations):
        raise RuntimeError(
            f"Expected {len(stations)} station responses for {run_init}; "
            f"received {len(payload) if isinstance(payload, list) else type(payload).__name__}."
        )
    return payload


# Only origins with labeled H168 examples are included. No future realized
# weather is read: the sole weather input is each selected archived forecast.
origin_pdf = (
    spark.table(EXAMPLES_TABLE)
    .filter((F.col("respondent") == "ERCO") & (F.col("horizon_hours") == HORIZON_HOURS))
    .filter(F.col("forecast_origin_utc") >= F.lit(ARCHIVE_START_UTC))
    .select("forecast_origin_utc")
    .distinct()
    .orderBy("forecast_origin_utc")
    .toPandas()
)
if origin_pdf.empty:
    raise RuntimeError(f"No H{HORIZON_HOURS} example origins found after {ARCHIVE_START_UTC}.")
origins = [pd.to_datetime(value, utc=True).to_pydatetime() for value in origin_pdf["forecast_origin_utc"]]

# Cache by ECMWF initialization. Hourly origins share the same 00Z/12Z run for
# about half a day, and each request bundles all four configured locations.
selected = {origin: eligible_run(origin) for origin in origins}
origins_by_run: dict[dt.datetime, list[dt.datetime]] = {}
for origin, selection in selected.items():
    origins_by_run.setdefault(selection[0], []).append(origin)
run_inits = sorted(origins_by_run)
print(
    f"Preparing {len(origins):,} H168 origins using {len(run_inits):,} unique archived ECMWF runs "
    f"from {origins[0]} onward. Each request bundles all four stations; checkpoints occur every "
    f"{CHECKPOINT_EVERY_RUNS} runs and can be resumed safely."
)
existing_origins = set()
if spark.catalog.tableExists(TARGET_TABLE):
    existing_origins = {
        pd.to_datetime(row["forecast_origin_utc"], utc=True).to_pydatetime()
        for row in spark.table(TARGET_TABLE)
        .filter(F.col("forecast_origin_utc") >= F.lit(ARCHIVE_START_UTC))
        .groupBy("forecast_origin_utc").count()
        .filter(F.col("count") == HORIZON_HOURS)
        .select("forecast_origin_utc").collect()
    }
pending_origins = {origin for origin in origins if origin not in existing_origins}
pending_run_inits = [run_init for run_init in run_inits if any(
    origin in pending_origins for origin in origins_by_run[run_init]
)]
print(
    f"Weather archive preview: {len(origins):,} hourly issue origins, "
    f"{len(pending_run_inits):,} new ECMWF API requests required, "
    f"maximum {MAX_NEW_RUNS_PER_EXECUTION:,} requests in one execution. "
    "Each request bundles all four configured stations."
)
if not RUN_API_BACKFILL:
    dbutils.notebook.exit(
        "Preview complete. No Open-Meteo request was made. Review the count, then set "
        "RUN_API_BACKFILL=True to begin a resumable batch."
    )

selected_run_inits = pending_run_inits[:MAX_NEW_RUNS_PER_EXECUTION]
if len(pending_run_inits) > len(selected_run_inits):
    print(
        f"This execution will process {len(selected_run_inits):,} requests. "
        f"{len(pending_run_inits) - len(selected_run_inits):,} requests remain for later resumable batches."
    )
pending_rows = []
processed_since_checkpoint = 0
run_temperature_cache = {}


def checkpoint(rows_to_write: list[dict]) -> None:
    if not rows_to_write:
        return
    batch_pdf = pd.DataFrame(rows_to_write)
    incoming_batch = spark.createDataFrame(batch_pdf)
    if not spark.catalog.tableExists(TARGET_TABLE):
        incoming_batch.write.format("delta").mode("overwrite").saveAsTable(TARGET_TABLE)
    else:
        DeltaTable.forName(spark, TARGET_TABLE).alias("target").merge(
            incoming_batch.alias("source"),
            "target.respondent = source.respondent "
            "AND target.forecast_origin_utc = source.forecast_origin_utc "
            "AND target.target_timestamp_utc = source.target_timestamp_utc",
        ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()


def temperature_map_for_run(run_init: dt.datetime) -> dict:
    """Load aligned station-hour data once per vintage and retain nulls for diagnosis."""
    global pending_rows
    if run_init in run_temperature_cache:
        return run_temperature_cache[run_init]
    try:
        payloads = request_run(run_init)
    except DailyApiQuotaExceeded as error:
        # Save only fully built origins. An incomplete origin is still local to
        # build_complete_origin and is never appended to this checkpoint batch.
        checkpoint(pending_rows)
        if pending_rows:
            existing_origins.update(row["forecast_origin_utc"] for row in pending_rows)
            print(
                f"Quota stop: checkpointed {len(pending_rows):,} complete weather rows "
                f"({len(existing_origins):,} complete origins are now resumable)."
            )
            pending_rows = []
        raise RuntimeError(
            f"{error} The completed work has been saved. Stop this notebook and rerun it "
            "after the provider quota resets; it will resume from completed 168-hour origins."
        ) from error
    times_by_station = []
    temps_by_station = []
    for station, payload in zip(stations, payloads):
        hourly = payload.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        if not times or len(times) != len(temps):
            raise RuntimeError(f"Malformed hourly temperature response for {station['station_id']} / {run_init}.")
        times_by_station.append([pd.to_datetime(value, utc=True).to_pydatetime() for value in times])
        temps_by_station.append(temps)
    if any(times != times_by_station[0] for times in times_by_station[1:]):
        raise RuntimeError(f"Station target timestamps differ within ECMWF run {run_init}.")
    run_temperature_cache[run_init] = {
        station["station_id"]: dict(zip(times_by_station[index], temps_by_station[index]))
        for index, station in enumerate(stations)
    }
    return run_temperature_cache[run_init]


def build_complete_origin(origin: dt.datetime, run_init: dt.datetime) -> tuple[list[dict] | None, dict | None]:
    """Build a full H168 weather vintage, returning a precise gap if it is incomplete."""
    available_at = run_init + dt.timedelta(hours=CONSERVATIVE_PUBLICATION_DELAY_HOURS)
    issue_proxy = selected[origin][2]
    temperatures = temperature_map_for_run(run_init)
    origin_rows = []
    for lead in range(1, HORIZON_HOURS + 1):
        target = origin + dt.timedelta(hours=lead)
        station_values = {
            station["station_id"]: temperatures[station["station_id"]].get(target)
            for station in stations
        }
        missing = [
            {"station": station_id,
             "reason": "timestamp_absent" if target not in temperatures[station_id] else "temperature_null"}
            for station_id, value in station_values.items() if value is None
        ]
        if missing:
            return None, {"target": target, "missing": missing, "run_init": run_init}
        station_values = {key: float(value) for key, value in station_values.items()}
        weighted_c = sum(station_values[key] * weights[key] for key in weights)
        weighted_f = weighted_c * 9.0 / 5.0 + 32.0
        origin_rows.append({
            "respondent": "ERCO",
            "forecast_origin_utc": origin,
            "target_timestamp_utc": target,
            "lead_hours": lead,
            "weather_model": MODEL,
            "weather_run_init_utc": run_init,
            "weather_available_at_utc": available_at,
            "weather_issue_time_proxy_utc": issue_proxy,
            "weighted_temperature_f": round(weighted_f, 3),
            "cdh_65_forecast": round(max(weighted_f - 65.0, 0.0), 3),
            "hdh_65_forecast": round(max(65.0 - weighted_f, 0.0), 3),
            **{f"temperature_{key}_c": value for key, value in station_values.items()},
        })
    return origin_rows, None


for request_index, run_init in enumerate(selected_run_inits, start=1):
    run_origins = [origin for origin in origins_by_run[run_init] if origin not in existing_origins]
    if not run_origins:
        continue
    for origin in run_origins:
        origin_rows, gap = build_complete_origin(origin, run_init)
        if gap is not None:
            # Never fill with realized weather or a later forecast. Try older
            # ECMWF cycles only when their conservative publication time still
            # precedes the historical issue-time proxy.
            fallback_init = run_init - dt.timedelta(hours=12)
            if fallback_init + dt.timedelta(hours=CONSERVATIVE_PUBLICATION_DELAY_HOURS) <= selected[origin][2]:
                fallback_rows, fallback_gap = build_complete_origin(origin, fallback_init)
                if fallback_gap is None:
                    print(
                        f"Used prior eligible ECMWF vintage {fallback_init} for origin {origin}; "
                        f"latest eligible vintage {run_init} has {gap['missing']} at {gap['target']}."
                    )
                    origin_rows = fallback_rows
                else:
                    raise RuntimeError(
                        f"Selected ECMWF run {run_init} has a gap ({gap['missing']} at {gap['target']}) "
                        f"and the immediately previous eligible run {fallback_init} also has a gap "
                        f"({fallback_gap['missing']} at {fallback_gap['target']}). "
                        "No partial origin was saved; inspect these two API responses before retrying."
                    )
            else:
                raise RuntimeError(
                    f"Selected ECMWF run {run_init} has a gap ({gap['missing']} at {gap['target']}) "
                    f"for origin {origin}, and no previous cycle is available by the issue-time proxy. "
                    "No partial origin was saved."
                )
        if len(origin_rows) != HORIZON_HOURS:
            raise RuntimeError(f"Origin {origin} produced {len(origin_rows)} rows; expected {HORIZON_HOURS}.")
        pending_rows.extend(origin_rows)
    if request_index % 50 == 0 or request_index == len(selected_run_inits):
        print(
            f"Processed {request_index:,}/{len(selected_run_inits):,} runs in this batch; "
            f"{len(pending_rows):,} rows are waiting for the next checkpoint."
        )
    processed_since_checkpoint += 1
    if processed_since_checkpoint >= CHECKPOINT_EVERY_RUNS:
        checkpoint(pending_rows)
        existing_origins.update(row["forecast_origin_utc"] for row in pending_rows)
        print(f"Saved {len(pending_rows):,} rows; completed vintages can be resumed safely.")
        pending_rows = []
        processed_since_checkpoint = 0
    time.sleep(REQUEST_PAUSE_SECONDS)

checkpoint(pending_rows)
if pending_rows:
    existing_origins.update(row["forecast_origin_utc"] for row in pending_rows)
    print(f"Saved final batch of {len(pending_rows):,} rows.")
remaining_origins = [origin for origin in origins if origin not in existing_origins]
if remaining_origins:
    dbutils.notebook.exit(
        f"Weather backfill batch completed. {len(remaining_origins):,} hourly origins remain; "
        "rerun after reviewing the next preview to resume. Candidate evaluation is deferred "
        "until all requested origins have complete weather paths."
    )
if not spark.catalog.tableExists(TARGET_TABLE):
    raise RuntimeError("No weather backtest vintages were materialized.")

summary = spark.table(TARGET_TABLE).filter(
    F.col("forecast_origin_utc") >= F.lit(ARCHIVE_START_UTC)
).agg(
    F.count("*").alias("weather_rows"),
    F.countDistinct("forecast_origin_utc").alias("origins"),
    F.countDistinct("target_timestamp_utc").alias("targets"),
    F.min("forecast_origin_utc").alias("first_origin"),
    F.max("forecast_origin_utc").alias("last_origin"),
    F.sum(F.when(F.col("weather_available_at_utc") <= F.col("weather_issue_time_proxy_utc"), 1).otherwise(0))
     .alias("available_by_issue_proxy"),
).first()
print(
    f"Materialized {summary['weather_rows']:,} experimental rows for {summary['origins']:,} origins "
    f"in {TARGET_TABLE}, {summary['first_origin']} through {summary['last_origin']}. "
    f"Rows available by the conservative issue-time proxy: {summary['available_by_issue_proxy']:,}. "
    "This is a research table only; it does not modify production features or model registration."
)

# Compare on the same origin/target rows and chronological folds. Rows without
# a complete archived weather vintage are excluded from BOTH control and
# candidate, preventing a coverage-driven sample difference from masquerading
# as a weather skill change.
example_columns = sorted(set(
    [
        "respondent", "forecast_origin_utc", "target_timestamp_utc", "horizon_hours",
        "actual_demand_mw", "seasonal_naive_mw", "is_valid_training_example",
    ]
    + features_for_horizon(24) + features_for_horizon(168)
))
examples_pdf = (
    spark.table(EXAMPLES_TABLE)
    .filter(F.col("respondent") == "ERCO")
    .filter(F.col("horizon_hours").isin(24, 168))
    .filter(F.col("forecast_origin_utc") >= F.lit(ARCHIVE_START_UTC))
    .select(*example_columns)
    .toPandas()
)
weather_pdf = (
    spark.table(TARGET_TABLE)
    .filter(F.col("forecast_origin_utc") >= F.lit(ARCHIVE_START_UTC))
    .select(
        "respondent", "forecast_origin_utc", "target_timestamp_utc", "weather_run_init_utc",
        "weather_available_at_utc", "weighted_temperature_f", "cdh_65_forecast", "hdh_65_forecast",
    )
    .toPandas()
)
origin_counts = weather_pdf.groupby("forecast_origin_utc").size()
incomplete_origins = origin_counts[origin_counts != HORIZON_HOURS]
if not incomplete_origins.empty or weather_pdf[[
    "weighted_temperature_f", "cdh_65_forecast", "hdh_65_forecast"
]].isna().any().any():
    raise RuntimeError(
        "The experimental weather table has incomplete 168-hour origins or null weather features; "
        "resume the backfill before comparing models."
    )
for frame in (examples_pdf, weather_pdf):
    frame["forecast_origin_utc"] = pd.to_datetime(frame["forecast_origin_utc"], utc=True)
    frame["target_timestamp_utc"] = pd.to_datetime(frame["target_timestamp_utc"], utc=True)

matched_pdf, weather_coverage_pdf = attach_weather_asof_issue(
    examples_pdf,
    weather_pdf,
    issue_delay_minutes=ISSUE_DELAY_MINUTES,
)
complete_issue_keys = weather_coverage_pdf.loc[
    weather_coverage_pdf["complete_weather_horizon"],
    ["respondent", "forecast_origin_utc", "horizon_hours"],
]
matched_pdf = matched_pdf.merge(
    complete_issue_keys,
    on=["respondent", "forecast_origin_utc", "horizon_hours"],
    how="inner",
    validate="many_to_one",
)
if matched_pdf.empty:
    raise RuntimeError(
        "No complete issue-time weather curves are available for a matched comparison. "
        "The as-of selector excludes later vintages and partial horizons."
    )
print(
    "As-of weather matching retained "
    f"{complete_issue_keys['forecast_origin_utc'].nunique():,} complete issue/horizon groups "
    f"across {matched_pdf['forecast_origin_utc'].nunique():,} origins; later/unavailable vintages "
    "and incomplete curves were excluded."
)
weather_features = ["weighted_temperature_f", "cdh_65_forecast", "hdh_65_forecast"]
model_params = {
    "objective": "regression_l1", "metric": "l1", "learning_rate": 0.05,
    "num_leaves": 31, "min_child_samples": 40, "feature_fraction": 0.85,
    "bagging_fraction": 0.85, "bagging_freq": 1, "seed": 42, "verbose": -1,
}


def fit_model(train: pd.DataFrame, validation: pd.DataFrame, feature_columns: list[str]):
    train_data = lgb.Dataset(train[feature_columns], label=train["residual_target_mw"])
    validation_data = lgb.Dataset(validation[feature_columns], label=validation["residual_target_mw"])
    return lgb.train(
        model_params, train_data, num_boost_round=1200, valid_sets=[validation_data],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )


def predict_demand(model, frame: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
    return frame["seasonal_naive_mw"] + model.predict(frame[feature_columns])


def report_pair(horizon: int) -> None:
    base_features = features_for_horizon(horizon)
    needed = list(dict.fromkeys(base_features + weather_features + [
        "actual_demand_mw", "seasonal_naive_mw", "is_valid_training_example",
        "forecast_origin_utc", "target_timestamp_utc", "hour_local", "lead_hours",
    ]))
    data = matched_pdf.loc[
        (matched_pdf["horizon_hours"] == horizon)
        & (matched_pdf["is_valid_training_example"] == True)
    ].copy()
    data = data.dropna(subset=needed).sort_values(
        ["forecast_origin_utc", "target_timestamp_utc"]
    )
    if data.empty:
        raise RuntimeError(f"No matched H{horizon} examples have complete weather and demand features.")
    data["residual_target_mw"] = data["actual_demand_mw"] - data["seasonal_naive_mw"]
    splits = generate_chronological_splits(data["forecast_origin_utc"], n_test_windows=4)
    selection_splits = [item for item in splits if item["kind"] == "selection"]
    if len(selection_splits) < 4:
        raise RuntimeError(
            f"H{horizon} has only {len(selection_splits)} usable chronological windows; "
            "the experiment requires all four selection windows."
        )

    selection_rows = []
    for index, split in enumerate(selection_splits, start=1):
        validation_start = split["test_start"] - pd.Timedelta(days=30)
        train = data[
            (data.forecast_origin_utc < validation_start)
            & (data.target_timestamp_utc < validation_start)
        ]
        validation = data[
            (data.forecast_origin_utc >= validation_start)
            & (data.forecast_origin_utc < split["test_start"])
            & (data.target_timestamp_utc < split["test_start"])
        ]
        test = data[
            (data.forecast_origin_utc >= split["test_start"])
            & (data.forecast_origin_utc < split["test_end"])
        ].copy()
        if train.empty or validation.empty or test.empty:
            raise RuntimeError(f"H{horizon} selection window {index} has an empty train/validation/test slice.")
        no_weather_model = fit_model(train, validation, base_features)
        weather_model = fit_model(train, validation, base_features + weather_features)
        peak = test["hour_local"].between(14, 18).to_numpy()
        no_weather_prediction = predict_demand(no_weather_model, test, base_features)
        weather_prediction = predict_demand(weather_model, test, base_features + weather_features)
        no_weather_metrics = compute_metrics(test["actual_demand_mw"], no_weather_prediction, peak)
        weather_metrics = compute_metrics(test["actual_demand_mw"], weather_prediction, peak)
        selection_rows.append({
            "candidate_mae": weather_metrics["mae"],
            "baseline_mae": no_weather_metrics["mae"],
            "candidate_peak_mae": weather_metrics["peak_mae"],
            "baseline_peak_mae": no_weather_metrics["peak_mae"],
        })
        print(
            f"H{horizon} selection {index}: same rows={len(test):,}; "
            f"no-weather MAE={no_weather_metrics['mae']:,.1f} MW, "
            f"weather MAE={weather_metrics['mae']:,.1f} MW; "
            f"peak MAE {no_weather_metrics['peak_mae']:,.1f} vs "
            f"{weather_metrics['peak_mae']:,.1f} MW."
        )

    gate = evaluate_promotion_gate(selection_rows)
    final = next(item for item in splits if item["kind"] == "final_holdout")
    validation_start = final["test_start"] - pd.Timedelta(days=30)
    train = data[
        (data.forecast_origin_utc < validation_start)
        & (data.target_timestamp_utc < validation_start)
    ]
    validation = data[
        (data.forecast_origin_utc >= validation_start)
        & (data.forecast_origin_utc < final["test_start"])
        & (data.target_timestamp_utc < final["test_start"])
    ]
    pre_holdout = data[
        (data.forecast_origin_utc < final["test_start"])
        & (data.target_timestamp_utc < final["test_start"])
    ]
    holdout = data[
        (data.forecast_origin_utc >= final["test_start"])
        & (data.forecast_origin_utc < final["test_end"])
    ].copy()
    if train.empty or validation.empty or pre_holdout.empty or holdout.empty:
        raise RuntimeError(f"H{horizon} final holdout has an empty train/validation/test slice.")
    control_selection_model = fit_model(train, validation, base_features)
    weather_selection_model = fit_model(train, validation, base_features + weather_features)
    control_rounds = control_selection_model.best_iteration
    weather_rounds = weather_selection_model.best_iteration
    no_weather_model = lgb.train(
        model_params, lgb.Dataset(pre_holdout[base_features], label=pre_holdout.residual_target_mw),
        num_boost_round=control_rounds,
    )
    weather_model = lgb.train(
        model_params,
        lgb.Dataset(pre_holdout[base_features + weather_features], label=pre_holdout.residual_target_mw),
        num_boost_round=weather_rounds,
    )
    peak = holdout["hour_local"].between(14, 18).to_numpy()
    control_prediction = predict_demand(no_weather_model, holdout, base_features)
    weather_prediction = predict_demand(weather_model, holdout, base_features + weather_features)
    control_metrics = compute_metrics(holdout.actual_demand_mw, control_prediction, peak)
    weather_metrics = compute_metrics(holdout.actual_demand_mw, weather_prediction, peak)
    print(
        f"H{horizon} selection summary: weather vs no-weather promotion-gate diagnostic "
        f"accepted={gate['accepted']} ({gate['reason']}); "
        f"selection MAE {gate['candidate_mae']:,.1f} vs {gate['baseline_mae']:,.1f} MW; "
        f"window win rate={gate['win_rate']:.0%}; "
        f"selection peak MAE {gate['candidate_peak_mae']:,.1f} vs "
        f"{gate['baseline_peak_mae']:,.1f} MW."
    )
    print(
        f"H{horizon} untouched final holdout: same rows={len(holdout):,}; "
        f"no-weather MAE={control_metrics['mae']:,.1f} MW, "
        f"weather MAE={weather_metrics['mae']:,.1f} MW; "
        f"peak MAE {control_metrics['peak_mae']:,.1f} vs "
        f"{weather_metrics['peak_mae']:,.1f} MW."
    )
    for label, lower, upper in (
        ("lead 1-24", 1, 24), ("lead 25-48", 25, 48), ("lead 49-168", 49, 168),
    ):
        bucket = holdout[holdout["lead_hours"].between(lower, upper)]
        if bucket.empty:
            continue
        print(
            f"  {label}: control MAE={compute_metrics(bucket.actual_demand_mw, control_prediction.loc[bucket.index])['mae']:,.1f} MW, "
            f"weather MAE={compute_metrics(bucket.actual_demand_mw, weather_prediction.loc[bucket.index])['mae']:,.1f} MW; "
            f"rows={len(bucket):,}."
        )
    print("Research-only result: do not promote or alter production models from this notebook.")


for horizon in (24, 168):
    report_pair(horizon)
