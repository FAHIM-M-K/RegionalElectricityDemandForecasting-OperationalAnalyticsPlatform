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
the later comparison must report that assumption.
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
EXAMPLES_TABLE = f"{CATALOG}.gold.forecast_examples"
TARGET_TABLE = f"{CATALOG}.silver.weather_backtest_features"
API_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
MODEL = "ecmwf_ifs"
ARCHIVE_START_UTC = dt.datetime(2025, 1, 15, tzinfo=dt.timezone.utc)
ISSUE_DELAY_MINUTES = 65
CONSERVATIVE_PUBLICATION_DELAY_HOURS = 6
HORIZON_HOURS = 168
FORECAST_DAYS = 8
REQUEST_PAUSE_SECONDS = 0.15
CHECKPOINT_EVERY_RUNS = 20

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

# Cache by run initialization. With daily origins this is usually about one
# archive request per origin; four locations are bundled in each request.
selected = {origin: eligible_run(origin) for origin in origins}
run_inits = sorted({item[0] for item in selected.values()})
print(
    f"Preparing {len(origins):,} H168 origins using {len(run_inits):,} unique archived ECMWF runs "
    f"from {origins[0]} onward. Each request bundles all four stations; checkpoints occur every "
    f"{CHECKPOINT_EVERY_RUNS} runs and can be resumed safely."
)
existing_keys = set()
if spark.catalog.tableExists(TARGET_TABLE):
    existing_keys = {
        (pd.to_datetime(row["forecast_origin_utc"], utc=True).to_pydatetime(),
         pd.to_datetime(row["weather_run_init_utc"], utc=True).to_pydatetime())
        for row in spark.table(TARGET_TABLE)
        .filter(F.col("forecast_origin_utc") >= F.lit(ARCHIVE_START_UTC))
        .select("forecast_origin_utc", "weather_run_init_utc").distinct().collect()
    }
pending_rows = []
processed_since_checkpoint = 0
incomplete_origins = []


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


for request_index, run_init in enumerate(run_inits, start=1):
    run_origins = [
        origin for origin in origins
        if selected[origin][0] == run_init and (origin, run_init) not in existing_keys
    ]
    if not run_origins:
        continue
    payloads = request_run(run_init)
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
    temperature_by_station_time = {
        station["station_id"]: dict(zip(times_by_station[index], temps_by_station[index]))
        for index, station in enumerate(stations)
    }
    for origin in run_origins:
        _, available_at, issue_proxy = selected[origin]
        origin_rows = []
        missing_weather = None
        for lead in range(1, HORIZON_HOURS + 1):
            target = origin + dt.timedelta(hours=lead)
            station_values = {
                station["station_id"]: temperature_by_station_time[station["station_id"]].get(target)
                for station in stations
            }
            if any(value is None for value in station_values.values()):
                missing_weather = {
                    "target": target,
                    "stations": [key for key, value in station_values.items() if value is None],
                }
                break
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
        if missing_weather is not None:
            incomplete_origins.append({
                "origin": origin,
                "target": missing_weather["target"],
                "stations": missing_weather["stations"],
                "run_init": run_init,
            })
            continue
        if len(origin_rows) != HORIZON_HOURS:
            raise RuntimeError(f"Origin {origin} produced {len(origin_rows)} rows; expected {HORIZON_HOURS}.")
        pending_rows.extend(origin_rows)
    if request_index % 50 == 0 or request_index == len(run_inits):
        print(
            f"Processed {request_index:,}/{len(run_inits):,} unique runs; "
            f"{len(pending_rows):,} rows are waiting for the next checkpoint."
        )
    processed_since_checkpoint += 1
    if processed_since_checkpoint >= CHECKPOINT_EVERY_RUNS:
        checkpoint(pending_rows)
        existing_keys.update((row["forecast_origin_utc"], row["weather_run_init_utc"]) for row in pending_rows)
        print(f"Saved {len(pending_rows):,} rows; completed vintages can be resumed safely.")
        pending_rows = []
        processed_since_checkpoint = 0
    time.sleep(REQUEST_PAUSE_SECONDS)

checkpoint(pending_rows)
if pending_rows:
    existing_keys.update((row["forecast_origin_utc"], row["weather_run_init_utc"]) for row in pending_rows)
    print(f"Saved final batch of {len(pending_rows):,} rows.")
if not spark.catalog.tableExists(TARGET_TABLE):
    raise RuntimeError("No weather backtest vintages were materialized.")
if incomplete_origins:
    examples = "; ".join(
        f"origin {item['origin']} target {item['target']} missing {','.join(item['stations'])}"
        for item in incomplete_origins[:5]
    )
    print(
        f"Excluded {len(incomplete_origins)} origins with any missing station-hour "
        f"(whole origins excluded from both models). Examples: {examples}"
    )

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
examples_pdf = (
    spark.table(EXAMPLES_TABLE)
    .filter(F.col("respondent") == "ERCO")
    .filter(F.col("horizon_hours").isin(24, 168))
    .filter(F.col("forecast_origin_utc") >= F.lit(ARCHIVE_START_UTC))
    .toPandas()
)
weather_pdf = (
    spark.table(TARGET_TABLE)
    .filter(F.col("forecast_origin_utc") >= F.lit(ARCHIVE_START_UTC))
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

join_columns = [
    "respondent", "forecast_origin_utc", "target_timestamp_utc",
    "weighted_temperature_f", "cdh_65_forecast", "hdh_65_forecast",
    "weather_run_init_utc", "weather_available_at_utc", "weather_issue_time_proxy_utc",
]
matched_pdf = examples_pdf.merge(
    weather_pdf[join_columns],
    on=["respondent", "forecast_origin_utc", "target_timestamp_utc"],
    how="inner",
    validate="many_to_one",
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
