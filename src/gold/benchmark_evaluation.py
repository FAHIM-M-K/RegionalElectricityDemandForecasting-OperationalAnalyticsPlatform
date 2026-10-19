"""EIA H24 Day-Ahead Benchmark Evaluation and As-Of Vintage Matching.

Implements fair head-to-head comparison between rolling H24 model issues and
EIA-930 day-ahead reference forecasts:
1. Strict as-of vintage selection (first_observed_at_utc <= model_issued_at_utc).
2. Prevents look-ahead: future benchmark revisions or later vintages are rejected.
3. Pre-registered one-forecast-per-target headline policy to prevent overlapping
   hourly issues from biasing evaluation with repeated target weighting.
4. Lead slices (including lead 24), signed bias (forecast - actual), MAE, RMSE,
   and peak window metrics.
"""

from typing import Any, Dict, List, Optional, Tuple
import json
import numpy as np
import pandas as pd

from src.shared.forecast_contract import (
    HORIZON_H24,
)


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
    """Return the newest benchmark vintage for each target available as of as_of_time_utc.

    Enforces no look-ahead: vintages with first_observed_at_utc > as_of_time_utc are excluded.
    """
    if df_vintages.empty:
        return pd.DataFrame(columns=df_vintages.columns)

    frame = df_vintages.copy()
    frame["target_timestamp_utc"] = pd.to_datetime(frame["target_timestamp_utc"], utc=True)
    frame["first_observed_at_utc"] = pd.to_datetime(frame["first_observed_at_utc"], utc=True)
    as_of_ts = pd.to_datetime(as_of_time_utc, utc=True)

    # Filter to eligible vintages available by the as-of cutoff
    eligible = frame[
        (frame["benchmark_provider"] == provider)
        & (frame["respondent"] == respondent)
        & (frame["first_observed_at_utc"] <= as_of_ts)
        & frame["forecast_mw"].notna()
    ].copy()

    if eligible.empty:
        return pd.DataFrame(columns=frame.columns)

    # If provider published timestamp exists, also assert it is <= as_of_ts
    if "published_at_utc" in eligible.columns:
        valid_published = eligible["published_at_utc"].isna() | (
            pd.to_datetime(eligible["published_at_utc"], utc=True) <= as_of_ts
        )
        eligible = eligible[valid_published]

    # Select the latest observed vintage per target timestamp
    # Row selection must preserve a single real vintage. DataFrameGroupBy.last()
    # works column-by-column and can accidentally combine values from different
    # rows when nullable fields differ across vintages.
    eligible = eligible.sort_values(
        ["target_timestamp_utc", "first_observed_at_utc", "vintage_id"]
    )
    latest_vintages = eligible.drop_duplicates("target_timestamp_utc", keep="last")
    return latest_vintages


def match_h24_and_eia(
    df_h24_forecasts: pd.DataFrame,
    df_eia_vintages: pd.DataFrame,
    df_actuals: pd.DataFrame,
    headline_policy: str = "lead_24",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Match rolling H24 forecasts to EIA vintages on identical target hours and actual demand.

    Parameters:
      headline_policy: 'lead_24' selects exact lead-24 slice for headline metrics;
                       'all_leads' evaluates every rolling lead with lead-bucket breakdowns.
    """
    h24 = df_h24_forecasts[df_h24_forecasts["horizon_hours"] == HORIZON_H24].copy()
    h24["target_timestamp_utc"] = pd.to_datetime(h24["target_timestamp_utc"], utc=True)
    h24["issued_at_utc"] = pd.to_datetime(h24["issued_at_utc"], utc=True)
    issue_keys = ["respondent", "horizon_hours", "forecast_origin_utc", "issued_at_utc", "target_timestamp_utc"]
    if "forecast_issue_id" in h24.columns:
        issue_keys.insert(1, "forecast_issue_id")
    _require_unique(h24, issue_keys, "H24 forecasts")

    actuals = df_actuals[["respondent", "timestamp_utc", "actual_demand_mw"]].copy()
    actuals["timestamp_utc"] = pd.to_datetime(actuals["timestamp_utc"], utc=True)
    actuals = actuals[actuals["actual_demand_mw"].notna()]
    _require_unique(actuals, ["respondent", "timestamp_utc"], "Actual demand")

    matched_records = []
    # Evaluate by unique issue origin or across issues
    issue_group_keys = ["forecast_origin_utc", "issued_at_utc"]
    if "forecast_issue_id" in h24.columns:
        issue_group_keys.append("forecast_issue_id")
    for _, issue_group in h24.groupby(issue_group_keys, dropna=False):
        issued_at = issue_group["issued_at_utc"].iloc[0]
        as_of_benchmarks = select_as_of_benchmark_vintages(df_eia_vintages, issued_at)
        if as_of_benchmarks.empty:
            continue

        joined = issue_group.merge(
            as_of_benchmarks[["target_timestamp_utc", "forecast_mw"]].rename(
                columns={"forecast_mw": "eia_forecast_mw"}
            ),
            on="target_timestamp_utc",
            how="inner",
        ).merge(
            actuals.rename(columns={"timestamp_utc": "target_timestamp_utc"}),
            on=["respondent", "target_timestamp_utc"],
            how="inner",
        )
        matched_records.append(joined)

    if not matched_records:
        return pd.DataFrame(), {"matched_rows": 0, "unique_targets": 0}

    df_matched = pd.concat(matched_records, ignore_index=True)
    df_matched["model_error_mw"] = df_matched["predicted_demand_mw"] - df_matched["actual_demand_mw"]
    df_matched["model_abs_error_mw"] = df_matched["model_error_mw"].abs()
    df_matched["eia_error_mw"] = df_matched["eia_forecast_mw"] - df_matched["actual_demand_mw"]
    df_matched["eia_abs_error_mw"] = df_matched["eia_error_mw"].abs()

    # Apply headline policy to avoid repeated target sample inflation
    if headline_policy == "lead_24":
        headline_df = df_matched[df_matched["lead_hours"] == 24]
    elif headline_policy == "all_leads":
        # Default: latest issue per target
        sort_columns = ["target_timestamp_utc", "issued_at_utc", "forecast_origin_utc"]
        if "forecast_issue_id" in df_matched.columns:
            sort_columns.append("forecast_issue_id")
        headline_df = df_matched.sort_values(sort_columns).drop_duplicates(
            "target_timestamp_utc", keep="last"
        )
    else:
        raise ValueError("headline_policy must be 'lead_24' or 'all_leads'.")

    summary_metrics = {
        "matched_rows": int(len(headline_df)),
        "unique_targets": int(headline_df["target_timestamp_utc"].nunique()),
        "model_mae_mw": round(float(headline_df["model_abs_error_mw"].mean()), 2) if not headline_df.empty else 0.0,
        "eia_mae_mw": round(float(headline_df["eia_abs_error_mw"].mean()), 2) if not headline_df.empty else 0.0,
        "model_rmse_mw": round(float(np.sqrt((headline_df["model_error_mw"] ** 2).mean())), 2) if not headline_df.empty else 0.0,
        "eia_rmse_mw": round(float(np.sqrt((headline_df["eia_error_mw"] ** 2).mean())), 2) if not headline_df.empty else 0.0,
        "model_bias_mw": round(float(headline_df["model_error_mw"].mean()), 2) if not headline_df.empty else 0.0,
        "eia_bias_mw": round(float(headline_df["eia_error_mw"].mean()), 2) if not headline_df.empty else 0.0,
    }

    return df_matched, summary_metrics


def match_h168_and_ercot(
    df_h168_forecasts: pd.DataFrame,
    df_ercot_vintages: pd.DataFrame,
    df_actuals: pd.DataFrame,
    lead_buckets: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Match rolling H168 forecasts to official ERCOT 7-day benchmark vintages.

    Reports overall metrics, lead-bucket metrics (1-24, 25-48, 49-168), and lead-168 diagnostic.
    If no vintages exist, returns truthful PROSPECTIVE_PENDING_MATURED_DATA status without fabricating scores.
    """
    if lead_buckets is None:
        lead_buckets = [(1, 24), (25, 48), (49, 168)]

    if df_ercot_vintages.empty or df_h168_forecasts.empty:
        return pd.DataFrame(), {
            "status": "PROSPECTIVE_PENDING_MATURED_DATA",
            "matched_rows": 0,
            "unique_targets": 0,
            "lead_bucket_metrics": {},
            "lead_168_diagnostic": None,
            "notice": (
                "ERCOT seven-day historical issue vintages are unavailable retrospectively. "
                "Matched evaluation requires prospective snapshot collection."
            ),
        }

    h168 = df_h168_forecasts[df_h168_forecasts["horizon_hours"] == 168].copy()
    h168["target_timestamp_utc"] = pd.to_datetime(h168["target_timestamp_utc"], utc=True)
    h168["issued_at_utc"] = pd.to_datetime(h168["issued_at_utc"], utc=True)
    issue_keys = ["respondent", "horizon_hours", "forecast_origin_utc", "issued_at_utc", "target_timestamp_utc"]
    if "forecast_issue_id" in h168.columns:
        issue_keys.insert(1, "forecast_issue_id")
    _require_unique(h168, issue_keys, "H168 forecasts")

    actuals = df_actuals[["respondent", "timestamp_utc", "actual_demand_mw"]].copy()
    actuals["timestamp_utc"] = pd.to_datetime(actuals["timestamp_utc"], utc=True)
    actuals = actuals[actuals["actual_demand_mw"].notna()]
    _require_unique(actuals, ["respondent", "timestamp_utc"], "Actual demand")

    matched_records = []
    issue_group_keys = ["forecast_origin_utc", "issued_at_utc"]
    if "forecast_issue_id" in h168.columns:
        issue_group_keys.append("forecast_issue_id")
    for _, issue_group in h168.groupby(issue_group_keys, dropna=False):
        issued_at = issue_group["issued_at_utc"].iloc[0]
        as_of_benchmarks = select_as_of_benchmark_vintages(
            df_ercot_vintages, issued_at, provider="ERCOT", respondent="ERCO"
        )
        if as_of_benchmarks.empty:
            continue

        joined = issue_group.merge(
            as_of_benchmarks[["target_timestamp_utc", "forecast_mw"]].rename(
                columns={"forecast_mw": "ercot_forecast_mw"}
            ),
            on="target_timestamp_utc",
            how="inner",
        ).merge(
            actuals.rename(columns={"timestamp_utc": "target_timestamp_utc"}),
            on=["respondent", "target_timestamp_utc"],
            how="inner",
        )
        matched_records.append(joined)

    if not matched_records:
        return pd.DataFrame(), {
            "status": "PROSPECTIVE_PENDING_MATURED_DATA",
            "matched_rows": 0,
            "unique_targets": 0,
            "lead_bucket_metrics": {},
            "lead_168_diagnostic": None,
            "notice": "No mature overlapping actuals available yet for captured ERCOT vintages.",
        }

    df_matched = pd.concat(matched_records, ignore_index=True)
    df_matched["model_error_mw"] = df_matched["predicted_demand_mw"] - df_matched["actual_demand_mw"]
    df_matched["model_abs_error_mw"] = df_matched["model_error_mw"].abs()
    df_matched["ercot_error_mw"] = df_matched["ercot_forecast_mw"] - df_matched["actual_demand_mw"]
    df_matched["ercot_abs_error_mw"] = df_matched["ercot_error_mw"].abs()

    # H168 is a rolling issue product. These lead-bucket metrics describe each
    # distinct issued forecast decision; they are not a unique-target headline.
    # State both counts so repeated targets across successive issues are visible.
    bucket_results = {}
    for (start_lead, end_lead) in lead_buckets:
        b_slice = df_matched[(df_matched["lead_hours"] >= start_lead) & (df_matched["lead_hours"] <= end_lead)]
        if not b_slice.empty:
            bucket_results[f"leads_{start_lead}_{end_lead}"] = {
                "matched_rows": len(b_slice),
                "unique_targets": int(b_slice["target_timestamp_utc"].nunique()),
                "model_mae_mw": round(float(b_slice["model_abs_error_mw"].mean()), 2),
                "ercot_mae_mw": round(float(b_slice["ercot_abs_error_mw"].mean()), 2),
                "model_rmse_mw": round(float(np.sqrt((b_slice["model_error_mw"] ** 2).mean())), 2),
                "ercot_rmse_mw": round(float(np.sqrt((b_slice["ercot_error_mw"] ** 2).mean())), 2),
            }

    # Lead 168 diagnostic
    lead_168_slice = df_matched[df_matched["lead_hours"] == 168]
    lead_168_diag = None
    if not lead_168_slice.empty:
        lead_168_diag = {
            "matched_rows": len(lead_168_slice),
            "model_mae_mw": round(float(lead_168_slice["model_abs_error_mw"].mean()), 2),
            "ercot_mae_mw": round(float(lead_168_slice["ercot_abs_error_mw"].mean()), 2),
        }

    summary_metrics = {
        "status": "EVALUATED",
        "matched_rows": len(df_matched),
        "unique_targets": int(df_matched["target_timestamp_utc"].nunique()),
        "evaluation_unit": "forecast_issue_target_pair; repeated targets across rolling issues are retained",
        "model_mae_mw": round(float(df_matched["model_abs_error_mw"].mean()), 2),
        "ercot_mae_mw": round(float(df_matched["ercot_abs_error_mw"].mean()), 2),
        "model_rmse_mw": round(float(np.sqrt((df_matched["model_error_mw"] ** 2).mean())), 2),
        "ercot_rmse_mw": round(float(np.sqrt((df_matched["ercot_error_mw"] ** 2).mean())), 2),
        "model_bias_mw": round(float(df_matched["model_error_mw"].mean()), 2),
        "ercot_bias_mw": round(float(df_matched["ercot_error_mw"].mean()), 2),
        "lead_bucket_metrics": bucket_results,
        "lead_168_diagnostic": lead_168_diag,
    }

    return df_matched, summary_metrics


def run_benchmark_evaluation(spark_session) -> dict:
    """Evaluate the latest production run against available as-of EIA/ERCOT vintages.

    This is an operational materialization entry point. It evaluates the latest
    served model run per horizon, keeps hourly H168 issue-target pairs for lead
    buckets, and labels the ERCOT comparison pending until prospective vintages
    have matured against actual demand.
    """
    from datetime import datetime, timezone
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        DoubleType, IntegerType, LongType, StringType, StructField, StructType,
        TimestampType,
    )

    catalog = "main"
    forecasts_table = f"{catalog}.gold.forecast_evaluation"
    serving_table = f"{catalog}.gold.forecast_latest"
    actuals_table = f"{catalog}.gold.fact_hourly_electricity_demand"
    eia_table = f"{catalog}.bronze.eia_forecast_vintages"
    ercot_table = f"{catalog}.bronze.ercot_benchmark_raw"
    summary_table = f"{catalog}.gold.forecast_benchmark_metrics"
    detail_table = f"{catalog}.gold.forecast_benchmark_evaluation"

    for table in (forecasts_table, serving_table, actuals_table):
        if not spark_session.catalog.tableExists(table):
            raise RuntimeError(f"Benchmark evaluation requires table {table}.")
    latest_run_rows = (
        spark_session.table(serving_table)
        .select("horizon_hours", "model_run_id")
        .distinct()
        .collect()
    )
    latest_runs = {int(row["horizon_hours"]): row["model_run_id"] for row in latest_run_rows}
    if 24 not in latest_runs or 168 not in latest_runs:
        raise RuntimeError("Benchmark evaluation requires the current H24 and H168 served runs.")

    window_start = F.current_timestamp() - F.expr("INTERVAL 180 DAYS")
    all_forecasts = (
        spark_session.table(forecasts_table)
        .filter(
            (F.col("forecast_origin_utc") >= window_start)
            & (F.col("model_run_id").isin(*[str(value) for value in latest_runs.values()]))
        )
        .toPandas()
    )
    actuals = (
        spark_session.table(actuals_table)
        .filter((F.col("actual_demand_mw").isNotNull()) & (F.col("timestamp_utc") >= window_start))
        .select("respondent", "timestamp_utc", "actual_demand_mw")
        .toPandas()
    )
    for frame, columns in ((all_forecasts, ("forecast_origin_utc", "target_timestamp_utc", "issued_at_utc")),
                           (actuals, ("timestamp_utc",))):
        for column in columns:
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], utc=True)

    eia_vintages = (
        spark_session.table(eia_table).filter(F.col("target_timestamp_utc") >= window_start).toPandas()
        if spark_session.catalog.tableExists(eia_table)
        else pd.DataFrame(columns=[
            "benchmark_provider", "respondent", "vintage_id", "target_timestamp_utc",
            "forecast_mw", "published_at_utc", "first_observed_at_utc",
        ])
    )
    ercot_vintages = (
        spark_session.table(ercot_table).filter(F.col("target_timestamp_utc") >= window_start).toPandas()
        if spark_session.catalog.tableExists(ercot_table)
        else pd.DataFrame(columns=[
            "benchmark_provider", "respondent", "vintage_id", "target_timestamp_utc",
            "forecast_mw", "published_at_utc", "first_observed_at_utc",
        ])
    )
    for frame in (eia_vintages, ercot_vintages):
        for column in ("target_timestamp_utc", "published_at_utc", "first_observed_at_utc"):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], utc=True)

    summary_rows = []
    detail_frames = []
    now = datetime.now(timezone.utc)
    for horizon, provider, vintages in ((24, "EIA", eia_vintages), (168, "ERCOT", ercot_vintages)):
        run_id = latest_runs[horizon]
        forecasts = all_forecasts[
            (all_forecasts["horizon_hours"] == horizon)
            & (all_forecasts["model_run_id"] == run_id)
        ].copy()
        if provider == "EIA":
            matched, metrics = match_h24_and_eia(forecasts, vintages, actuals, headline_policy="lead_24")
            benchmark_mae_key = "eia_mae_mw"
            benchmark_rmse_key = "eia_rmse_mw"
            benchmark_bias_key = "eia_bias_mw"
        else:
            matched, metrics = match_h168_and_ercot(forecasts, vintages, actuals)
            benchmark_mae_key = "ercot_mae_mw"
            benchmark_rmse_key = None
            benchmark_bias_key = None

        status = metrics.get("status", "EVALUATED" if metrics.get("matched_rows", 0) else "PROSPECTIVE_PENDING_MATURED_DATA")
        row = {
            "benchmark_provider": provider,
            "horizon_hours": horizon,
            "model_run_id": str(run_id),
            "status": status,
            "matched_rows": int(metrics.get("matched_rows", 0)),
            "unique_targets": int(metrics.get("unique_targets", 0)),
            "model_mae_mw": metrics.get("model_mae_mw"),
            "benchmark_mae_mw": metrics.get(benchmark_mae_key),
            "model_rmse_mw": metrics.get("model_rmse_mw"),
            "benchmark_rmse_mw": metrics.get(benchmark_rmse_key) if benchmark_rmse_key else None,
            "model_bias_mw": metrics.get("model_bias_mw"),
            "benchmark_bias_mw": metrics.get(benchmark_bias_key) if benchmark_bias_key else None,
            "metrics_json": json.dumps(metrics, default=str, sort_keys=True),
            "computed_at_utc": now,
        }
        summary_rows.append(row)
        if not matched.empty:
            matched.insert(0, "benchmark_provider", provider)
            matched.insert(1, "horizon_hours_evaluated", horizon)
            detail_frames.append(matched)

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
    spark_session.createDataFrame(summary_rows, schema=summary_schema).write.format("delta").mode(
        "overwrite"
    ).option("overwriteSchema", "true").saveAsTable(summary_table)

    if detail_frames:
        details = pd.concat(detail_frames, ignore_index=True)
        for column in details.columns:
            if pd.api.types.is_datetime64_any_dtype(details[column]):
                details[column] = pd.to_datetime(details[column], utc=True).dt.tz_localize(None)
        spark_session.createDataFrame(details).write.format("delta").mode("overwrite").option(
            "overwriteSchema", "true"
        ).saveAsTable(detail_table)
    elif spark_session.catalog.tableExists(detail_table):
        # Do not leave old comparison rows visible when today's as-of match is empty.
        from delta.tables import DeltaTable
        DeltaTable.forName(spark_session, detail_table).delete()

    print(
        f"Materialized H24/EIA and H168/ERCOT benchmark status in {summary_table}; "
        f"matched detail rows={sum(len(frame) for frame in detail_frames)}."
    )
    return {"summary_table": summary_table, "detail_table": detail_table, "rows": summary_rows}


if __name__ == "__main__" and "spark" in globals():
    run_benchmark_evaluation(spark)

