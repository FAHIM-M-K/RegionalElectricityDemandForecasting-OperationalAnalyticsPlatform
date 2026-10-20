"""Point-in-time weather selection for rolling forecast model experiments.

This module is deliberately separate from production scoring. It selects only
weather forecast values available by each simulated issue time and identifies
complete forecast paths before any model comparison is attempted.
"""

from __future__ import annotations

import pandas as pd


WEATHER_FEATURES = [
    "weighted_temperature_f",
    "cdh_65_forecast",
    "hdh_65_forecast",
]
EXAMPLE_KEY = [
    "respondent", "forecast_origin_utc", "target_timestamp_utc", "horizon_hours",
]


def attach_weather_asof_issue(
    examples: pd.DataFrame,
    weather_vintages: pd.DataFrame,
    *,
    issue_delay_minutes: int = 65,
    weather_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach the latest eligible weather vintage to every issue-target row.

    ``weather_available_at_utc`` is the eligibility timestamp. A row is usable
    only when its vintage was available by ``forecast_origin_utc + delay``.
    The latest eligible vintage wins for each target. The function returns all
    demand examples (including unmatched rows with null weather) and an issue
    coverage table; callers must filter to complete issue/horizon curves before
    comparing models.
    """
    features = list(weather_features or WEATHER_FEATURES)
    required_examples = set(EXAMPLE_KEY + ["lead_hours"])
    required_weather = {
        "respondent", "target_timestamp_utc", "weather_available_at_utc",
        "weather_run_init_utc", *features,
    }
    missing_examples = required_examples - set(examples.columns)
    missing_weather = required_weather - set(weather_vintages.columns)
    if missing_examples:
        raise ValueError(f"Forecast examples are missing columns: {sorted(missing_examples)}")
    if missing_weather:
        raise ValueError(f"Weather vintages are missing columns: {sorted(missing_weather)}")
    if issue_delay_minutes < 0:
        raise ValueError("Issue delay must be nonnegative.")

    demand = examples.copy().reset_index(drop=True)
    weather = weather_vintages.copy()
    for frame, columns in (
        (demand, ["forecast_origin_utc", "target_timestamp_utc"]),
        (weather, ["target_timestamp_utc", "weather_available_at_utc", "weather_run_init_utc"]),
    ):
        for column in columns:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
            if frame[column].isna().any():
                raise ValueError(f"Null or invalid UTC timestamp in {column}.")

    if demand.duplicated(EXAMPLE_KEY).any():
        raise ValueError("Forecast examples contain duplicate issue-target keys.")
    source_key = ["respondent", "target_timestamp_utc", "weather_available_at_utc", "weather_run_init_utc"]
    if weather.duplicated(source_key).any():
        raise ValueError("Weather vintages contain duplicate target and vintage keys.")

    demand["_issue_cutoff_utc"] = (
        demand["forecast_origin_utc"] + pd.to_timedelta(issue_delay_minutes, unit="m")
    )
    weather_columns = [
        "respondent", "target_timestamp_utc", "weather_available_at_utc",
        "weather_run_init_utc", *features,
    ]
    left = demand.sort_values(
        ["_issue_cutoff_utc", "respondent", "target_timestamp_utc"], kind="stable"
    )
    right = weather[weather_columns].sort_values(
        ["weather_available_at_utc", "respondent", "target_timestamp_utc", "weather_run_init_utc"],
        kind="stable",
    )
    result = pd.merge_asof(
        left,
        right,
        left_on="_issue_cutoff_utc",
        right_on="weather_available_at_utc",
        by=["respondent", "target_timestamp_utc"],
        direction="backward",
        allow_exact_matches=True,
    )
    result = result.rename(columns={"_issue_cutoff_utc": "weather_issue_cutoff_utc"})
    result["weather_available_by_issue"] = (
        result["weather_available_at_utc"].notna()
        & (result["weather_available_at_utc"] <= result["weather_issue_cutoff_utc"])
    )

    group_keys = ["respondent", "forecast_origin_utc", "horizon_hours"]
    coverage = result.groupby(group_keys, dropna=False).agg(
        expected_horizon_rows=("lead_hours", "size"),
        distinct_leads=("lead_hours", "nunique"),
        weather_rows_available=("weather_available_by_issue", "sum"),
    ).reset_index()
    expected_counts = coverage["horizon_hours"].astype(int)
    complete = (
        (coverage["expected_horizon_rows"] == expected_counts)
        & (coverage["distinct_leads"] == expected_counts)
        & (coverage["weather_rows_available"] == expected_counts)
    )
    for feature in features:
        feature_counts = result.groupby(group_keys, dropna=False)[feature].count().to_numpy()
        complete &= feature_counts == expected_counts.to_numpy()
    coverage["complete_weather_horizon"] = complete
    result = result.drop(columns=["weather_available_by_issue"])
    return result, coverage
