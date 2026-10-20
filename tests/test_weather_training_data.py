import pandas as pd
import pytest

from src.mlops.weather_training_data import attach_weather_asof_issue


def _example_rows(origin: str, leads: list[int], horizon: int = 24) -> pd.DataFrame:
    origin_ts = pd.Timestamp(origin)
    return pd.DataFrame([
        {
            "respondent": "ERCO",
            "forecast_origin_utc": origin_ts,
            "target_timestamp_utc": origin_ts + pd.Timedelta(hours=lead),
            "horizon_hours": horizon,
            "lead_hours": lead,
        }
        for lead in leads
    ])


def _vintage(target: str, available: str, run_init: str, temperature: float) -> dict:
    return {
        "respondent": "ERCO",
        "target_timestamp_utc": pd.Timestamp(target),
        "weather_available_at_utc": pd.Timestamp(available),
        "weather_run_init_utc": pd.Timestamp(run_init),
        "weighted_temperature_f": temperature,
        "cdh_65_forecast": max(temperature - 65, 0),
        "hdh_65_forecast": max(65 - temperature, 0),
    }


def test_selects_latest_vintage_available_at_simulated_issue_time():
    origin = "2026-01-01T00:00:00Z"
    examples = _example_rows(origin, [1, 2])
    weather = pd.DataFrame([
        _vintage("2026-01-01T01:00:00Z", "2025-12-31T23:00:00Z", "2025-12-31T12:00:00Z", 50),
        _vintage("2026-01-01T01:00:00Z", "2026-01-01T02:00:00Z", "2026-01-01T00:00:00Z", 70),
        _vintage("2026-01-01T02:00:00Z", "2025-12-31T23:00:00Z", "2025-12-31T12:00:00Z", 51),
    ])

    selected, coverage = attach_weather_asof_issue(examples, weather)

    assert selected["weighted_temperature_f"].tolist() == [50, 51]
    assert selected["weather_available_at_utc"].iloc[0] <= selected["weather_issue_cutoff_utc"].iloc[0]
    assert coverage["complete_weather_horizon"].tolist() == [False]
    assert coverage["weather_rows_available"].tolist() == [2]


def test_complete_horizon_requires_every_lead_and_feature():
    origin = "2026-01-01T00:00:00Z"
    examples = _example_rows(origin, [1, 2, 3], horizon=3)
    weather = pd.DataFrame([
        _vintage(f"2026-01-01T0{lead}:00:00Z", "2025-12-31T23:00:00Z", "2025-12-31T12:00:00Z", 60 + lead)
        for lead in (1, 2, 3)
    ])

    _, coverage = attach_weather_asof_issue(examples, weather)

    assert coverage.loc[0, "complete_weather_horizon"]


def test_daily_168_hour_vintage_cannot_supply_full_hourly_h168_issue():
    hourly_origin = pd.Timestamp("2026-01-01T01:00:00Z")
    examples = _example_rows(hourly_origin.isoformat(), list(range(1, 169)), horizon=168)
    # The archived daily issue at 00:00 contains targets through Jan 8 00:00.
    # An hourly issue at 01:00 needs one more target hour than that vintage has.
    weather = pd.DataFrame([
        _vintage(
            (pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(hours=lead)).isoformat(),
            "2026-01-01T00:30:00Z",
            "2025-12-31T18:00:00Z",
            60 + lead / 100,
        )
        for lead in range(1, 169)
    ])

    _, coverage = attach_weather_asof_issue(examples, weather)

    assert coverage.loc[0, "weather_rows_available"] == 167
    assert not coverage.loc[0, "complete_weather_horizon"]


def test_rejects_duplicate_example_keys():
    examples = _example_rows("2026-01-01T00:00:00Z", [1, 1])
    weather = pd.DataFrame([
        _vintage("2026-01-01T01:00:00Z", "2025-12-31T23:00:00Z", "2025-12-31T12:00:00Z", 60)
    ])

    with pytest.raises(ValueError, match="duplicate issue-target keys"):
        attach_weather_asof_issue(examples, weather)


def test_rejects_weather_duplicate_vintage_rows():
    examples = _example_rows("2026-01-01T00:00:00Z", [1])
    row = _vintage("2026-01-01T01:00:00Z", "2025-12-31T23:00:00Z", "2025-12-31T12:00:00Z", 60)
    weather = pd.DataFrame([row, row])

    with pytest.raises(ValueError, match="duplicate target and vintage keys"):
        attach_weather_asof_issue(examples, weather)
