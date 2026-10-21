import datetime as dt

import pytest

from src.shared.eia_time import operating_day_end_utc


@pytest.mark.parametrize(
    ("now_utc", "expected_end"),
    [
        # Central Daylight Time: local midnight is 05:00 UTC.
        ("2026-10-21T15:00:00+00:00", "2026-10-22T05:00:00+00:00"),
        # Central Standard Time: local midnight is 06:00 UTC.
        ("2026-11-02T15:00:00+00:00", "2026-11-03T06:00:00+00:00"),
        # The fall-back operating day is 25 hours; next midnight is CST.
        ("2026-11-01T15:00:00+00:00", "2026-11-02T06:00:00+00:00"),
        # The spring-forward operating day is 23 hours; next midnight is CDT.
        ("2026-03-08T15:00:00+00:00", "2026-03-09T05:00:00+00:00"),
    ],
)
def test_operating_day_end_utc_includes_full_central_day(now_utc, expected_end):
    actual = operating_day_end_utc(
        dt.datetime.fromisoformat(now_utc), "America/Chicago"
    )
    assert actual == dt.datetime.fromisoformat(expected_end)


def test_operating_day_end_requires_timezone_aware_now():
    with pytest.raises(ValueError, match="timezone-aware"):
        operating_day_end_utc(dt.datetime(2026, 10, 21, 15), "America/Chicago")
