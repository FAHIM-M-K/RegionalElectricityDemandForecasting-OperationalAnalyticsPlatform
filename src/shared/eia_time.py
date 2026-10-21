"""Small timezone helpers for EIA-930 UTC-frequency queries."""

import datetime as dt
from zoneinfo import ZoneInfo


def operating_day_end_utc(now_utc: dt.datetime, reporting_timezone: str) -> dt.datetime:
    """Return next reporting-zone midnight as UTC for a complete local-day query.

    EIA-930's ``hourly`` frequency uses UTC timestamps. Its day-ahead forecast,
    however, covers the balancing authority's reporting day. Converting local
    midnight with ``zoneinfo`` handles 23/25-hour daylight-saving days.
    """
    if now_utc.tzinfo is None or now_utc.utcoffset() is None:
        raise ValueError("now_utc must be timezone-aware")
    local_date = now_utc.astimezone(ZoneInfo(reporting_timezone)).date()
    next_local_midnight = dt.datetime.combine(
        local_date + dt.timedelta(days=1), dt.time.min,
        tzinfo=ZoneInfo(reporting_timezone),
    )
    return next_local_midnight.astimezone(dt.timezone.utc)
