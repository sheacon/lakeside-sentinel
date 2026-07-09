from datetime import date, datetime, time, timedelta

DAY_START_HOUR = 5
DAY_END_HOUR = 21


def _local_span_for(local_date: date) -> tuple[datetime, datetime]:
    """Return (start, end) bounds for a local-calendar date as aware datetimes
    in the system's local timezone.
    """
    tz = datetime.now().astimezone().tzinfo
    start = datetime.combine(local_date, time(DAY_START_HOUR, 0), tzinfo=tz)
    end = datetime.combine(local_date, time(DAY_END_HOUR, 0), tzinfo=tz)
    return start, end


def is_daylight(dt: datetime) -> bool:
    """Whether `dt` falls within the hardcoded daylight window (5am–9pm local)."""
    local = dt.astimezone()
    start, end = _local_span_for(local.date())
    return start <= dt <= end


def get_daylight_span(dt: datetime) -> tuple[datetime, datetime]:
    """Return (start, end) of the most recent daylight period relative to `dt`.

    - After today's window: today's full window.
    - Inside today's window: today's start to `dt`.
    - Before today's window: yesterday's full window.
    """
    local = dt.astimezone()
    today_start, today_end = _local_span_for(local.date())

    if dt >= today_end:
        return today_start, today_end
    if dt >= today_start:
        return today_start, dt
    yest = local.date() - timedelta(days=1)
    return _local_span_for(yest)


def get_daylight_span_for_date(target_date: date) -> tuple[datetime, datetime]:
    """Return the daylight window (start, end) for a specific local date."""
    return _local_span_for(target_date)
