from datetime import date, datetime, timedelta, timezone

from astral import Observer
from astral.sun import sunrise, sunset

_BUFFER = timedelta(minutes=30)


def is_daylight(dt: datetime, latitude: float, longitude: float) -> bool:
    """Check whether a datetime falls between sunrise and sunset.

    Args:
        dt: Timezone-aware datetime to check.
        latitude: Camera latitude.
        longitude: Camera longitude.

    Returns:
        True if the time is between sunrise and sunset.
    """
    observer = Observer(latitude=latitude, longitude=longitude)
    rise = sunrise(observer, date=dt.date(), tzinfo=dt.tzinfo) - _BUFFER
    sset = sunset(observer, date=dt.date(), tzinfo=dt.tzinfo) + _BUFFER
    return rise <= dt <= sset


def get_daylight_span(dt: datetime, latitude: float, longitude: float) -> tuple[datetime, datetime]:
    """Return (start, end) of the most recent daylight period.

    - After today's sunset: today's sunrise to today's sunset.
    - Between sunrise and sunset: today's sunrise to ``dt``.
    - Before today's sunrise: yesterday's sunrise to yesterday's sunset.
    """
    observer = Observer(latitude=latitude, longitude=longitude)
    today_rise = sunrise(observer, date=dt.date(), tzinfo=dt.tzinfo) - _BUFFER
    today_set = sunset(observer, date=dt.date(), tzinfo=dt.tzinfo) + _BUFFER

    if dt >= today_set:
        return today_rise, today_set
    elif dt >= today_rise:
        return today_rise, dt
    else:
        yest = dt.date() - timedelta(days=1)
        yest_rise = sunrise(observer, date=yest, tzinfo=dt.tzinfo) - _BUFFER
        yest_set = sunset(observer, date=yest, tzinfo=dt.tzinfo) + _BUFFER
        return yest_rise, yest_set


def get_daylight_span_for_date(
    target_date: date, latitude: float, longitude: float
) -> tuple[datetime, datetime]:
    """Return (sunrise, sunset) for a specific date."""
    observer = Observer(latitude=latitude, longitude=longitude)
    rise = sunrise(observer, date=target_date, tzinfo=timezone.utc) - _BUFFER
    sset = sunset(observer, date=target_date, tzinfo=timezone.utc) + _BUFFER
    return rise, sset
