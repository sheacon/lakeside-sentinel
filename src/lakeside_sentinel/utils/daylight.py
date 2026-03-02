from datetime import date, datetime, timedelta, timezone

from astral import Observer
from astral.sun import sun


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
    s = sun(observer, date=dt.date(), tzinfo=dt.tzinfo)
    return s["sunrise"] <= dt <= s["sunset"]


def get_daylight_span(dt: datetime, latitude: float, longitude: float) -> tuple[datetime, datetime]:
    """Return (start, end) of the most recent daylight period.

    - After today's sunset: today's sunrise to today's sunset.
    - Between sunrise and sunset: today's sunrise to ``dt``.
    - Before today's sunrise: yesterday's sunrise to yesterday's sunset.
    """
    observer = Observer(latitude=latitude, longitude=longitude)
    today = sun(observer, date=dt.date(), tzinfo=dt.tzinfo)

    if dt >= today["sunset"]:
        return today["sunrise"], today["sunset"]
    elif dt >= today["sunrise"]:
        return today["sunrise"], dt
    else:
        yesterday = sun(observer, date=dt.date() - timedelta(days=1), tzinfo=dt.tzinfo)
        return yesterday["sunrise"], yesterday["sunset"]


def get_daylight_span_for_date(
    target_date: date, latitude: float, longitude: float
) -> tuple[datetime, datetime]:
    """Return (sunrise, sunset) for a specific date."""
    observer = Observer(latitude=latitude, longitude=longitude)
    s = sun(observer, date=target_date, tzinfo=timezone.utc)
    return s["sunrise"], s["sunset"]
