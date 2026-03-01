from datetime import datetime

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
