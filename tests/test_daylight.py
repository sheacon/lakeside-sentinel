from datetime import date, datetime, timedelta, timezone

from lakeside_motorbikes.utils.daylight import (
    get_daylight_span,
    get_daylight_span_for_date,
    is_daylight,
)

# London coordinates for testing
LAT = 51.5074
LON = -0.1278


class TestIsDaylight:
    def test_midday_is_daylight(self) -> None:
        dt = datetime(2026, 6, 21, 12, 0, 0, tzinfo=timezone.utc)
        assert is_daylight(dt, LAT, LON) is True

    def test_midnight_is_not_daylight(self) -> None:
        dt = datetime(2026, 6, 21, 0, 0, 0, tzinfo=timezone.utc)
        assert is_daylight(dt, LAT, LON) is False

    def test_early_morning_before_sunrise(self) -> None:
        # 3 AM UTC in June — before sunrise in London
        dt = datetime(2026, 6, 21, 3, 0, 0, tzinfo=timezone.utc)
        assert is_daylight(dt, LAT, LON) is False

    def test_late_evening_after_sunset(self) -> None:
        # 11 PM UTC in June — after sunset in London
        dt = datetime(2026, 6, 21, 23, 0, 0, tzinfo=timezone.utc)
        assert is_daylight(dt, LAT, LON) is False

    def test_winter_midday_is_daylight(self) -> None:
        # Midday in December — should still be daylight
        dt = datetime(2026, 12, 21, 12, 0, 0, tzinfo=timezone.utc)
        assert is_daylight(dt, LAT, LON) is True

    def test_winter_early_evening_is_dark(self) -> None:
        # 5 PM UTC in December — after sunset in London (sunset ~15:53)
        dt = datetime(2026, 12, 21, 17, 0, 0, tzinfo=timezone.utc)
        assert is_daylight(dt, LAT, LON) is False


class TestGetDaylightSpan:
    def test_after_sunset_returns_todays_span(self) -> None:
        # 11 PM UTC on June 21 — well after sunset in London
        dt = datetime(2026, 6, 21, 23, 0, 0, tzinfo=timezone.utc)
        start, end = get_daylight_span(dt, LAT, LON)
        assert start.date() == dt.date()
        assert end.date() == dt.date()
        assert start < end
        assert end <= dt

    def test_during_day_returns_sunrise_to_now(self) -> None:
        # Midday on June 21 — between sunrise and sunset
        dt = datetime(2026, 6, 21, 12, 0, 0, tzinfo=timezone.utc)
        start, end = get_daylight_span(dt, LAT, LON)
        assert start.date() == dt.date()
        assert end == dt
        assert start < end

    def test_before_sunrise_returns_yesterdays_span(self) -> None:
        # 3 AM UTC on June 21 — before sunrise in London
        dt = datetime(2026, 6, 21, 3, 0, 0, tzinfo=timezone.utc)
        start, end = get_daylight_span(dt, LAT, LON)
        yesterday = dt.date() - timedelta(days=1)
        assert start.date() == yesterday
        assert end.date() == yesterday
        assert start < end

    def test_span_covers_reasonable_hours(self) -> None:
        # After sunset in winter — shorter daylight span
        dt = datetime(2026, 12, 21, 17, 0, 0, tzinfo=timezone.utc)
        start, end = get_daylight_span(dt, LAT, LON)
        duration = (end - start).total_seconds() / 3600
        # London winter: ~8 hours of daylight
        assert 6 < duration < 10


class TestGetDaylightSpanForDate:
    def test_returns_sunrise_sunset_for_date(self) -> None:
        target = date(2026, 6, 21)
        start, end = get_daylight_span_for_date(target, LAT, LON)
        assert start.date() == target
        assert end.date() == target
        assert start < end

    def test_summer_longer_than_winter(self) -> None:
        summer = date(2026, 6, 21)
        winter = date(2026, 12, 21)
        s_start, s_end = get_daylight_span_for_date(summer, LAT, LON)
        w_start, w_end = get_daylight_span_for_date(winter, LAT, LON)
        summer_hours = (s_end - s_start).total_seconds() / 3600
        winter_hours = (w_end - w_start).total_seconds() / 3600
        assert summer_hours > winter_hours

    def test_returns_utc_aware_datetimes(self) -> None:
        target = date(2026, 3, 15)
        start, end = get_daylight_span_for_date(target, LAT, LON)
        assert start.tzinfo is not None
        assert end.tzinfo is not None
