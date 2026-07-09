from datetime import date, datetime, timedelta

from lakeside_sentinel.utils.daylight import (
    DAY_END_HOUR,
    DAY_START_HOUR,
    get_daylight_span,
    get_daylight_span_for_date,
    is_daylight,
)


def _local_tz():
    return datetime.now().astimezone().tzinfo


def _local(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=_local_tz())


class TestIsDaylight:
    def test_midday_is_daylight(self) -> None:
        assert is_daylight(_local(2026, 6, 21, 12)) is True

    def test_midnight_is_not_daylight(self) -> None:
        assert is_daylight(_local(2026, 6, 21, 0)) is False

    def test_just_before_start_is_not_daylight(self) -> None:
        assert is_daylight(_local(2026, 6, 21, DAY_START_HOUR - 1, 59)) is False

    def test_at_start_boundary_is_daylight(self) -> None:
        assert is_daylight(_local(2026, 6, 21, DAY_START_HOUR)) is True

    def test_at_end_boundary_is_daylight(self) -> None:
        assert is_daylight(_local(2026, 6, 21, DAY_END_HOUR)) is True

    def test_just_after_end_is_not_daylight(self) -> None:
        assert is_daylight(_local(2026, 6, 21, DAY_END_HOUR, 1)) is False

    def test_winter_midday_is_daylight(self) -> None:
        assert is_daylight(_local(2026, 12, 21, 12)) is True


class TestGetDaylightSpan:
    def test_after_window_returns_full_window(self) -> None:
        dt = _local(2026, 6, 21, DAY_END_HOUR + 1)
        start, end = get_daylight_span(dt)
        assert start.date() == dt.date()
        assert end.date() == dt.date()
        assert start.hour == DAY_START_HOUR
        assert end.hour == DAY_END_HOUR

    def test_inside_window_returns_start_to_now(self) -> None:
        dt = _local(2026, 6, 21, 12)
        start, end = get_daylight_span(dt)
        assert start.hour == DAY_START_HOUR
        assert end == dt

    def test_before_window_returns_yesterdays_window(self) -> None:
        dt = _local(2026, 6, 21, DAY_START_HOUR - 2)
        start, end = get_daylight_span(dt)
        yesterday = dt.date() - timedelta(days=1)
        assert start.date() == yesterday
        assert end.date() == yesterday
        assert start.hour == DAY_START_HOUR
        assert end.hour == DAY_END_HOUR


class TestGetDaylightSpanForDate:
    def test_returns_hardcoded_window(self) -> None:
        target = date(2026, 6, 21)
        start, end = get_daylight_span_for_date(target)
        assert start.date() == target
        assert end.date() == target
        assert start.hour == DAY_START_HOUR
        assert end.hour == DAY_END_HOUR

    def test_summer_and_winter_are_identical_length(self) -> None:
        s_start, s_end = get_daylight_span_for_date(date(2026, 6, 21))
        w_start, w_end = get_daylight_span_for_date(date(2026, 12, 21))
        assert (s_end - s_start) == (w_end - w_start)
        assert (s_end - s_start).total_seconds() == (DAY_END_HOUR - DAY_START_HOUR) * 3600

    def test_returns_aware_datetimes(self) -> None:
        start, end = get_daylight_span_for_date(date(2026, 3, 15))
        assert start.tzinfo is not None
        assert end.tzinfo is not None
