from datetime import datetime, timezone

from lakeside_motorbikes.utils.daylight import is_daylight

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
