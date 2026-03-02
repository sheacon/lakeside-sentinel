from datetime import datetime, timedelta, timezone

from lakeside_sentinel.camera.nest_api import _format_iso, _parse_events


class TestFormatIso:
    def test_formats_utc_datetime(self) -> None:
        dt = datetime(2026, 2, 28, 14, 30, 45, 250000, tzinfo=timezone.utc)
        assert _format_iso(dt) == "2026-02-28T14:30:45.250Z"

    def test_formats_zero_microseconds(self) -> None:
        dt = datetime(2026, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc)
        assert _format_iso(dt) == "2026-01-01T00:00:00.000Z"

    def test_converts_non_utc_to_utc(self) -> None:
        tz_plus_5 = timezone(timedelta(hours=5))
        dt = datetime(2026, 2, 28, 19, 30, 0, 0, tzinfo=tz_plus_5)
        assert _format_iso(dt) == "2026-02-28T14:30:00.000Z"


class TestParseEvents:
    def test_parses_three_events(self, sample_dashmanifest_xml: bytes) -> None:
        events = _parse_events(sample_dashmanifest_xml)
        assert len(events) == 3

    def test_first_event_start_time(self, sample_dashmanifest_xml: bytes) -> None:
        events = _parse_events(sample_dashmanifest_xml)
        assert events[0].start_time == datetime(2026, 2, 28, 10, 0, 0, tzinfo=timezone.utc)

    def test_first_event_duration(self, sample_dashmanifest_xml: bytes) -> None:
        events = _parse_events(sample_dashmanifest_xml)
        assert events[0].duration == timedelta(seconds=20)

    def test_second_event_end_time(self, sample_dashmanifest_xml: bytes) -> None:
        events = _parse_events(sample_dashmanifest_xml)
        expected_end = datetime(2026, 2, 28, 10, 6, 15, 500000, tzinfo=timezone.utc)
        assert events[1].end_time == expected_end

    def test_long_event_preserves_full_duration(self, sample_dashmanifest_xml: bytes) -> None:
        events = _parse_events(sample_dashmanifest_xml)
        # Third event has duration PT2M30S and should be preserved in full
        assert events[2].duration == timedelta(minutes=2, seconds=30)

    def test_event_ids_are_unique(self, sample_dashmanifest_xml: bytes) -> None:
        events = _parse_events(sample_dashmanifest_xml)
        ids = [e.event_id for e in events]
        assert len(ids) == len(set(ids))

    def test_empty_xml_returns_empty_list(self) -> None:
        xml = b'<?xml version="1.0"?><MPD xmlns="urn:mpeg:dash:schema:mpd:2011"></MPD>'
        events = _parse_events(xml)
        assert events == []
