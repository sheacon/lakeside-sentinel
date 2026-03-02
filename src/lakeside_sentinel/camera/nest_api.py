import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import httpx
import isodate

from lakeside_sentinel.camera.auth import NestAuth
from lakeside_sentinel.camera.models import CameraEvent

logger = logging.getLogger(__name__)

BASE_URL = "https://nest-camera-frontend.googleapis.com"
EVENTS_PATH = "/dashmanifest/namespace/nest-phoenix-prod/device/{device_id}"
CLIP_PATH = "/mp4clip/namespace/nest-phoenix-prod/device/{device_id}"
MPD_NS = "{urn:mpeg:dash:schema:mpd:2011}"


class NestCameraAPI:
    """Client for the Google Nest internal camera API."""

    def __init__(self, auth: NestAuth, device_id: str) -> None:
        self._auth = auth
        self._device_id = device_id
        self._client = httpx.Client(timeout=30.0)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._auth.get_access_token()}"}

    def get_events(self, start_time: datetime, end_time: datetime) -> list[CameraEvent]:
        """Fetch camera events from the dashmanifest endpoint."""
        url = BASE_URL + EVENTS_PATH.format(device_id=self._device_id)
        params = {
            "start_time": _format_iso(start_time),
            "end_time": _format_iso(end_time),
            "types": "4",
            "variant": "2",
        }

        response = self._client.get(url, params=params, headers=self._headers())
        response.raise_for_status()

        return _parse_events(response.content)

    def download_clip(self, event: CameraEvent) -> bytes:
        """Download an MP4 clip for the given camera event."""
        url = BASE_URL + CLIP_PATH.format(device_id=self._device_id)
        params = {
            "start_time": int(event.start_time.timestamp() * 1000),
            "end_time": int(event.end_time.timestamp() * 1000),
        }

        response = self._client.get(url, params=params, headers=self._headers())
        response.raise_for_status()

        return response.content


def _format_iso(dt: datetime) -> str:
    """Format datetime as ISO 8601 with millisecond precision and Z suffix."""
    utc_dt = dt.astimezone(timezone.utc)
    return utc_dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{utc_dt.microsecond // 1000:03d}Z"


def _parse_events(xml_bytes: bytes) -> list[CameraEvent]:
    """Parse MPEG-DASH MPD XML into CameraEvent objects."""
    root = ET.fromstring(xml_bytes)
    periods = root.findall(f".//{MPD_NS}Period")

    events: list[CameraEvent] = []
    for period in periods:
        attrib = period.attrib
        if "programDateTime" not in attrib or "duration" not in attrib:
            continue

        start = datetime.fromisoformat(attrib["programDateTime"])
        duration = isodate.parse_duration(attrib["duration"])

        events.append(CameraEvent(start_time=start, duration=duration))

    logger.info("Parsed %d events from dashmanifest", len(events))
    return events
