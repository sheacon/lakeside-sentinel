import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler

from lakeside_motorbikes.camera.auth import NestAuth
from lakeside_motorbikes.camera.models import CameraEvent
from lakeside_motorbikes.camera.nest_api import NestCameraAPI
from lakeside_motorbikes.cli import parse_args
from lakeside_motorbikes.config import Settings
from lakeside_motorbikes.detection.yolo_detector import MotorcycleDetector
from lakeside_motorbikes.notification.email_sender import EmailSender
from lakeside_motorbikes.utils.image import crop_to_bbox
from lakeside_motorbikes.utils.video import extract_frames

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class Monitor:
    """Orchestrates the detection pipeline."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._auth = NestAuth(settings.google_master_token, settings.google_username)
        self._api = NestCameraAPI(self._auth, settings.nest_device_id)
        self._detector = MotorcycleDetector(
            confidence_threshold=settings.yolo_confidence_threshold,
        )
        self._email = EmailSender(
            api_key=settings.resend_api_key,
            from_email=settings.alert_email_from,
            to_email=settings.alert_email_to,
        )
        self._processed_events: set[str] = set()
        self._last_poll_time: datetime = datetime.now(timezone.utc)

    def process_event(self, event: CameraEvent) -> bool:
        """Process a single camera event. Returns True if a motorbike was detected."""
        logger.info(
            "Processing event: %s (duration: %s)",
            event.start_time.isoformat(),
            event.duration,
        )

        mp4_bytes = self._api.download_clip(event)
        if not mp4_bytes:
            logger.warning("Empty clip for event %s", event.event_id)
            return False

        frames = extract_frames(mp4_bytes)
        if not frames:
            logger.warning("No frames extracted for event %s", event.event_id)
            return False

        detection = self._detector.detect_best(frames)
        if detection is None:
            logger.info("No motorcycle in event %s", event.event_id)
            return False

        cropped = crop_to_bbox(
            detection.frame,
            detection.bbox,
            padding=self._settings.crop_padding,
        )

        self._email.send_alert(
            cropped_image=cropped,
            confidence=detection.confidence,
            event_time=event.start_time,
        )
        return True

    def poll(self) -> None:
        """Poll for new events since the last poll."""
        now = datetime.now(timezone.utc)
        start = self._last_poll_time
        self._last_poll_time = now

        logger.info("Polling events from %s to %s", start.isoformat(), now.isoformat())

        try:
            events = self._api.get_events(start, now)
        except Exception:
            logger.exception("Failed to fetch events")
            return

        new_events = [e for e in events if e.event_id not in self._processed_events]
        logger.info("Found %d new events (of %d total)", len(new_events), len(events))

        for event in new_events:
            self._processed_events.add(event.event_id)
            try:
                self.process_event(event)
            except Exception:
                logger.exception("Error processing event %s", event.event_id)

    def backfill(self) -> None:
        """Download and analyze all events from the past 24 hours."""
        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=24)

        logger.info("Backfill: fetching events from %s to %s", start.isoformat(), now.isoformat())

        events = self._api.get_events(start, now)
        logger.info("Backfill: found %d events", len(events))

        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        detections = 0
        for i, event in enumerate(events, 1):
            logger.info("Backfill: processing event %d/%d", i, len(events))
            try:
                if self.process_event(event):
                    detections += 1
            except Exception:
                logger.exception("Error processing event %s", event.event_id)

        logger.info(
            "Backfill complete: %d events processed, %d detections",
            len(events),
            detections,
        )

    def run_live(self) -> None:
        """Start the live monitor with scheduled polling."""
        logger.info(
            "Starting live monitor (poll every %ds)", self._settings.poll_interval_seconds
        )

        # Run an initial poll immediately
        self.poll()

        scheduler = BlockingScheduler()
        scheduler.add_job(
            self.poll,
            "interval",
            seconds=self._settings.poll_interval_seconds,
        )

        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Monitor stopped")


def main() -> None:
    args = parse_args()
    settings = Settings()  # type: ignore[call-arg]

    monitor = Monitor(settings)

    if args.backfill:
        monitor.backfill()
    else:
        monitor.run_live()


if __name__ == "__main__":
    main()
