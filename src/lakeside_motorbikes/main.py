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

    def backfill(self, debug_dump: bool = False) -> None:
        """Download and analyze all events from the past 24 hours."""
        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=24)

        print(f"\n{'='*60}")
        print(f"  BACKFILL — Last 24 hours")
        print(f"  From: {start.astimezone().strftime('%d %b %Y %H:%M:%S %Z')}")
        print(f"  To:   {now.astimezone().strftime('%d %b %Y %H:%M:%S %Z')}")
        print(f"{'='*60}\n")

        print("[1/4] Fetching event list from Nest API...", flush=True)
        events = self._api.get_events(start, now)
        print(f"       Found {len(events)} events\n")

        if not events:
            print("No events to process.")
            return

        dump_dir: Path | None = None
        if debug_dump:
            dump_dir = Path("output") / f"debug_{now.strftime('%Y%m%d_%H%M%S')}"
            dump_dir.mkdir(parents=True, exist_ok=True)
            print(f"[dump] Saving clips to: {dump_dir.resolve()}\n")

        print(f"[2/4] Downloading clips...")
        clips: list[tuple[int, bytes]] = []
        download_errors = 0
        total_bytes = 0
        for i, event in enumerate(events):
            local_time = event.start_time.astimezone()
            label = local_time.strftime("%H:%M:%S")
            try:
                mp4_bytes = self._api.download_clip(event)
                if not mp4_bytes:
                    print(f"  [{i+1:3d}/{len(events)}] {label} — empty clip (skipped)")
                    download_errors += 1
                    continue
                total_bytes += len(mp4_bytes)
                clips.append((i, mp4_bytes))
                size_mb = len(mp4_bytes) / 1_000_000
                print(f"  [{i+1:3d}/{len(events)}] {label} — {size_mb:.1f} MB ({event.duration.total_seconds():.0f}s)", flush=True)

                if dump_dir is not None:
                    filename = local_time.strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"
                    (dump_dir / filename).write_bytes(mp4_bytes)
            except Exception as e:
                print(f"  [{i+1:3d}/{len(events)}] {label} — ERROR: {e}")
                download_errors += 1

        total_mb = total_bytes / 1_000_000
        print(f"\n       Downloaded {len(clips)}/{len(events)} clips ({total_mb:.1f} MB total)")
        if download_errors:
            print(f"       {download_errors} download errors")
        print()

        print(f"[3/4] Analyzing frames with YOLO...")
        detections = 0
        emails_sent = 0
        total_frames = 0
        for idx, (event_i, mp4_bytes) in enumerate(clips):
            event = events[event_i]
            local_time = event.start_time.astimezone()
            label = local_time.strftime("%H:%M:%S")

            frames = extract_frames(mp4_bytes)
            total_frames += len(frames)
            if not frames:
                print(f"  [{idx+1:3d}/{len(clips)}] {label} — no frames extracted")
                continue

            detection = self._detector.detect_best(frames)
            if detection is None:
                print(f"  [{idx+1:3d}/{len(clips)}] {label} — {len(frames):2d} frames — no motorcycle", flush=True)
                continue

            detections += 1
            print(
                f"  [{idx+1:3d}/{len(clips)}] {label} — {len(frames):2d} frames — "
                f"MOTORCYCLE (confidence: {detection.confidence:.0%})",
                flush=True,
            )

            cropped = crop_to_bbox(
                detection.frame,
                detection.bbox,
                padding=self._settings.crop_padding,
            )

            result = self._email.send_alert(
                cropped_image=cropped,
                confidence=detection.confidence,
                event_time=event.start_time,
            )
            if result:
                emails_sent += 1
                print(f"         → Email sent ({result})")
            else:
                print(f"         → Email FAILED")

        print(f"\n{'='*60}")
        print(f"  BACKFILL COMPLETE")
        print(f"  Events:     {len(events)}")
        print(f"  Downloaded: {len(clips)} clips ({total_mb:.1f} MB)")
        print(f"  Frames:     {total_frames} analyzed")
        print(f"  Detections: {detections}")
        print(f"  Emails:     {emails_sent} sent")
        if dump_dir is not None:
            print(f"  Clips:      {dump_dir.resolve()}")
        print(f"{'='*60}\n")

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
        monitor.backfill(debug_dump=args.debug_dump)
    else:
        monitor.run_live()


if __name__ == "__main__":
    main()
