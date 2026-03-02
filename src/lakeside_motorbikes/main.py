from __future__ import annotations

import logging
import webbrowser
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from apscheduler.schedulers.blocking import BlockingScheduler

from lakeside_motorbikes.camera.auth import NestAuth
from lakeside_motorbikes.camera.models import CameraEvent
from lakeside_motorbikes.camera.nest_api import NestCameraAPI
from lakeside_motorbikes.cli import parse_args
from lakeside_motorbikes.config import Settings
from lakeside_motorbikes.detection.hsp_detector import HSPDetector
from lakeside_motorbikes.detection.models import Detection
from lakeside_motorbikes.detection.vehicle_detector import VehicleDetector
from lakeside_motorbikes.notification.email_sender import EmailSender
from lakeside_motorbikes.notification.html_report import ClipReport, generate_report
from lakeside_motorbikes.utils.daylight import (
    get_daylight_span,
    get_daylight_span_for_date,
    is_daylight,
)
from lakeside_motorbikes.utils.image import crop_to_bbox, crop_to_roi
from lakeside_motorbikes.utils.video import extract_frames

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_SENSITIVE_KEYWORDS = {"token", "key", "password", "secret"}


def _print_settings(settings: Settings) -> None:
    """Print all settings, masking sensitive values."""
    for name, value in settings.model_dump().items():
        if any(kw in name for kw in _SENSITIVE_KEYWORDS):
            display = "****"
        else:
            display = value
        label = name.replace("_", " ").title()
        print(f"  {label + ':':<28} {display}")


class Monitor:
    """Orchestrates the detection pipeline."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._auth = NestAuth(settings.google_master_token, settings.google_username)
        self._api = NestCameraAPI(self._auth, settings.nest_device_id)
        self._detector = VehicleDetector(
            model_name=settings.yolo_model,
            confidence_threshold=settings.yolo_confidence_threshold,
            batch_size=settings.yolo_batch_size,
        )
        self._email = EmailSender(
            api_key=settings.resend_api_key,
            from_email=settings.alert_email_from,
            to_email=settings.alert_email_to,
        )
        self._processed_events: set[str] = set()
        self._last_poll_time: datetime = datetime.now(timezone.utc)

    def _filter_daylight(self, events: list[CameraEvent]) -> list[CameraEvent]:
        """Filter events to only those occurring during daylight hours."""
        daylight_events = [
            e
            for e in events
            if is_daylight(
                e.start_time,
                self._settings.camera_latitude,
                self._settings.camera_longitude,
            )
        ]
        filtered = len(events) - len(daylight_events)
        if filtered:
            logger.info(
                "Filtered %d nighttime events (%d remaining)",
                filtered,
                len(daylight_events),
            )
        return daylight_events

    def process_event(self, event: CameraEvent) -> bool:
        """Process a single camera event. Returns True if a vehicle was detected."""
        logger.info(
            "Processing event: %s (duration: %s)",
            event.start_time.isoformat(),
            event.duration,
        )

        mp4_bytes = self._api.download_clip(event)
        if not mp4_bytes:
            logger.warning("Empty clip for event %s", event.event_id)
            return False

        frames = extract_frames(mp4_bytes, fps_sample=self._settings.fps_sample)
        if not frames:
            logger.warning("No frames extracted for event %s", event.event_id)
            return False

        frames = crop_to_roi(
            frames,
            y_start=self._settings.roi_y_start,
            y_end=self._settings.roi_y_end,
            x_start=self._settings.roi_x_start,
            x_end=self._settings.roi_x_end,
        )

        detection = self._detector.detect_best(frames)
        if detection is None:
            logger.info("No vehicle in event %s", event.event_id)
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
            class_name=detection.class_name,
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

        events = self._filter_daylight(events)
        new_events = [e for e in events if e.event_id not in self._processed_events]
        logger.info("Found %d new events (of %d total)", len(new_events), len(events))

        for event in new_events:
            self._processed_events.add(event.event_id)
            try:
                self.process_event(event)
            except Exception:
                logger.exception("Error processing event %s", event.event_id)

    def backfill(self, debug_dump: bool = False, backfill_date: date | None = None) -> None:
        """Download and analyze events for a daylight period."""
        if backfill_date is not None:
            start, end = get_daylight_span_for_date(
                backfill_date,
                self._settings.camera_latitude,
                self._settings.camera_longitude,
            )
            label = f"BACKFILL — {backfill_date.isoformat()}"
        else:
            now = datetime.now(timezone.utc)
            start, end = get_daylight_span(
                now,
                self._settings.camera_latitude,
                self._settings.camera_longitude,
            )
            label = "BACKFILL — Most recent daylight"
        now = end

        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"  From: {start.astimezone().strftime('%d %b %Y %H:%M:%S %Z')}")
        print(f"  To:   {now.astimezone().strftime('%d %b %Y %H:%M:%S %Z')}")
        print()
        _print_settings(self._settings)
        print(f"{'=' * 60}\n")

        print("[1/4] Fetching event list from Nest API...", flush=True)
        events = self._api.get_events(start, now)
        total_events = len(events)
        events = self._filter_daylight(events)
        filtered = total_events - len(events)
        print(f"       Found {total_events} events", end="")
        if filtered:
            print(f" ({filtered} nighttime filtered)")
        else:
            print()
        print()

        if not events:
            print("No events to process.")
            return

        dump_dir: Path | None = None
        if debug_dump:
            dump_dir = Path("output") / "backfill"
            dump_dir.mkdir(parents=True, exist_ok=True)
            print(f"[dump] Saving clips to: {dump_dir.resolve()}\n")

        print("[2/4] Downloading clips...")
        clips: list[tuple[int, bytes]] = []
        download_errors = 0
        total_bytes = 0
        cached_count = 0
        for i, event in enumerate(events):
            local_time = event.start_time.astimezone()
            label = local_time.strftime("%H:%M:%S")

            if dump_dir is not None:
                filename = local_time.strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"
                filepath = dump_dir / filename
                if filepath.exists():
                    mp4_bytes = filepath.read_bytes()
                    total_bytes += len(mp4_bytes)
                    clips.append((i, mp4_bytes))
                    cached_count += 1
                    size_mb = len(mp4_bytes) / 1_000_000
                    print(
                        f"  [{i + 1:3d}/{len(events)}] {label} — {size_mb:.1f} MB (cached)",
                        flush=True,
                    )
                    continue

            try:
                mp4_bytes = self._api.download_clip(event)
                if not mp4_bytes:
                    print(f"  [{i + 1:3d}/{len(events)}] {label} — empty clip (skipped)")
                    download_errors += 1
                    continue
                total_bytes += len(mp4_bytes)
                clips.append((i, mp4_bytes))
                size_mb = len(mp4_bytes) / 1_000_000
                dur = event.duration.total_seconds()
                print(
                    f"  [{i + 1:3d}/{len(events)}] {label} — {size_mb:.1f} MB ({dur:.0f}s)",
                    flush=True,
                )

                if dump_dir is not None:
                    filename = local_time.strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"
                    (dump_dir / filename).write_bytes(mp4_bytes)
            except Exception as e:
                print(f"  [{i + 1:3d}/{len(events)}] {label} — ERROR: {e}")
                download_errors += 1

        total_mb = total_bytes / 1_000_000
        downloaded = len(clips) - cached_count
        print(
            f"\n       {downloaded} downloaded, {cached_count} cached"
            f" — {len(clips)}/{len(events)} clips ({total_mb:.1f} MB total)"
        )
        if download_errors:
            print(f"       {download_errors} download errors")
        print()

        print("[3/4] Analyzing frames with YOLO...")
        detection_count = 0
        total_frames = 0
        collected_detections: list[tuple["np.ndarray", float, str, datetime]] = []
        clip_reports: list[ClipReport] = []
        for idx, (event_i, mp4_bytes) in enumerate(clips):
            event = events[event_i]
            local_time = event.start_time.astimezone()
            label = local_time.strftime("%H:%M:%S")

            frames = extract_frames(mp4_bytes, fps_sample=self._settings.fps_sample)
            frames = crop_to_roi(
                frames,
                y_start=self._settings.roi_y_start,
                y_end=self._settings.roi_y_end,
                x_start=self._settings.roi_x_start,
                x_end=self._settings.roi_x_end,
            )
            total_frames += len(frames)
            if not frames:
                print(f"  [{idx + 1:3d}/{len(clips)}] {label} — no frames extracted")
                if debug_dump:
                    mp4_fn = local_time.strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"
                    clip_reports.append(
                        ClipReport(
                            event_time=event.start_time,
                            mp4_filename=mp4_fn,
                            best_detection=None,
                            class_detections={},
                        )
                    )
                continue

            if debug_dump:
                detection, class_best = self._detector.detect_detailed(frames)
            else:
                detection = self._detector.detect_best(frames)

            if debug_dump:
                mp4_fn = local_time.strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"
                clip_reports.append(
                    ClipReport(
                        event_time=event.start_time,
                        mp4_filename=mp4_fn,
                        best_detection=detection,
                        class_detections=class_best,
                    )
                )

            if detection is None:
                print(
                    f"  [{idx + 1:3d}/{len(clips)}] {label} — {len(frames):2d} frames — no vehicle",
                    flush=True,
                )
                if debug_dump and class_best:
                    breakdown = "  ".join(
                        f"{name}: {det.confidence:.0%}" for name, det in sorted(class_best.items())
                    )
                    print(f"            {breakdown}")
                continue

            detection_count += 1
            print(
                f"  [{idx + 1:3d}/{len(clips)}] {label} — {len(frames):2d} frames — "
                f"{detection.class_name.upper()} (confidence: {detection.confidence:.0%})",
                flush=True,
            )
            if debug_dump:
                breakdown = "  ".join(
                    f"{name}: {det.confidence:.0%}" for name, det in sorted(class_best.items())
                )
                print(f"            {breakdown}")

            cropped = crop_to_bbox(
                detection.frame,
                detection.bbox,
                padding=self._settings.crop_padding,
            )

            collected_detections.append(
                (cropped, detection.confidence, detection.class_name, event.start_time)
            )

        email_id: str | None = None
        report_path: Path | None = None

        if debug_dump and dump_dir is not None:
            print("\n[4/4] Generating HTML report...")
            report_path = generate_report(
                clip_reports,
                dump_dir,
                crop_padding=self._settings.crop_padding,
            )
            print(f"       Report: {report_path.resolve()}")
            webbrowser.open(report_path.resolve().as_uri())
        else:
            print("\n[4/4] Sending summary email...")
            email_id = self._email.send_backfill_summary(collected_detections)
            if email_id:
                print(f"       Email sent ({email_id})")
            elif collected_detections:
                print("       Email FAILED")
            else:
                print("       No detections — no email sent")

        print(f"\n{'=' * 60}")
        print("  BACKFILL COMPLETE")
        print(f"  Events:     {len(events)}")
        print(f"  Downloaded: {len(clips)} clips ({total_mb:.1f} MB)")
        print(f"  Frames:     {total_frames} analyzed")
        print(f"  Detections: {detection_count}")
        if report_path:
            print(f"  Report:     {report_path.resolve()}")
        else:
            print(f"  Email:      {'sent' if email_id else 'none'}")
        if dump_dir is not None:
            print(f"  Clips:      {dump_dir.resolve()}")
        print(f"{'=' * 60}\n")

    def backfill_hsp(self, debug_dump: bool = False, backfill_date: date | None = None) -> None:
        """Backfill with experimental high-speed person (HSP) detection."""
        if backfill_date is not None:
            start, end = get_daylight_span_for_date(
                backfill_date,
                self._settings.camera_latitude,
                self._settings.camera_longitude,
            )
            label = f"HSP BACKFILL (experimental) — {backfill_date.isoformat()}"
        else:
            now = datetime.now(timezone.utc)
            start, end = get_daylight_span(
                now,
                self._settings.camera_latitude,
                self._settings.camera_longitude,
            )
            label = "HSP BACKFILL (experimental) — Most recent daylight"
        now = end

        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"  From: {start.astimezone().strftime('%d %b %Y %H:%M:%S %Z')}")
        print(f"  To:   {now.astimezone().strftime('%d %b %Y %H:%M:%S %Z')}")
        print()
        _print_settings(self._settings)
        print(f"{'=' * 60}\n")

        hsp_detector = HSPDetector(
            model_name=self._settings.yolo_model,
            person_confidence=self._settings.hsp_person_confidence,
            displacement_threshold=self._settings.hsp_displacement_threshold,
            max_match_distance=self._settings.hsp_max_match_distance,
            batch_size=self._settings.yolo_batch_size,
        )

        print("[1/4] Fetching event list from Nest API...", flush=True)
        events = self._api.get_events(start, now)
        total_events = len(events)
        events = self._filter_daylight(events)
        filtered = total_events - len(events)
        print(f"       Found {total_events} events", end="")
        if filtered:
            print(f" ({filtered} nighttime filtered)")
        else:
            print()
        print()

        if not events:
            print("No events to process.")
            return

        dump_dir: Path | None = None
        if debug_dump:
            dump_dir = Path("output") / "backfill-hsp"
            dump_dir.mkdir(parents=True, exist_ok=True)
            print(f"[dump] Saving clips to: {dump_dir.resolve()}\n")

        print("[2/4] Downloading clips...")
        clips: list[tuple[int, bytes]] = []
        download_errors = 0
        total_bytes = 0
        cached_count = 0
        for i, event in enumerate(events):
            local_time = event.start_time.astimezone()
            label = local_time.strftime("%H:%M:%S")

            if dump_dir is not None:
                filename = local_time.strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"
                filepath = dump_dir / filename
                if filepath.exists():
                    mp4_bytes = filepath.read_bytes()
                    total_bytes += len(mp4_bytes)
                    clips.append((i, mp4_bytes))
                    cached_count += 1
                    size_mb = len(mp4_bytes) / 1_000_000
                    print(
                        f"  [{i + 1:3d}/{len(events)}] {label} — {size_mb:.1f} MB (cached)",
                        flush=True,
                    )
                    continue

            try:
                mp4_bytes = self._api.download_clip(event)
                if not mp4_bytes:
                    print(f"  [{i + 1:3d}/{len(events)}] {label} — empty clip (skipped)")
                    download_errors += 1
                    continue
                total_bytes += len(mp4_bytes)
                clips.append((i, mp4_bytes))
                size_mb = len(mp4_bytes) / 1_000_000
                dur = event.duration.total_seconds()
                print(
                    f"  [{i + 1:3d}/{len(events)}] {label} — {size_mb:.1f} MB ({dur:.0f}s)",
                    flush=True,
                )

                if dump_dir is not None:
                    filename = local_time.strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"
                    (dump_dir / filename).write_bytes(mp4_bytes)
            except Exception as e:
                print(f"  [{i + 1:3d}/{len(events)}] {label} — ERROR: {e}")
                download_errors += 1

        total_mb = total_bytes / 1_000_000
        downloaded = len(clips) - cached_count
        print(
            f"\n       {downloaded} downloaded, {cached_count} cached"
            f" — {len(clips)}/{len(events)} clips ({total_mb:.1f} MB total)"
        )
        if download_errors:
            print(f"       {download_errors} download errors")
        print()

        fps = self._settings.hsp_fps_sample
        print(f"[3/4] Analyzing frames for high-speed persons (fps={fps})...")
        detection_count = 0
        total_frames = 0
        collected_detections: list[tuple[np.ndarray, float, str, datetime]] = []
        clip_reports: list[ClipReport] = []
        for idx, (event_i, mp4_bytes) in enumerate(clips):
            event = events[event_i]
            local_time = event.start_time.astimezone()
            label = local_time.strftime("%H:%M:%S")

            frames = extract_frames(mp4_bytes, fps_sample=fps)
            frames = crop_to_roi(
                frames,
                y_start=self._settings.roi_y_start,
                y_end=self._settings.roi_y_end,
                x_start=self._settings.roi_x_start,
                x_end=self._settings.roi_x_end,
            )
            total_frames += len(frames)
            if not frames:
                print(f"  [{idx + 1:3d}/{len(clips)}] {label} — no frames extracted")
                if debug_dump:
                    mp4_fn = local_time.strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"
                    clip_reports.append(
                        ClipReport(
                            event_time=event.start_time,
                            mp4_filename=mp4_fn,
                            best_detection=None,
                            class_detections={},
                        )
                    )
                continue

            tracks = hsp_detector.detect_all_tracks(frames)
            detection = hsp_detector.detect(frames)

            if debug_dump:
                # Log all track displacements for threshold tuning
                for t_idx, track in enumerate(tracks):
                    disp = track.displacement_per_interval
                    pts = len(track.points)
                    logger.info(
                        "  Track %d: %d points, displacement=%.1f px/interval",
                        t_idx,
                        pts,
                        disp,
                    )

                mp4_fn = local_time.strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"
                # Build class_detections with the HSP detection if present
                class_dets: dict[str, Detection] = {}
                if detection:
                    class_dets["HSP"] = detection
                clip_reports.append(
                    ClipReport(
                        event_time=event.start_time,
                        mp4_filename=mp4_fn,
                        best_detection=detection,
                        class_detections=class_dets,
                    )
                )

            track_info = ""
            if tracks:
                displacements = [t.displacement_per_interval for t in tracks]
                max_disp = max(displacements)
                track_info = f" — {len(tracks)} tracks (max disp: {max_disp:.1f})"

            if detection is None:
                print(
                    f"  [{idx + 1:3d}/{len(clips)}] {label} — {len(frames):2d} frames"
                    f" — no HSP{track_info}",
                    flush=True,
                )
                continue

            detection_count += 1
            print(
                f"  [{idx + 1:3d}/{len(clips)}] {label} — {len(frames):2d} frames — "
                f"HSP (confidence: {detection.confidence:.0%}){track_info}",
                flush=True,
            )

            cropped = crop_to_bbox(
                detection.frame,
                detection.bbox,
                padding=self._settings.crop_padding,
            )
            collected_detections.append(
                (cropped, detection.confidence, detection.class_name, event.start_time)
            )

        report_path: Path | None = None

        if debug_dump and dump_dir is not None:
            print("\n[4/4] Generating HTML report...")
            report_path = generate_report(
                clip_reports,
                dump_dir,
                crop_padding=self._settings.crop_padding,
            )
            print(f"       Report: {report_path.resolve()}")
            webbrowser.open(report_path.resolve().as_uri())
        else:
            print("\n[4/4] Sending summary email...")
            email_id = self._email.send_backfill_summary(collected_detections)
            if email_id:
                print(f"       Email sent ({email_id})")
            elif collected_detections:
                print("       Email FAILED")
            else:
                print("       No detections — no email sent")

        print(f"\n{'=' * 60}")
        print("  HSP BACKFILL COMPLETE")
        print(f"  Events:     {len(events)}")
        print(f"  Downloaded: {len(clips)} clips ({total_mb:.1f} MB)")
        print(f"  Frames:     {total_frames} analyzed (fps={fps})")
        print(f"  Detections: {detection_count}")
        if report_path:
            print(f"  Report:     {report_path.resolve()}")
        if dump_dir is not None:
            print(f"  Clips:      {dump_dir.resolve()}")
        print(f"{'=' * 60}\n")

    def run_live(self) -> None:
        """Start the live monitor with scheduled polling."""
        logger.info("Starting live monitor (poll every %ds)", self._settings.poll_interval_seconds)

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

    backfill_date: date | None = None
    if args.date:
        backfill_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        args.backfill = True  # --date implies --backfill

    monitor = Monitor(settings)

    if args.hsp and args.backfill:
        monitor.backfill_hsp(debug_dump=args.debug_dump, backfill_date=backfill_date)
    elif args.backfill:
        monitor.backfill(debug_dump=args.debug_dump, backfill_date=backfill_date)
    else:
        monitor.run_live()


if __name__ == "__main__":
    main()
