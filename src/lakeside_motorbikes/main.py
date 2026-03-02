from __future__ import annotations

import logging
import webbrowser
from datetime import date, datetime, timezone
from pathlib import Path

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
from lakeside_motorbikes.utils.image import crop_to_roi
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

    def run(self, send_email: bool = False, target_date: date | None = None) -> None:
        """Download and analyze events for a daylight period."""
        if target_date is not None:
            start, end = get_daylight_span_for_date(
                target_date,
                self._settings.camera_latitude,
                self._settings.camera_longitude,
            )
            label = f"DAILY DETECTION — {target_date.isoformat()}"
        else:
            now = datetime.now(timezone.utc)
            start, end = get_daylight_span(
                now,
                self._settings.camera_latitude,
                self._settings.camera_longitude,
            )
            label = "DAILY DETECTION — Most recent daylight"
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

        dump_dir = Path("output") / "detections"
        dump_dir.mkdir(parents=True, exist_ok=True)
        print(f"[clips] Saving to: {dump_dir.resolve()}\n")

        print("[2/4] Downloading clips...")
        clips: list[tuple[int, bytes]] = []
        download_errors = 0
        total_bytes = 0
        cached_count = 0
        for i, event in enumerate(events):
            local_time = event.start_time.astimezone()
            label = local_time.strftime("%H:%M:%S")

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

            detection, class_best = self._detector.detect_detailed(frames)

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
                if class_best:
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
            breakdown = "  ".join(
                f"{name}: {det.confidence:.0%}" for name, det in sorted(class_best.items())
            )
            print(f"            {breakdown}")

        print("\n[4/4] Generating HTML report...")
        html = generate_report(
            clip_reports,
            crop_padding=self._settings.crop_padding,
            include_video=True,
        )
        report_path = dump_dir / "report.html"
        report_path.write_text(html)
        logger.info("HTML report written to %s", report_path)
        print(f"       Report: {report_path.resolve()}")
        webbrowser.open(report_path.resolve().as_uri())

        email_id: str | None = None
        if send_email:
            email_html = generate_report(
                clip_reports,
                crop_padding=self._settings.crop_padding,
                include_video=False,
            )
            email_id = self._email.send_report(email_html, f"Daily Detection Report — {label}")
            if email_id:
                print(f"       Email sent ({email_id})")
            else:
                print("       Email FAILED")

        print(f"\n{'=' * 60}")
        print("  DAILY DETECTION COMPLETE")
        print(f"  Events:     {len(events)}")
        print(f"  Downloaded: {len(clips)} clips ({total_mb:.1f} MB)")
        print(f"  Frames:     {total_frames} analyzed")
        print(f"  Detections: {detection_count}")
        print(f"  Report:     {report_path.resolve()}")
        if send_email:
            print(f"  Email:      {'sent' if email_id else 'failed'}")
        print(f"  Clips:      {dump_dir.resolve()}")
        print(f"{'=' * 60}\n")

    def run_hsp(self, send_email: bool = False, target_date: date | None = None) -> None:
        """Run with experimental high-speed person (HSP) detection."""
        if target_date is not None:
            start, end = get_daylight_span_for_date(
                target_date,
                self._settings.camera_latitude,
                self._settings.camera_longitude,
            )
            label = f"HSP DETECTION (experimental) — {target_date.isoformat()}"
        else:
            now = datetime.now(timezone.utc)
            start, end = get_daylight_span(
                now,
                self._settings.camera_latitude,
                self._settings.camera_longitude,
            )
            label = "HSP DETECTION (experimental) — Most recent daylight"
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

        dump_dir = Path("output") / "detections-hsp"
        dump_dir.mkdir(parents=True, exist_ok=True)
        print(f"[clips] Saving to: {dump_dir.resolve()}\n")

        print("[2/4] Downloading clips...")
        clips: list[tuple[int, bytes]] = []
        download_errors = 0
        total_bytes = 0
        cached_count = 0
        for i, event in enumerate(events):
            local_time = event.start_time.astimezone()
            label = local_time.strftime("%H:%M:%S")

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

        print("\n[4/4] Generating HTML report...")
        html = generate_report(
            clip_reports,
            crop_padding=self._settings.crop_padding,
            include_video=True,
        )
        report_path = dump_dir / "report.html"
        report_path.write_text(html)
        logger.info("HTML report written to %s", report_path)
        print(f"       Report: {report_path.resolve()}")
        webbrowser.open(report_path.resolve().as_uri())

        email_id: str | None = None
        if send_email:
            email_html = generate_report(
                clip_reports,
                crop_padding=self._settings.crop_padding,
                include_video=False,
            )
            email_id = self._email.send_report(email_html, f"HSP Detection Report — {label}")
            if email_id:
                print(f"       Email sent ({email_id})")
            else:
                print("       Email FAILED")

        print(f"\n{'=' * 60}")
        print("  HSP DETECTION COMPLETE")
        print(f"  Events:     {len(events)}")
        print(f"  Downloaded: {len(clips)} clips ({total_mb:.1f} MB)")
        print(f"  Frames:     {total_frames} analyzed (fps={fps})")
        print(f"  Detections: {detection_count}")
        print(f"  Report:     {report_path.resolve()}")
        if send_email:
            print(f"  Email:      {'sent' if email_id else 'failed'}")
        print(f"  Clips:      {dump_dir.resolve()}")
        print(f"{'=' * 60}\n")


def main() -> None:
    args = parse_args()
    settings = Settings()  # type: ignore[call-arg]

    target_date: date | None = None
    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()

    monitor = Monitor(settings)

    if args.hsp:
        monitor.run_hsp(send_email=args.email, target_date=target_date)
    else:
        monitor.run(send_email=args.email, target_date=target_date)


if __name__ == "__main__":
    main()
