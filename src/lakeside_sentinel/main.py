from __future__ import annotations

import logging
import webbrowser
from datetime import date, datetime, timezone
from pathlib import Path

from lakeside_sentinel.camera.auth import NestAuth
from lakeside_sentinel.camera.models import CameraEvent
from lakeside_sentinel.camera.nest_api import NestCameraAPI
from lakeside_sentinel.cli import parse_args
from lakeside_sentinel.config import Settings
from lakeside_sentinel.detection.claude_verifier import ClaudeVerifier
from lakeside_sentinel.detection.hsp_detector import HSPDetector
from lakeside_sentinel.detection.models import Detection
from lakeside_sentinel.detection.veh_detector import VEHDetector
from lakeside_sentinel.notification.email_sender import EmailSender
from lakeside_sentinel.notification.html_report import ClipReport, generate_report
from lakeside_sentinel.utils.daylight import (
    get_daylight_span,
    get_daylight_span_for_date,
    is_daylight,
)
from lakeside_sentinel.utils.image import crop_to_roi
from lakeside_sentinel.utils.video import extract_frames

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
        self._veh_detector = VEHDetector(
            model_name=settings.yolo_model,
            confidence_threshold=settings.veh_confidence_threshold,
            batch_size=settings.yolo_batch_size,
        )
        self._hsp_detector = HSPDetector(
            model_name=settings.yolo_model,
            person_confidence=settings.hsp_person_confidence_threshold,
            displacement_threshold=settings.hsp_displacement_threshold,
            max_match_distance=settings.hsp_max_match_distance,
            batch_size=settings.yolo_batch_size,
            fps_sample=settings.hsp_fps_sample,
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

    # ── Shared pipeline helpers ──────────────────────────────────────

    def _resolve_daylight_span(
        self,
        target_date: date | None,
        label_prefix: str,
    ) -> tuple[datetime, datetime, str]:
        """Resolve daylight start/end and print the banner.

        Returns (start, end, label).
        """
        if target_date is not None:
            start, end = get_daylight_span_for_date(
                target_date,
                self._settings.camera_latitude,
                self._settings.camera_longitude,
            )
            label = f"{label_prefix} — {target_date.isoformat()}"
        else:
            now = datetime.now(timezone.utc)
            start, end = get_daylight_span(
                now,
                self._settings.camera_latitude,
                self._settings.camera_longitude,
            )
            label = f"{label_prefix} — Most recent daylight"

        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"  From: {start.astimezone().strftime('%d %b %Y %H:%M:%S %Z')}")
        print(f"  To:   {end.astimezone().strftime('%d %b %Y %H:%M:%S %Z')}")
        print()
        _print_settings(self._settings)
        print(f"{'=' * 60}\n")

        return start, end, label

    def _fetch_events(
        self,
        start: datetime,
        end: datetime,
    ) -> list[CameraEvent]:
        """Step 1: Fetch events from Nest API and filter to daylight."""
        print("[1/4] Fetching event list from Nest API...", flush=True)
        events = self._api.get_events(start, end)
        total_events = len(events)
        events = self._filter_daylight(events)
        filtered = total_events - len(events)
        print(f"       Found {total_events} events", end="")
        if filtered:
            print(f" ({filtered} nighttime filtered)")
        else:
            print()
        print()
        return events

    def _download_clips(
        self,
        events: list[CameraEvent],
    ) -> tuple[list[tuple[int, bytes]], float]:
        """Step 2: Download clips (with caching).

        Returns (clips, total_mb).
        """
        dump_dir = Path("output") / "video"
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

        return clips, total_mb

    def _detect_veh(
        self,
        clips: list[tuple[int, bytes]],
        events: list[CameraEvent],
    ) -> tuple[list[ClipReport], int, int]:
        """Step 3 VEH: YOLO detection.

        Returns (clip_reports, detection_count, total_frames).
        """
        print("[3/4] Analyzing frames with YOLO...")
        detection_count = 0
        total_frames = 0
        clip_reports: list[ClipReport] = []
        for idx, (event_i, mp4_bytes) in enumerate(clips):
            event = events[event_i]
            local_time = event.start_time.astimezone()
            label = local_time.strftime("%H:%M:%S")

            frames = extract_frames(mp4_bytes, fps_sample=self._settings.veh_fps_sample)
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
                mp4_fn = "video/" + local_time.strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"
                clip_reports.append(
                    ClipReport(
                        event_time=event.start_time,
                        mp4_filename=mp4_fn,
                        best_detection=None,
                        class_detections={},
                    )
                )
                continue

            detection, class_best = self._veh_detector.detect_detailed(frames)

            # Filter to only above-threshold detections for the report
            class_above = {
                name: det
                for name, det in class_best.items()
                if det.confidence >= self._settings.veh_confidence_threshold
            }

            mp4_fn = "video/" + local_time.strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"
            clip_reports.append(
                ClipReport(
                    event_time=event.start_time,
                    mp4_filename=mp4_fn,
                    best_detection=detection,
                    class_detections=class_above,
                )
            )

            if detection is None:
                print(
                    f"  [{idx + 1:3d}/{len(clips)}] {label}"
                    f" — {len(frames):2d} frames — no detection",
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

        return clip_reports, detection_count, total_frames

    def _detect_hsp(
        self,
        clips: list[tuple[int, bytes]],
        events: list[CameraEvent],
    ) -> tuple[list[ClipReport], int, int]:
        """Step 3 HSP: Person tracking.

        Returns (clip_reports, detection_count, total_frames).
        """
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
                mp4_fn = "video/" + local_time.strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"
                clip_reports.append(
                    ClipReport(
                        event_time=event.start_time,
                        mp4_filename=mp4_fn,
                        best_detection=None,
                        class_detections={},
                    )
                )
                continue

            tracks = self._hsp_detector.detect_all_tracks(frames)
            detection = self._hsp_detector.detect(frames)

            # Log all track displacements for threshold tuning
            for t_idx, track in enumerate(tracks):
                disp = track.displacement_per_second(fps)
                pts = len(track.points)
                logger.info(
                    "  Track %d: %d points, displacement=%.1f px/sec",
                    t_idx,
                    pts,
                    disp,
                )

            mp4_fn = "video/" + local_time.strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"
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
                displacements = [t.displacement_per_second(fps) for t in tracks]
                max_disp = max(displacements)
                track_info = f" — {len(tracks)} tracks (max disp: {max_disp:.1f} px/sec)"

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

        return clip_reports, detection_count, total_frames

    def _verify_with_claude(
        self,
        clip_reports: list[ClipReport],
        keep_rejected: bool,
    ) -> tuple[list[ClipReport], int]:
        """Step 3.5: Claude verification.

        Returns (updated_clip_reports, detection_count).
        """
        print("\n[3.5/4] Verifying detections with Claude Vision...")
        verifier = ClaudeVerifier(
            api_key=self._settings.anthropic_api_key,
            model=self._settings.claude_vision_model,
            crop_padding=self._settings.crop_padding,
        )
        confirmed_count = 0
        rejected_count = 0
        for idx, report in enumerate(clip_reports):
            if not report.class_detections:
                continue
            verified = verifier.verify_detections(report.class_detections)
            rejected_in_clip = len(report.class_detections) - len(verified)
            confirmed_count += len(verified)
            rejected_count += rejected_in_clip

            if keep_rejected:
                new_class_dets = report.class_detections
            else:
                new_class_dets = verified

            # Pick best confirmed detection (or None if all rejected)
            confirmed_dets = [
                d for d in new_class_dets.values() if d.verification_status == "confirmed"
            ]
            error_dets = [d for d in new_class_dets.values() if d.verification_status is None]
            candidates = confirmed_dets + error_dets
            new_best = max(candidates, key=lambda d: d.confidence) if candidates else None

            clip_reports[idx] = ClipReport(
                event_time=report.event_time,
                mp4_filename=report.mp4_filename,
                best_detection=new_best,
                class_detections=new_class_dets,
            )
        print(f"       {confirmed_count} confirmed, {rejected_count} rejected")

        detection_count = sum(1 for r in clip_reports if r.best_detection is not None)
        return clip_reports, detection_count

    def _generate_and_send_report(
        self,
        clip_reports: list[ClipReport],
        date_str: str,
        mode: str,
        title: str,
        send_email: bool,
        label: str,
    ) -> tuple[Path, str | None]:
        """Step 4: Generate HTML report and optionally send email."""
        print("\n[4/4] Generating HTML report...")
        html = generate_report(
            clip_reports,
            crop_padding=self._settings.crop_padding,
            include_video=True,
            title=title,
            mode=mode,
        )

        if mode == "present":
            report_path = Path("output") / f"report-{date_str}.html"
        else:
            report_path = Path("output") / f"report-{mode}-{date_str}.html"
        report_path.parent.mkdir(parents=True, exist_ok=True)
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
                title=title,
                mode=mode,
            )
            email_id = self._email.send_report(email_html, f"{title} — {label}")
            if email_id:
                print(f"       Email sent ({email_id})")
            else:
                print("       Email FAILED")

        return report_path, email_id

    @staticmethod
    def _merge_clip_reports(
        veh_reports: list[ClipReport],
        hsp_reports: list[ClipReport],
    ) -> list[ClipReport]:
        """Merge VEH and HSP reports by mp4_filename.

        Keys in class_detections are naturally disjoint (Motorcycle/Bicycle vs HSP).
        """
        hsp_by_file: dict[str, ClipReport] = {r.mp4_filename: r for r in hsp_reports}
        merged: list[ClipReport] = []
        for veh in veh_reports:
            hsp = hsp_by_file.get(veh.mp4_filename)
            if hsp is None:
                merged.append(veh)
                continue

            combined_dets = {**veh.class_detections, **hsp.class_detections}

            # Pick best detection across both modes
            candidates = list(combined_dets.values())
            confirmed = [d for d in candidates if d.verification_status == "confirmed"]
            unverified = [d for d in candidates if d.verification_status is None]
            best_pool = confirmed + unverified
            best = max(best_pool, key=lambda d: d.confidence) if best_pool else None

            merged.append(
                ClipReport(
                    event_time=veh.event_time,
                    mp4_filename=veh.mp4_filename,
                    best_detection=best,
                    class_detections=combined_dets,
                )
            )
        return merged

    # ── Public run methods ───────────────────────────────────────────

    def run_present(
        self,
        send_email: bool = False,
        target_date: date | None = None,
    ) -> None:
        """Present mode: run both VEH + HSP with Claude verification."""
        start, end, label = self._resolve_daylight_span(target_date, "DETECTION REPORT")

        events = self._fetch_events(start, end)
        if not events:
            print("No events to process.")
            return

        if target_date is not None:
            date_str = target_date.isoformat()
        else:
            date_str = start.astimezone().strftime("%Y-%m-%d")

        clips, total_mb = self._download_clips(events)

        veh_reports, veh_count, veh_frames = self._detect_veh(clips, events)
        hsp_reports, hsp_count, hsp_frames = self._detect_hsp(clips, events)

        merged_reports = self._merge_clip_reports(veh_reports, hsp_reports)

        merged_reports, detection_count = self._verify_with_claude(merged_reports, False)

        report_path, email_id = self._generate_and_send_report(
            merged_reports, date_str, "present", "Detection Report", send_email, label
        )

        print(f"\n{'=' * 60}")
        print("  DETECTION COMPLETE")
        print(f"  Events:     {len(events)}")
        print(f"  Downloaded: {len(clips)} clips ({total_mb:.1f} MB)")
        print(f"  Frames:     {veh_frames} VEH + {hsp_frames} HSP analyzed")
        print(f"  Detections: {detection_count}")
        print(f"  Report:     {report_path.resolve()}")
        if send_email:
            print(f"  Email:      {'sent' if email_id else 'failed'}")
        dump_dir = Path("output") / "video"
        print(f"  Clips:      {dump_dir.resolve()}")
        print(f"{'=' * 60}\n")

    def run_debug_veh(
        self,
        send_email: bool = False,
        target_date: date | None = None,
        use_claude: bool = False,
        claude_keep_rejected: bool = False,
    ) -> None:
        """Debug mode: download and analyze events (VEH detection)."""
        start, end, label = self._resolve_daylight_span(target_date, "VEH DETECTION")

        events = self._fetch_events(start, end)
        if not events:
            print("No events to process.")
            return

        if target_date is not None:
            date_str = target_date.isoformat()
        else:
            date_str = start.astimezone().strftime("%Y-%m-%d")

        clips, total_mb = self._download_clips(events)
        clip_reports, detection_count, total_frames = self._detect_veh(clips, events)

        if use_claude:
            clip_reports, detection_count = self._verify_with_claude(
                clip_reports, claude_keep_rejected
            )

        report_path, email_id = self._generate_and_send_report(
            clip_reports, date_str, "veh", "VEH Detection Report", send_email, label
        )

        print(f"\n{'=' * 60}")
        print("  VEH DETECTION COMPLETE")
        print(f"  Events:     {len(events)}")
        print(f"  Downloaded: {len(clips)} clips ({total_mb:.1f} MB)")
        print(f"  Frames:     {total_frames} analyzed")
        print(f"  Detections: {detection_count}")
        print(f"  Report:     {report_path.resolve()}")
        if send_email:
            print(f"  Email:      {'sent' if email_id else 'failed'}")
        dump_dir = Path("output") / "video"
        print(f"  Clips:      {dump_dir.resolve()}")
        print(f"{'=' * 60}\n")

    def run_debug_hsp(
        self,
        send_email: bool = False,
        target_date: date | None = None,
        use_claude: bool = False,
        claude_keep_rejected: bool = False,
    ) -> None:
        """Debug mode: run with experimental HSP detection."""
        start, end, label = self._resolve_daylight_span(target_date, "HSP DETECTION (experimental)")

        events = self._fetch_events(start, end)
        if not events:
            print("No events to process.")
            return

        if target_date is not None:
            date_str = target_date.isoformat()
        else:
            date_str = start.astimezone().strftime("%Y-%m-%d")

        clips, total_mb = self._download_clips(events)
        clip_reports, detection_count, total_frames = self._detect_hsp(clips, events)

        if use_claude:
            clip_reports, detection_count = self._verify_with_claude(
                clip_reports, claude_keep_rejected
            )

        fps = self._settings.hsp_fps_sample
        report_path, email_id = self._generate_and_send_report(
            clip_reports, date_str, "hsp", "HSP Detection Report", send_email, label
        )

        print(f"\n{'=' * 60}")
        print("  HSP DETECTION COMPLETE")
        print(f"  Events:     {len(events)}")
        print(f"  Downloaded: {len(clips)} clips ({total_mb:.1f} MB)")
        print(f"  Frames:     {total_frames} analyzed (fps={fps})")
        print(f"  Detections: {detection_count}")
        print(f"  Report:     {report_path.resolve()}")
        if send_email:
            print(f"  Email:      {'sent' if email_id else 'failed'}")
        dump_dir = Path("output") / "video"
        print(f"  Clips:      {dump_dir.resolve()}")
        print(f"{'=' * 60}\n")


def main() -> None:
    args = parse_args()
    settings = Settings()  # type: ignore[call-arg]

    target_date: date | None = None
    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()

    if args.debug:
        # Debug mode
        if args.claude and not settings.anthropic_api_key:
            logger.error("--claude requires ANTHROPIC_API_KEY to be set in .env")
            raise SystemExit(1)

        monitor = Monitor(settings)

        if args.veh:
            monitor.run_debug_veh(
                send_email=args.email,
                target_date=target_date,
                use_claude=args.claude,
                claude_keep_rejected=args.claude_keep_rejected,
            )
        elif args.hsp:
            monitor.run_debug_hsp(
                send_email=args.email,
                target_date=target_date,
                use_claude=args.claude,
                claude_keep_rejected=args.claude_keep_rejected,
            )
    else:
        # Present mode (default)
        if not settings.anthropic_api_key:
            logger.error("Present mode requires ANTHROPIC_API_KEY to be set in .env")
            raise SystemExit(1)

        monitor = Monitor(settings)
        monitor.run_present(
            send_email=args.email,
            target_date=target_date,
        )


if __name__ == "__main__":
    main()
