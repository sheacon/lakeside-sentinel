from __future__ import annotations

import logging
import time
import webbrowser
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from ultralytics import YOLO

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
from lakeside_sentinel.review.staging import discover_unreviewed, load_staged_detections
from lakeside_sentinel.utils.daylight import (
    get_daylight_span,
    get_daylight_span_for_date,
    is_daylight,
)
from lakeside_sentinel.utils.image import crop_to_roi
from lakeside_sentinel.utils.video import extract_frames

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Configure root logger explicitly — logging.basicConfig() is a no-op when
# a library (glocaltokens) has already added handlers during import.
_root = logging.getLogger()
_root.setLevel(logging.INFO)
if not any(
    isinstance(h, logging.StreamHandler) and h.stream.name == "<stderr>" for h in _root.handlers
):
    _console = logging.StreamHandler()
    _console.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
    _root.addHandler(_console)
else:
    for h in _root.handlers:
        if isinstance(h, logging.StreamHandler):
            h.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))

logger = logging.getLogger(__name__)
_CLEANUP_MAX_AGE_DAYS = 14
_EXPIRY_WARNING_DAYS = 3
_SENSITIVE_KEYWORDS = {"token", "key", "password", "secret"}


def _setup_file_logging() -> None:
    """Add a FileHandler to the root logger writing to output/logs/."""
    log_dir = Path("output") / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = log_dir / f"{timestamp}.log"
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
    logging.getLogger().addHandler(handler)
    logger.info("Log file: %s", log_path.resolve())


def _cleanup_old_files(directory: Path, suffix: str, max_age_days: int) -> None:
    """Delete files in directory matching suffix that are older than max_age_days."""
    if not directory.exists():
        return
    cutoff = time.time() - (max_age_days * 86400)
    for filepath in directory.iterdir():
        if filepath.suffix == suffix and filepath.stat().st_mtime < cutoff:
            filepath.unlink()
            logger.info("Cleaned up old file: %s", filepath)


def _cleanup_old_dirs(directory: Path, max_age_days: int) -> None:
    """Delete subdirectories in directory that are older than max_age_days."""
    if not directory.exists():
        return
    cutoff = time.time() - (max_age_days * 86400)
    import shutil

    for dirpath in directory.iterdir():
        if dirpath.is_dir() and dirpath.stat().st_mtime < cutoff:
            shutil.rmtree(dirpath)
            logger.info("Cleaned up old directory: %s", dirpath)


def _warn_expiring_staging(
    staging_dir: Path,
    max_age_days: int,
    warning_days: int,
    email_sender: EmailSender,
) -> None:
    """Send a warning email when unreviewed staging data is close to auto-deletion."""
    if not staging_dir.exists():
        return

    unreviewed = discover_unreviewed()
    if not unreviewed:
        return

    now = time.time()
    expiring: list[tuple[Path, int, int]] = []  # (dir, detection_count, days_remaining)

    for d in unreviewed:
        age_seconds = now - d.stat().st_mtime
        age_days = int(age_seconds / 86400)
        days_remaining = max_age_days - age_days
        if days_remaining <= warning_days:
            try:
                data = load_staged_detections(d)
                det_count = len(data.get("detections", []))
            except Exception:
                logger.warning("Failed to read staging data from %s", d)
                det_count = 0
            expiring.append((d, det_count, days_remaining))

    if not expiring:
        return

    rows = ""
    for d, det_count, days_rem in expiring:
        date_str = d.name
        rows += (
            f"<tr>"
            f"<td style='padding:4px 12px'>{date_str}</td>"
            f"<td style='padding:4px 12px'>{det_count}</td>"
            f"<td style='padding:4px 12px'>{days_rem}</td>"
            f"</tr>"
        )

    html = (
        "<html><body>"
        "<h2>Unreviewed staging data expiring soon</h2>"
        "<p>The following staged detections will be auto-deleted if not reviewed:</p>"
        "<table border='1' cellpadding='4' cellspacing='0'>"
        "<tr><th>Date</th><th>Detections</th><th>Days remaining</th></tr>"
        f"{rows}"
        "</table>"
        "<p>Run <code>python -m lakeside_sentinel --review</code> to review.</p>"
        "</body></html>"
    )

    dates_str = ", ".join(d.name for d, _, _ in expiring)
    subject = f"Lakeside Sentinel: staging data expiring soon ({dates_str})"
    email_sender.send_report(html, subject)
    logger.info("Sent expiry warning email for %d staging dir(s)", len(expiring))


def _print_settings(settings: Settings) -> None:
    """Log all settings, masking sensitive values."""
    for name, value in settings.model_dump().items():
        if any(kw in name for kw in _SENSITIVE_KEYWORDS):
            display = "****"
        else:
            display = value
        label = name.replace("_", " ").title()
        logger.info("  %s %s", f"{label + ':':<28}", display)


def _dates_needing_analysis(
    max_age_days: int,
    latitude: float,
    longitude: float,
) -> list[date]:
    """Return dates within the last max_age_days that have no staging dir and no report file."""
    today = date.today()
    dates: list[date] = []
    for days_ago in range(max_age_days):
        d = today - timedelta(days=days_ago)
        date_str = d.isoformat()

        staging_dir = Path("output") / "staging" / date_str
        report_file = Path("output") / f"report-{date_str}.html"

        if staging_dir.exists() or report_file.exists():
            continue
        dates.append(d)

    # Return in chronological order (oldest first)
    dates.reverse()
    return dates


class Monitor:
    """Orchestrates the detection pipeline."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._auth = NestAuth(settings.google_master_token, settings.google_username)
        self._api = NestCameraAPI(self._auth, settings.nest_device_id)
        # Load YOLO weights once and share the instance between VEH and HSP.
        # The two detectors only differ in post-processing (which COCO classes
        # they care about), so a second load would just duplicate the weights
        # in RAM (and on the unified-memory MPS device).
        yolo_model = YOLO(settings.yolo_model)
        self._veh_detector = VEHDetector(
            model=yolo_model,
            confidence_threshold=settings.veh_confidence_threshold,
            batch_size=settings.yolo_batch_size,
        )
        self._hsp_detector = HSPDetector(
            model=yolo_model,
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
        """Resolve daylight start/end and log the banner.

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

        logger.info("")
        logger.info("=" * 60)
        logger.info("  %s", label)
        logger.info("  From: %s", start.astimezone().strftime("%d %b %Y %H:%M:%S %Z"))
        logger.info("  To:   %s", end.astimezone().strftime("%d %b %Y %H:%M:%S %Z"))
        logger.info("")
        _print_settings(self._settings)
        logger.info("=" * 60)
        logger.info("")

        return start, end, label

    def _fetch_events(
        self,
        start: datetime,
        end: datetime,
        step_label: str = "[1/4]",
    ) -> list[CameraEvent]:
        """Fetch events from Nest API and filter to daylight."""
        t0 = time.monotonic()
        now_str = datetime.now().strftime("%H:%M:%S")
        logger.info("%s Fetching event list from Nest API... (%s)", step_label, now_str)
        events = self._api.get_events(start, end)
        total_events = len(events)
        events = self._filter_daylight(events)
        filtered = total_events - len(events)
        if filtered:
            logger.info("       Found %d events (%d nighttime filtered)", total_events, filtered)
        else:
            logger.info("       Found %d events", total_events)
        elapsed = time.monotonic() - t0
        logger.info("       Done in %.1fs", elapsed)
        logger.info("")
        return events

    def _download_clips(
        self,
        events: list[CameraEvent],
        step_label: str = "[2/4]",
    ) -> tuple[list[tuple[int, Path]], float]:
        """Download clips (with caching).

        Returns (clips, total_mb), where each clip entry is (event_index, mp4_path).
        Clip bytes are written straight to disk and never held in memory beyond
        the brief window between download and write.
        """
        t0 = time.monotonic()
        dump_dir = Path("output") / "video"
        dump_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[clips] Saving to: %s", dump_dir.resolve())
        logger.info("")

        now_str = datetime.now().strftime("%H:%M:%S")
        logger.info("%s Downloading clips... (%s)", step_label, now_str)
        clips: list[tuple[int, Path]] = []
        download_errors = 0
        total_bytes = 0
        cached_count = 0
        for i, event in enumerate(events):
            local_time = event.start_time.astimezone()
            label = local_time.strftime("%m-%d %H:%M:%S")

            filename = local_time.strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"
            filepath = dump_dir / filename
            if filepath.exists():
                size_bytes = filepath.stat().st_size
                total_bytes += size_bytes
                clips.append((i, filepath))
                cached_count += 1
                size_mb = size_bytes / 1_000_000
                logger.info(
                    "  [%3d/%d] %s — %.1f MB (cached)",
                    i + 1,
                    len(events),
                    label,
                    size_mb,
                )
                continue

            try:
                mp4_bytes = self._api.download_clip(event)
                if not mp4_bytes:
                    logger.info("  [%3d/%d] %s — empty clip (skipped)", i + 1, len(events), label)
                    download_errors += 1
                    continue
                size_bytes = len(mp4_bytes)
                total_bytes += size_bytes
                size_mb = size_bytes / 1_000_000
                dur = event.duration.total_seconds()
                logger.info(
                    "  [%3d/%d] %s — %.1f MB (%.0fs)",
                    i + 1,
                    len(events),
                    label,
                    size_mb,
                    dur,
                )

                filepath.write_bytes(mp4_bytes)
                # Drop the bytes reference now that they're persisted; the detectors
                # will read the file back per-clip via extract_frames.
                del mp4_bytes
                clips.append((i, filepath))
            except Exception as e:
                logger.info("  [%3d/%d] %s — ERROR: %s", i + 1, len(events), label, e)
                download_errors += 1

        total_mb = total_bytes / 1_000_000
        downloaded = len(clips) - cached_count
        logger.info(
            "       %d downloaded, %d cached — %d/%d clips (%.1f MB total)",
            downloaded,
            cached_count,
            len(clips),
            len(events),
            total_mb,
        )
        if download_errors:
            logger.info("       %d download errors", download_errors)
        elapsed = time.monotonic() - t0
        logger.info("       Done in %.1fs", elapsed)
        logger.info("")

        return clips, total_mb

    def _detect_veh(
        self,
        clips: list[tuple[int, Path]],
        events: list[CameraEvent],
        step_label: str = "[3/4]",
    ) -> tuple[list[ClipReport], int, int, list[ClipReport]]:
        """VEH: YOLO detection.

        Returns (clip_reports, detection_count, total_frames, debug_clip_reports).
        debug_clip_reports contains all classes (no threshold filter).
        """
        t0 = time.monotonic()
        now_str = datetime.now().strftime("%H:%M:%S")
        logger.info("%s Analyzing frames for vehicles... (%s)", step_label, now_str)
        detection_count = 0
        total_frames = 0
        clip_reports: list[ClipReport] = []
        debug_clip_reports: list[ClipReport] = []
        for idx, (event_i, mp4_path) in enumerate(clips):
            event = events[event_i]
            local_time = event.start_time.astimezone()
            label = local_time.strftime("%m-%d %H:%M:%S")

            frames = extract_frames(mp4_path, fps_sample=self._settings.veh_fps_sample)
            frames = crop_to_roi(
                frames,
                y_start=self._settings.roi_y_start,
                y_end=self._settings.roi_y_end,
                x_start=self._settings.roi_x_start,
                x_end=self._settings.roi_x_end,
            )
            total_frames += len(frames)
            if not frames:
                logger.info("  [%3d/%d] %s — no frames extracted", idx + 1, len(clips), label)
                mp4_fn = "video/" + local_time.strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"
                empty_report = ClipReport(
                    event_time=event.start_time,
                    mp4_filename=mp4_fn,
                    best_detection=None,
                    class_detections={},
                )
                clip_reports.append(empty_report)
                debug_clip_reports.append(empty_report)
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
            debug_clip_reports.append(
                ClipReport(
                    event_time=event.start_time,
                    mp4_filename=mp4_fn,
                    best_detection=detection,
                    class_detections=dict(class_best),
                )
            )

            if detection is None:
                logger.info(
                    "  [%3d/%d] %s — %2d frames — no detection",
                    idx + 1,
                    len(clips),
                    label,
                    len(frames),
                )
                if class_best:
                    breakdown = "  ".join(
                        f"{name}: {det.confidence:.0%}" for name, det in sorted(class_best.items())
                    )
                    logger.info("            %s", breakdown)
                continue

            detection_count += 1
            logger.info(
                "  [%3d/%d] %s — %2d frames — %s (confidence: %s)",
                idx + 1,
                len(clips),
                label,
                len(frames),
                detection.class_name.upper(),
                f"{detection.confidence:.0%}",
            )
            breakdown = "  ".join(
                f"{name}: {det.confidence:.0%}" for name, det in sorted(class_best.items())
            )
            logger.info("            %s", breakdown)

        elapsed = time.monotonic() - t0
        logger.info("       Done in %.1fs", elapsed)

        return clip_reports, detection_count, total_frames, debug_clip_reports

    def _detect_hsp(
        self,
        clips: list[tuple[int, Path]],
        events: list[CameraEvent],
        step_label: str = "[3/4]",
    ) -> tuple[list[ClipReport], int, int, list[ClipReport]]:
        """HSP: Person tracking.

        Returns (clip_reports, detection_count, total_frames, debug_clip_reports).
        debug_clip_reports contains the fastest track regardless of threshold.
        """
        t0 = time.monotonic()
        fps = self._settings.hsp_fps_sample
        now_str = datetime.now().strftime("%H:%M:%S")
        logger.info(
            "%s Analyzing frames for high-speed persons (fps=%d)... (%s)",
            step_label,
            fps,
            now_str,
        )
        detection_count = 0
        total_frames = 0
        clip_reports: list[ClipReport] = []
        debug_clip_reports: list[ClipReport] = []
        for idx, (event_i, mp4_path) in enumerate(clips):
            event = events[event_i]
            local_time = event.start_time.astimezone()
            label = local_time.strftime("%m-%d %H:%M:%S")

            frames = extract_frames(mp4_path, fps_sample=fps)
            frames = crop_to_roi(
                frames,
                y_start=self._settings.roi_y_start,
                y_end=self._settings.roi_y_end,
                x_start=self._settings.roi_x_start,
                x_end=self._settings.roi_x_end,
            )
            total_frames += len(frames)
            if not frames:
                logger.info("  [%3d/%d] %s — no frames extracted", idx + 1, len(clips), label)
                mp4_fn = "video/" + local_time.strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"
                empty_report = ClipReport(
                    event_time=event.start_time,
                    mp4_filename=mp4_fn,
                    best_detection=None,
                    class_detections={},
                )
                clip_reports.append(empty_report)
                debug_clip_reports.append(empty_report)
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

            # Build debug report with fastest track (regardless of threshold)
            debug_class_dets: dict[str, Detection] = {}
            debug_best: Detection | None = None
            multi_point_tracks = [t for t in tracks if len(t.points) >= 2]
            if multi_point_tracks:
                fastest = max(multi_point_tracks, key=lambda t: t.displacement_per_second(fps))
                best_pt = fastest.best_point
                debug_det = Detection(
                    frame=best_pt.frame,
                    bbox=best_pt.bbox,
                    confidence=best_pt.confidence,
                    class_name="HSP",
                    speed=fastest.displacement_per_second(fps),
                )
                debug_class_dets["HSP"] = debug_det
                debug_best = debug_det
            debug_clip_reports.append(
                ClipReport(
                    event_time=event.start_time,
                    mp4_filename=mp4_fn,
                    best_detection=debug_best,
                    class_detections=debug_class_dets,
                )
            )

            track_info = ""
            if tracks:
                displacements = [t.displacement_per_second(fps) for t in tracks]
                max_disp = max(displacements)
                track_info = f" — {len(tracks)} tracks (max disp: {max_disp:.1f} px/sec)"

            if detection is None:
                logger.info(
                    "  [%3d/%d] %s — %2d frames — no HSP%s",
                    idx + 1,
                    len(clips),
                    label,
                    len(frames),
                    track_info,
                )
                continue

            detection_count += 1
            logger.info(
                "  [%3d/%d] %s — %2d frames — HSP (confidence: %s)%s",
                idx + 1,
                len(clips),
                label,
                len(frames),
                f"{detection.confidence:.0%}",
                track_info,
            )

        elapsed = time.monotonic() - t0
        logger.info("       Done in %.1fs", elapsed)

        return clip_reports, detection_count, total_frames, debug_clip_reports

    def _verify_with_claude(
        self,
        clip_reports: list[ClipReport],
        keep_rejected: bool,
        step_label: str = "[4/5]",
    ) -> tuple[list[ClipReport], int]:
        """Claude verification.

        Returns (updated_clip_reports, detection_count).
        """
        t0 = time.monotonic()
        now_str = datetime.now().strftime("%H:%M:%S")
        logger.info("")
        logger.info("%s Verifying detections with Claude Vision... (%s)", step_label, now_str)
        verifier = ClaudeVerifier(
            api_key=self._settings.anthropic_api_key,
            model=self._settings.claude_vision_model,
            crop_padding=self._settings.crop_padding,
            prompt=self._settings.claude_vision_prompt,
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
        logger.info("       %d confirmed, %d rejected", confirmed_count, rejected_count)
        elapsed = time.monotonic() - t0
        logger.info("       Done in %.1fs", elapsed)

        detection_count = sum(1 for r in clip_reports if r.best_detection is not None)
        return clip_reports, detection_count

    def _generate_and_send_report(
        self,
        clip_reports: list[ClipReport],
        date_str: str,
        mode: str,
        title: str,
        send_email: bool,
        subtitle: str | None = None,
        step_label: str = "[4/4]",
        open_browser: bool = True,
        total_clips: int | None = None,
    ) -> tuple[Path, str | None]:
        """Generate HTML report and optionally send email."""
        t0 = time.monotonic()
        now_str = datetime.now().strftime("%H:%M:%S")
        logger.info("")
        logger.info("%s Generating HTML report... (%s)", step_label, now_str)
        settings_dict = self._settings.model_dump() if mode != "default" else None
        html, _ = generate_report(
            clip_reports,
            crop_padding=self._settings.crop_padding,
            include_video=True,
            title=title,
            mode=mode,
            settings=settings_dict,
            subtitle=subtitle,
            total_clips=total_clips,
        )

        if mode == "default":
            report_path = Path("output") / f"report-{date_str}.html"
        else:
            report_path = Path("output") / f"report-{mode}-{date_str}.html"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(html)
        logger.info("HTML report written to %s", report_path)
        logger.info("       Report: %s", report_path.resolve())
        if open_browser:
            webbrowser.open(report_path.resolve().as_uri())

        email_id: str | None = None
        if send_email:
            email_html, email_attachments = generate_report(
                clip_reports,
                crop_padding=self._settings.crop_padding,
                include_video=False,
                title=title,
                mode=mode,
                settings=settings_dict,
                subtitle=subtitle,
                for_email=True,
                total_clips=total_clips,
            )
            email_id = self._email.send_report(
                email_html, f"{title} - {date_str}", attachments=email_attachments
            )
            if email_id:
                logger.info("       Email sent (%s)", email_id)
            else:
                logger.info("       Email FAILED")

        elapsed = time.monotonic() - t0
        logger.info("       Done in %.1fs", elapsed)

        return report_path, email_id

    def _write_debug_report(
        self,
        clip_reports: list[ClipReport],
        date_str: str,
        mode: str,
        title: str,
    ) -> Path:
        """Generate an HTML debug report and write to disk (no browser, no email)."""
        settings_dict = self._settings.model_dump()
        html, _ = generate_report(
            clip_reports,
            crop_padding=self._settings.crop_padding,
            include_video=True,
            title=title,
            mode=mode,
            settings=settings_dict,
        )
        report_path = Path("output") / f"report-{mode}-{date_str}.html"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(html)
        logger.info("Debug %s report written to %s", mode.upper(), report_path)
        return report_path

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

    # ── Detection pipeline ────────────────────────────────────────────

    def _run_detection_pipeline(
        self,
        target_date: date,
    ) -> (
        tuple[
            list[ClipReport],
            list[ClipReport],
            list[ClipReport],
            list[CameraEvent],
            list[tuple[int, Path]],
            float,
            int,
            int,
        ]
        | None
    ):
        """Run the full detection pipeline for a date.

        Returns:
            Tuple of (merged_reports, veh_debug_reports, hsp_debug_reports,
                      events, clips, total_mb, veh_frames, hsp_frames)
            or None if no events found.
        """
        start, end = get_daylight_span_for_date(
            target_date,
            self._settings.camera_latitude,
            self._settings.camera_longitude,
        )
        date_str = target_date.isoformat()

        logger.info("")
        logger.info("=" * 60)
        logger.info("  DETECTION PIPELINE — %s", date_str)
        logger.info("  From: %s", start.astimezone().strftime("%d %b %Y %H:%M:%S %Z"))
        logger.info("  To:   %s", end.astimezone().strftime("%d %b %Y %H:%M:%S %Z"))
        logger.info("=" * 60)
        logger.info("")

        events = self._fetch_events(start, end, step_label="[1/5]")
        if not events:
            logger.info("No events for %s.", date_str)
            return None

        clips, total_mb = self._download_clips(events, step_label="[2/5]")

        veh_reports, veh_count, veh_frames, veh_debug_reports = self._detect_veh(
            clips, events, step_label="[3/5]"
        )
        hsp_reports, hsp_count, hsp_frames, hsp_debug_reports = self._detect_hsp(
            clips, events, step_label="[3/5]"
        )

        merged_reports = self._merge_clip_reports(veh_reports, hsp_reports)

        merged_reports, detection_count = self._verify_with_claude(
            merged_reports, False, step_label="[4/5]"
        )

        logger.info("")
        logger.info("  Pipeline complete for %s: %d detections", date_str, detection_count)

        return (
            merged_reports,
            veh_debug_reports,
            hsp_debug_reports,
            events,
            clips,
            total_mb,
            veh_frames,
            hsp_frames,
        )

    # ── Public run methods ───────────────────────────────────────────

    def run(
        self,
        target_date: date | None = None,
    ) -> None:
        """Default mode: run detection pipeline and stage results for review."""
        from lakeside_sentinel.review.staging import stage_detections

        # Determine which dates need analysis
        if target_date is not None:
            dates_to_analyze = [target_date]
        else:
            dates_to_analyze = _dates_needing_analysis(
                _CLEANUP_MAX_AGE_DAYS,
                self._settings.camera_latitude,
                self._settings.camera_longitude,
            )

        if dates_to_analyze:
            logger.info("Dates needing analysis: %s", [d.isoformat() for d in dates_to_analyze])
        else:
            logger.info("No new dates to analyze.")
            return

        staged_count = 0
        skipped_count = 0

        for d in dates_to_analyze:
            date_str = d.isoformat()
            staging_dir = Path("output") / "staging" / date_str
            if staging_dir.exists():
                logger.info("Staging data already exists for %s, skipping.", date_str)
                skipped_count += 1
                continue

            result = self._run_detection_pipeline(d)
            if result is None:
                continue

            merged, veh_debug, hsp_debug, events, clips, total_mb, veh_frames, hsp_frames = result
            stage_detections(
                date_str,
                merged,
                veh_debug,
                hsp_debug,
                self._settings.crop_padding,
                total_clips=len(clips),
            )
            staged_count += 1

        logger.info("")
        logger.info("=" * 60)
        logger.info("  DETECTION COMPLETE")
        logger.info("  Dates processed: %d", staged_count)
        logger.info("  Dates skipped:   %d", skipped_count)
        logger.info("=" * 60)
        logger.info("")

    def run_review(
        self,
        review_port: int = 5000,
    ) -> None:
        """Review mode: launch web app for already-staged data."""
        from lakeside_sentinel.review.fine_tuning import (
            ensure_data_yaml,
            save_annotation,
            save_other,
        )
        from lakeside_sentinel.review.server import run_review_server
        from lakeside_sentinel.review.staging import (
            cleanup_staging,
            discover_unreviewed,
            load_frame,
            load_staged_detections,
            rebuild_clip_reports,
        )

        # Check if there's anything to review
        unreviewed = discover_unreviewed()
        if not unreviewed:
            logger.info("No staged data to review.")
            return

        logger.info("Launching review server with %d day(s) queued.", len(unreviewed))

        # Launch review server (blocks until submit or exit)
        result = run_review_server(port=review_port)

        if result is None:
            logger.info("Review deferred — staged data preserved.")
            return

        # Process submit result
        days_data = result.get("days", {})
        logger.info("Processing submit for %d day(s).", len(days_data))

        fine_tuning_dir = Path("output") / "fine-tuning"
        has_annotations = False

        all_present_htmls: list[str] = []
        all_attachments: list[dict[str, str | bytes]] = []
        cid_offset = 0

        for date_str, day_info in sorted(days_data.items()):
            selected_ids = set(day_info.get("selected", []))
            classifications = day_info.get("classifications", {})

            staging_dir = Path("output") / "staging" / date_str
            if not staging_dir.exists():
                logger.warning("Staging dir missing for %s, skipping.", date_str)
                continue

            staging_data = load_staged_detections(staging_dir)
            stored_total_clips = staging_data.get("total_clips")

            # Save fine-tuning annotations
            for det_dict in staging_data["detections"]:
                det_id = det_dict["id"]
                class_label = classifications.get(det_id)
                if not class_label:
                    continue

                frame = load_frame(staging_dir, det_dict["frame_filename"])
                bbox = tuple(det_dict["bbox"])
                image_id = f"{date_str}_{det_id}"

                if class_label == "other":
                    save_other(frame, bbox, image_id, fine_tuning_dir)
                else:
                    save_annotation(frame, bbox, class_label, image_id, fine_tuning_dir)
                    has_annotations = True

            # Rebuild clip reports for selected detections
            clip_reports = rebuild_clip_reports(staging_dir, selected_ids)

            # Generate present report for this day
            report_path, _ = self._generate_and_send_report(
                clip_reports,
                date_str,
                "default",
                "Motorized Vehicle Detection Report",
                False,
                subtitle=date_str,
                step_label="[5/5]",
                open_browser=True,
                total_clips=stored_total_clips,
            )

            # Generate debug reports
            # Rebuild full VEH/HSP debug reports from staging
            veh_ids = {d["id"] for d in staging_data["detections"] if d["source"] == "veh"}
            hsp_ids = {d["id"] for d in staging_data["detections"] if d["source"] == "hsp"}
            veh_debug_reports = rebuild_clip_reports(staging_dir, veh_ids)
            hsp_debug_reports = rebuild_clip_reports(staging_dir, hsp_ids)

            self._write_debug_report(veh_debug_reports, date_str, "veh", "VEH Detection Report")
            self._write_debug_report(hsp_debug_reports, date_str, "hsp", "HSP Detection Report")

            # Collect email HTML for this day
            email_html, email_attachments = generate_report(
                clip_reports,
                crop_padding=self._settings.crop_padding,
                include_video=False,
                title=f"Detection Report — {date_str}",
                mode="default",
                subtitle=date_str,
                for_email=True,
                total_clips=stored_total_clips,
                cid_start=cid_offset,
            )
            all_present_htmls.append(email_html)
            all_attachments.extend(email_attachments)
            cid_offset += len(email_attachments)

            # Clean up staging (videos age out via the 14-day cleanup at startup,
            # so the HTML report's video player keeps working for a while after review)
            cleanup_staging(staging_dir)

        # Send one combined email
        if all_present_htmls:
            combined_html = "<br/><hr/><br/>".join(all_present_htmls)
            date_range = sorted(days_data.keys())
            if len(date_range) == 1:
                subject = f"Motorized Vehicle Detection Report - {date_range[0]}"
            else:
                subject = (
                    f"Motorized Vehicle Detection Report - {date_range[0]} to {date_range[-1]}"
                )
            email_id = self._email.send_report(combined_html, subject, attachments=all_attachments)
            if email_id:
                logger.info("Combined email sent (%s)", email_id)
            else:
                logger.info("Combined email FAILED")

        if has_annotations:
            ensure_data_yaml(fine_tuning_dir)

        logger.info("Review complete.")

    def run_debug_veh(
        self,
        target_date: date | None = None,
        use_claude: bool = False,
        claude_keep_rejected: bool = False,
    ) -> None:
        """Debug mode: download and analyze events (VEH detection)."""
        start, end, label = self._resolve_daylight_span(target_date, "VEH DETECTION")

        if target_date is not None:
            date_str = target_date.isoformat()
        else:
            date_str = start.astimezone().strftime("%Y-%m-%d")

        total = 5 if use_claude else 4
        events = self._fetch_events(start, end, step_label=f"[1/{total}]")
        if not events:
            logger.info("No events to process.")
            return

        clips, total_mb = self._download_clips(events, step_label=f"[2/{total}]")
        clip_reports, detection_count, total_frames, _ = self._detect_veh(
            clips, events, step_label=f"[3/{total}]"
        )

        if use_claude:
            clip_reports, detection_count = self._verify_with_claude(
                clip_reports, claude_keep_rejected, step_label=f"[4/{total}]"
            )

        report_path, email_id = self._generate_and_send_report(
            clip_reports,
            date_str,
            "veh",
            "VEH Detection Report",
            False,
            step_label=f"[{total}/{total}]",
        )

        logger.info("")
        logger.info("=" * 60)
        logger.info("  VEH DETECTION COMPLETE")
        logger.info("  Events:     %d", len(events))
        logger.info("  Downloaded: %d clips (%.1f MB)", len(clips), total_mb)
        logger.info("  Frames:     %d analyzed", total_frames)
        logger.info("  Detections: %d", detection_count)
        logger.info("  Report:     %s", report_path.resolve())
        dump_dir = Path("output") / "video"
        logger.info("  Clips:      %s", dump_dir.resolve())
        logger.info("=" * 60)
        logger.info("")

    def run_debug_hsp(
        self,
        target_date: date | None = None,
        use_claude: bool = False,
        claude_keep_rejected: bool = False,
    ) -> None:
        """Debug mode: run with experimental HSP detection."""
        start, end, label = self._resolve_daylight_span(target_date, "HSP DETECTION (experimental)")

        if target_date is not None:
            date_str = target_date.isoformat()
        else:
            date_str = start.astimezone().strftime("%Y-%m-%d")

        total = 5 if use_claude else 4
        events = self._fetch_events(start, end, step_label=f"[1/{total}]")
        if not events:
            logger.info("No events to process.")
            return

        clips, total_mb = self._download_clips(events, step_label=f"[2/{total}]")
        clip_reports, detection_count, total_frames, _ = self._detect_hsp(
            clips, events, step_label=f"[3/{total}]"
        )

        if use_claude:
            clip_reports, detection_count = self._verify_with_claude(
                clip_reports, claude_keep_rejected, step_label=f"[4/{total}]"
            )

        fps = self._settings.hsp_fps_sample
        report_path, email_id = self._generate_and_send_report(
            clip_reports,
            date_str,
            "hsp",
            "HSP Detection Report",
            False,
            step_label=f"[{total}/{total}]",
        )

        logger.info("")
        logger.info("=" * 60)
        logger.info("  HSP DETECTION COMPLETE")
        logger.info("  Events:     %d", len(events))
        logger.info("  Downloaded: %d clips (%.1f MB)", len(clips), total_mb)
        logger.info("  Frames:     %d analyzed (fps=%d)", total_frames, fps)
        logger.info("  Detections: %d", detection_count)
        logger.info("  Report:     %s", report_path.resolve())
        dump_dir = Path("output") / "video"
        logger.info("  Clips:      %s", dump_dir.resolve())
        logger.info("=" * 60)
        logger.info("")


def main() -> None:
    args = parse_args()
    settings = Settings()  # type: ignore[call-arg]

    _setup_file_logging()
    _cleanup_old_files(Path("output") / "logs", ".log", _CLEANUP_MAX_AGE_DAYS)
    _cleanup_old_files(Path("output") / "video", ".mp4", _CLEANUP_MAX_AGE_DAYS)
    _warn_expiring_staging(
        Path("output") / "staging",
        _CLEANUP_MAX_AGE_DAYS,
        _EXPIRY_WARNING_DAYS,
        EmailSender(
            api_key=settings.resend_api_key,
            from_email=settings.alert_email_from,
            to_email=settings.alert_email_to,
        ),
    )
    _cleanup_old_dirs(Path("output") / "staging", _CLEANUP_MAX_AGE_DAYS)

    target_date: date | None = None
    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.veh or args.hsp:
        # Single detector mode
        if args.claude and not settings.anthropic_api_key:
            logger.error("--claude requires ANTHROPIC_API_KEY to be set in .env")
            raise SystemExit(1)

        monitor = Monitor(settings)

        if args.veh:
            monitor.run_debug_veh(
                target_date=target_date,
                use_claude=args.claude,
                claude_keep_rejected=args.claude_keep_rejected,
            )
        elif args.hsp:
            monitor.run_debug_hsp(
                target_date=target_date,
                use_claude=args.claude,
                claude_keep_rejected=args.claude_keep_rejected,
            )
    elif args.review:
        # Review mode — launches web app for already-staged data
        monitor = Monitor(settings)
        monitor.run_review(
            review_port=settings.review_port,
        )
    else:
        # Default mode — process + stage
        if not settings.anthropic_api_key:
            logger.error("Default mode requires ANTHROPIC_API_KEY to be set in .env")
            raise SystemExit(1)

        monitor = Monitor(settings)
        monitor.run(
            target_date=target_date,
        )


if __name__ == "__main__":
    main()
