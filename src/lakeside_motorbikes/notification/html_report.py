"""Self-contained HTML debug report for backfill analysis."""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from lakeside_motorbikes.detection.models import Detection
from lakeside_motorbikes.utils.image import crop_to_bbox

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClipReport:
    """Analysis results for a single camera clip."""

    event_time: datetime
    mp4_filename: str
    best_detection: Detection | None
    class_detections: dict[str, Detection]


def _encode_cropped_png(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    padding: float,
) -> str:
    """Crop a detection and return a base64-encoded PNG data URI."""
    cropped = crop_to_bbox(frame, bbox, padding=padding)
    _, png_bytes = cv2.imencode(".png", cropped)
    b64 = base64.b64encode(png_bytes.tobytes()).decode()
    return f"data:image/png;base64,{b64}"


def generate_report(
    clip_reports: list[ClipReport],
    output_dir: Path,
    crop_padding: float = 0.2,
) -> Path:
    """Generate a self-contained HTML debug report.

    Args:
        clip_reports: Analysis results for each clip.
        output_dir: Directory to write the report into (also contains MP4s).
        crop_padding: Padding fraction for cropping detection images.

    Returns:
        Path to the generated report.html file.
    """
    total_clips = len(clip_reports)
    detected_clips = sum(1 for r in clip_reports if r.best_detection is not None)

    sections: list[str] = []
    for report in clip_reports:
        local_time = report.event_time.astimezone()
        time_str = local_time.strftime("%H:%M:%S")
        has_detection = report.best_detection is not None
        border_color = "#22c55e" if has_detection else "#94a3b8"
        bg_color = "#f0fdf4" if has_detection else "#f8fafc"

        # Class detection cards
        cards_html = ""
        if report.class_detections:
            card_items: list[str] = []
            for class_name, det in sorted(
                report.class_detections.items(),
                key=lambda x: x[1].confidence,
                reverse=True,
            ):
                img_uri = _encode_cropped_png(det.frame, det.bbox, crop_padding)
                conf_pct = f"{det.confidence:.0%}"
                card_items.append(
                    f'<div style="border:1px solid #e2e8f0;border-radius:8px;padding:8px;'
                    f'text-align:center;width:160px">'
                    f'<img src="{img_uri}" style="max-width:144px;max-height:144px;'
                    f'border-radius:4px;display:block;margin:0 auto 6px" />'
                    f"<strong>{class_name}</strong><br/>"
                    f'<span style="color:#64748b">{conf_pct}</span>'
                    f"</div>"
                )
            cards_html = (
                '<div style="display:flex;flex-wrap:wrap;gap:10px;margin-top:10px">'
                + "".join(card_items)
                + "</div>"
            )

        # Best detection summary
        best_str = ""
        if report.best_detection:
            best_str = (
                f'<span style="color:#16a34a;font-weight:bold">'
                f"{report.best_detection.class_name} "
                f"({report.best_detection.confidence:.0%})</span>"
            )
        else:
            best_str = '<span style="color:#94a3b8">No detection above threshold</span>'

        sections.append(
            f'<div style="border-left:4px solid {border_color};background:{bg_color};'
            f'padding:12px 16px;margin-bottom:16px;border-radius:0 8px 8px 0">'
            f'<h3 style="margin:0 0 8px">{time_str} &mdash; '
            f'<code style="font-size:0.85em">{report.mp4_filename}</code></h3>'
            f'<p style="margin:4px 0">{best_str}</p>'
            f'<video controls preload="metadata" style="max-width:100%;max-height:300px;'
            f'border-radius:6px;margin-top:6px">'
            f'<source src="{report.mp4_filename}" type="video/mp4" />'
            f"</video>"
            f"{cards_html}"
            f"</div>"
        )

    html = (
        "<!DOCTYPE html>"
        '<html lang="en"><head><meta charset="utf-8"/>'
        "<title>Backfill Debug Report</title>"
        '<meta name="viewport" content="width=device-width,initial-scale=1"/>'
        "<style>"
        "body{font-family:system-ui,sans-serif;max-width:900px;"
        "margin:0 auto;padding:20px;background:#fff}"
        "h1{margin-bottom:4px} .stats{color:#64748b;margin-bottom:24px}"
        "</style></head><body>"
        "<h1>Backfill Debug Report</h1>"
        f'<p class="stats">{total_clips} clips analysed &middot; '
        f"{detected_clips} with detections</p>" + "".join(sections) + "</body></html>"
    )

    output_path = output_dir / "report.html"
    output_path.write_text(html)
    logger.info("HTML report written to %s", output_path)
    return output_path
