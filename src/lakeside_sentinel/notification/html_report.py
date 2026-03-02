"""Self-contained HTML report for detection analysis."""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

from lakeside_sentinel.detection.models import Detection
from lakeside_sentinel.utils.image import crop_to_bbox

logger = logging.getLogger(__name__)

_PRESENT_LABEL = "Potential Motorized Vehicle"
_SENSITIVE_KEYWORDS = {"token", "key", "password", "secret"}


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


def _render_settings_html(settings: dict[str, object]) -> str:
    """Render settings as an HTML details/summary block, masking sensitive values."""
    rows: list[str] = []
    for name, value in settings.items():
        if any(kw in name for kw in _SENSITIVE_KEYWORDS):
            display = "****"
        else:
            display = str(value)
        label = name.replace("_", " ").title()
        rows.append(
            f"<tr><td style='padding:2px 12px 2px 0;color:#64748b'>{label}</td>"
            f"<td style='padding:2px 0'><code>{display}</code></td></tr>"
        )
    return (
        '<details style="margin-bottom:20px;color:#64748b;font-size:0.85em">'
        "<summary style='cursor:pointer;font-weight:600'>Parameters</summary>"
        '<table style="margin-top:8px;border-collapse:collapse">'
        + "".join(rows)
        + "</table></details>"
    )


def generate_report(
    clip_reports: list[ClipReport],
    crop_padding: float = 0.2,
    include_video: bool = True,
    title: str = "Detection Report",
    mode: str = "veh",
    settings: dict[str, object] | None = None,
) -> str:
    """Generate a self-contained HTML report.

    Args:
        clip_reports: Analysis results for each clip.
        crop_padding: Padding fraction for cropping detection images.
        include_video: Whether to include video player elements.
        title: Title for the HTML report.
        mode: Detection mode ("veh", "hsp", or "present"). Controls sorting and display.
        settings: Optional settings dict to display in debug mode reports.

    Returns:
        The generated HTML string.
    """
    is_present = mode == "present"

    # Filter out clips with no above-threshold detections
    filtered_reports = [r for r in clip_reports if r.class_detections]

    if mode == "hsp":
        # Sort by HSP speed (highest first)
        def _hsp_sort_key(r: ClipReport) -> float:
            hsp = r.class_detections.get("HSP")
            return hsp.speed if hsp is not None and hsp.speed is not None else 0.0

        filtered_reports.sort(key=_hsp_sort_key, reverse=True)
        sort_label = "sorted by speed"
    elif is_present:
        # Sort chronologically by event_time
        filtered_reports.sort(key=lambda r: r.event_time)
        sort_label = "sorted chronologically"
    else:
        # Sort by motorcycle confidence (highest first), then bicycle confidence
        def _veh_sort_key(r: ClipReport) -> tuple[float, float]:
            moto = r.class_detections.get("Motorcycle")
            bike = r.class_detections.get("Bicycle")
            return (
                moto.confidence if moto is not None else 0.0,
                bike.confidence if bike is not None else 0.0,
            )

        filtered_reports.sort(key=_veh_sort_key, reverse=True)
        sort_label = "sorted by motorcycle confidence"

    total_clips = len(clip_reports)
    detected_clips = len(filtered_reports)

    sections: list[str] = []
    for report in filtered_reports:
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

                if is_present:
                    # Present mode: generic label, no metrics, no Claude info
                    display_name = _PRESENT_LABEL
                    metric_html = ""
                    badge_html = ""
                else:
                    # Debug modes: show actual class name, metrics, Claude info
                    display_name = class_name
                    if mode == "hsp" and det.speed is not None:
                        metric_label = f"{det.speed:.0f} px/sec"
                    else:
                        metric_label = f"{det.confidence:.0%}"
                    metric_html = f'<br/><span style="color:#64748b">{metric_label}</span>'
                    badge_html = ""
                    if det.verification_status == "confirmed":
                        badge_html = (
                            '<div style="color:#16a34a;font-size:0.8em;margin-top:4px">'
                            "&#10003; Claude verified</div>"
                        )
                    elif det.verification_status == "rejected":
                        badge_html = (
                            '<div style="color:#dc2626;font-size:0.8em;margin-top:4px">'
                            "&#10007; Claude rejected</div>"
                        )
                    if det.verification_response is not None:
                        badge_html += (
                            '<div style="color:#94a3b8;font-size:0.75em;'
                            'font-style:italic;margin-top:2px">'
                            f"Claude: &ldquo;{det.verification_response}&rdquo;</div>"
                        )

                card_items.append(
                    f'<div style="border:1px solid #e2e8f0;border-radius:8px;padding:8px;'
                    f'text-align:center;width:160px">'
                    f'<img src="{img_uri}" style="max-width:144px;max-height:144px;'
                    f'border-radius:4px;display:block;margin:0 auto 6px" />'
                    f"<strong>{display_name}</strong>"
                    f"{metric_html}"
                    f"{badge_html}"
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
            if is_present:
                best_str = f'<span style="color:#16a34a;font-weight:bold">{_PRESENT_LABEL}</span>'
            else:
                best_str = (
                    f'<span style="color:#16a34a;font-weight:bold">'
                    f"{report.best_detection.class_name} "
                    f"({report.best_detection.confidence:.0%})</span>"
                )
        else:
            if is_present:
                best_str = '<span style="color:#94a3b8">No detection</span>'
            else:
                best_str = '<span style="color:#94a3b8">No detection above threshold</span>'

        video_html = ""
        if include_video:
            video_html = (
                f'<video controls preload="metadata" style="max-width:100%;max-height:300px;'
                f'border-radius:6px;margin-top:6px">'
                f'<source src="{report.mp4_filename}" type="video/mp4" />'
                f"</video>"
            )

        sections.append(
            f'<div style="border-left:4px solid {border_color};background:{bg_color};'
            f'padding:12px 16px;margin-bottom:16px;border-radius:0 8px 8px 0">'
            f'<h3 style="margin:0 0 8px">{time_str} &mdash; '
            f'<code style="font-size:0.85em">{report.mp4_filename}</code></h3>'
            f'<p style="margin:4px 0">{best_str}</p>'
            f"{video_html}"
            f"{cards_html}"
            f"</div>"
        )

    settings_html = ""
    if settings and not is_present:
        settings_html = _render_settings_html(settings)

    html = (
        "<!DOCTYPE html>"
        '<html lang="en"><head><meta charset="utf-8"/>'
        f"<title>{title}</title>"
        '<meta name="viewport" content="width=device-width,initial-scale=1"/>'
        "<style>"
        "body{font-family:system-ui,sans-serif;max-width:900px;"
        "margin:0 auto;padding:20px;background:#fff}"
        "h1{margin-bottom:4px} .stats{color:#64748b;margin-bottom:24px}"
        "</style></head><body>"
        f"<h1>{title}</h1>"
        f'<p class="stats">{total_clips} clips analysed &middot; '
        f"{detected_clips} with detections ({sort_label})</p>"
        + settings_html
        + "".join(sections)
        + "</body></html>"
    )

    return html
