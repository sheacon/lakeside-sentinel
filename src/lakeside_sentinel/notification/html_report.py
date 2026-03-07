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


def _sharpen_image(image: np.ndarray, sigma: float = 1.0, strength: float = 1.5) -> np.ndarray:
    """Apply unsharp mask sharpening to an image."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)


def _encode_cropped_png(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    padding: float,
    *,
    enhance: bool = False,
    for_email: bool = False,
    cid_index: int = 0,
) -> str | tuple[str, str, bytes]:
    """Crop a detection and return image data for HTML embedding.

    Args:
        frame: The full video frame.
        bbox: Bounding box coordinates (x1, y1, x2, y2).
        padding: Padding fraction for cropping.
        enhance: When True, upscale 2x with cubic interpolation and sharpen.
        for_email: When True, return CID reference and JPEG bytes for email attachment.
        cid_index: Index used to generate unique CID identifiers.

    Returns:
        When for_email is False: a base64-encoded PNG data URI string.
        When for_email is True: a tuple of (cid_src, cid_id, jpeg_bytes).
    """
    cropped = crop_to_bbox(frame, bbox, padding=padding)
    if enhance:
        h, w = cropped.shape[:2]
        cropped = cv2.resize(cropped, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        cropped = _sharpen_image(cropped)

    if for_email:
        cid = f"detection-{cid_index}"
        _, jpeg_buf = cv2.imencode(".jpg", cropped, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return f"cid:{cid}", cid, jpeg_buf.tobytes()

    _, png_bytes = cv2.imencode(".png", cropped)
    b64 = base64.b64encode(png_bytes.tobytes()).decode()
    return f"data:image/png;base64,{b64}"


def _render_settings_html(settings: dict[str, object]) -> str:
    """Render settings as an HTML block, masking sensitive values."""
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
        '<div style="margin-bottom:20px;color:#64748b;font-size:0.85em">'
        '<h3 style="margin:0 0 8px;font-size:1em;font-weight:600">Parameters</h3>'
        '<table style="border-collapse:collapse">' + "".join(rows) + "</table></div>"
    )


def generate_report(
    clip_reports: list[ClipReport],
    crop_padding: float = 0.2,
    include_video: bool = True,
    title: str = "Detection Report",
    mode: str = "veh",
    settings: dict[str, object] | None = None,
    subtitle: str | None = None,
    for_email: bool = False,
    total_clips: int | None = None,
) -> tuple[str, list[dict[str, str | bytes]]]:
    """Generate a self-contained HTML report.

    Args:
        clip_reports: Analysis results for each clip.
        crop_padding: Padding fraction for cropping detection images.
        include_video: Whether to include video player elements.
        title: Title for the HTML report.
        mode: Detection mode ("veh", "hsp", or "present"). Controls sorting and display.
        settings: Optional settings dict to display in debug mode reports.
        subtitle: Optional subtitle shown below the title.
        for_email: When True, use CID inline attachments (JPEG) instead of base64 data URIs.

    Returns:
        A tuple of (html_string, attachments) where attachments is a list of dicts
        for Resend CID attachments. Empty list when for_email is False.
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

    total_clips_count = total_clips if total_clips is not None else len(clip_reports)
    detected_clips = len(filtered_reports)

    sections: list[str] = []
    attachments: list[dict[str, str | bytes]] = []
    cid_counter = 0
    for report in filtered_reports:
        local_time = report.event_time.astimezone()
        time_str = local_time.strftime("%m-%d %H:%M:%S")
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
                if for_email:
                    result = _encode_cropped_png(
                        det.frame,
                        det.bbox,
                        crop_padding,
                        enhance=is_present,
                        for_email=True,
                        cid_index=cid_counter,
                    )
                    assert isinstance(result, tuple)
                    img_uri, cid_id, jpeg_bytes = result
                    attachments.append(
                        {
                            "filename": f"{cid_id}.jpg",
                            "content": base64.b64encode(jpeg_bytes).decode(),
                            "content_type": "image/jpeg",
                            "content_id": cid_id,
                        }
                    )
                    cid_counter += 1
                else:
                    result = _encode_cropped_png(
                        det.frame, det.bbox, crop_padding, enhance=is_present
                    )
                    assert isinstance(result, str)
                    img_uri = result

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

                card_w = 320 if is_present else 160
                img_max = 288 if is_present else 144
                card_items.append(
                    f'<div style="border:1px solid #e2e8f0;border-radius:8px;padding:8px;'
                    f'text-align:center;width:{card_w}px">'
                    f'<img src="{img_uri}" style="max-width:{img_max}px;max-height:{img_max}px;'
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

        if for_email:
            heading_html = f'<h3 style="margin:0 0 8px">{time_str}</h3>'
        else:
            heading_html = (
                f'<h3 style="margin:0 0 8px">{time_str} &mdash; '
                f'<code style="font-size:0.85em">{report.mp4_filename}</code></h3>'
            )

        sections.append(
            f'<div style="border-left:4px solid {border_color};background:{bg_color};'
            f'padding:12px 16px;margin-bottom:16px;border-radius:0 8px 8px 0">'
            f"{heading_html}"
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
        + (
            f'<p style="color:#475569;margin-top:0;margin-bottom:8px">{subtitle}</p>'
            if subtitle
            else ""
        )
        + f'<p class="stats">{total_clips_count} clips analysed &middot; '
        f"{detected_clips} with detections ({sort_label})</p>"
        + settings_html
        + "".join(sections)
        + "</body></html>"
    )

    return html, attachments
