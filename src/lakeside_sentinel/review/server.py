"""Flask web app for human-in-the-loop review of detections."""

from __future__ import annotations

import logging
import socket
import threading
import webbrowser
from collections.abc import Callable
from pathlib import Path
from queue import Queue

import cv2
from flask import Flask, Response, jsonify, redirect, render_template, request, send_file, url_for

from lakeside_sentinel.review.staging import (
    discover_unreviewed,
    load_frame,
    load_staged_detections,
)
from lakeside_sentinel.utils.image import crop_to_bbox

logger = logging.getLogger(__name__)

SubmitCallback = Callable[[dict[str, object]], dict[str, object]]

_shutdown_queue: Queue[dict[str, object]] = Queue()

_VEH_CLASS_PRIORITY = {"Motorcycle": 0, "Bicycle": 1}


def _sort_detections(detections: list[dict[str, object]]) -> list[dict[str, object]]:
    """Sort detections within each section for display.

    - confirmed: first, original order
    - veh_debug: by class priority (Motorcycle > Bicycle), then confidence desc
    - hsp_debug: by speed desc
    """

    def _sort_key(det: dict[str, object]) -> tuple:
        section = det["section"]
        if section == "veh_debug":
            return (1, _VEH_CLASS_PRIORITY.get(det["class_name"], 2), -(det["confidence"] or 0))
        elif section == "hsp_debug":
            return (2, -(det["speed"] or 0))
        else:  # confirmed
            return (0,)

    return sorted(detections, key=_sort_key)


def _video_review_score(detections: list[dict[str, object]]) -> float:
    """Compute a review-worthiness score for a video's detections.

    Higher scores mean more likely to be interesting. Used for ranking
    video groups in the review UI.
    """
    max_score = 0.0
    for det in detections:
        if det["source"] == "veh":
            class_boost = {"Motorcycle": 1.0, "Bicycle": 0.5}.get(det["class_name"], 0.0)
            score = class_boost + (det["confidence"] or 0.0)
        else:  # hsp
            score = (det["speed"] or 0.0) / 300.0
        max_score = max(max_score, score)
    return max_score


def _group_by_video(
    detections: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Group detections by mp4_filename for video-grouped display.

    Returns a list of video group dicts sorted by: confirmed first,
    then by review score descending, then by event time.
    """
    groups: dict[str, list[dict[str, object]]] = {}
    for det in detections:
        mp4 = det["mp4_filename"]
        groups.setdefault(mp4, []).append(det)

    result: list[dict[str, object]] = []
    for mp4, dets in groups.items():
        sorted_dets = _sort_detections(dets)
        has_confirmed = any(d["section"] == "confirmed" for d in dets)
        review_score = _video_review_score(dets)
        event_time_iso = dets[0]["event_time_iso"]
        result.append(
            {
                "mp4_filename": mp4,
                "event_time_iso": event_time_iso,
                "has_confirmed": has_confirmed,
                "review_score": review_score,
                "detections": sorted_dets,
            }
        )

    result.sort(key=lambda g: (not g["has_confirmed"], -g["review_score"], g["event_time_iso"]))
    return result


def _create_app(submit_callback: SubmitCallback | None = None) -> Flask:
    """Create and configure the Flask app.

    When `submit_callback` is provided, the server runs in always-on service
    mode: `/submit` invokes the callback inline and the page redirects back
    to the index. When omitted, `/submit` and `/exit` push onto the module
    shutdown queue (legacy one-shot mode used by `run_review_server`).
    """
    template_dir = Path(__file__).parent / "templates"
    app = Flask(__name__, template_folder=str(template_dir))
    service_mode = submit_callback is not None

    @app.route("/")
    def index() -> Response | str:
        dirs = discover_unreviewed()
        if not dirs:
            return render_template("empty.html", service_mode=service_mode)
        first_date = dirs[0].name
        return redirect(url_for("review_day", date_str=first_date))

    @app.route("/<date_str>")
    def review_day(date_str: str) -> str | tuple[str, int]:
        """Render review page for a specific day."""
        dirs = discover_unreviewed()
        date_names = [d.name for d in dirs]

        if date_str not in date_names:
            return f"No staging data for {date_str}", 404

        idx = date_names.index(date_str)
        prev_date = date_names[idx - 1] if idx > 0 else None
        next_date = date_names[idx + 1] if idx < len(date_names) - 1 else None

        # Load all days' data for client-side state
        all_days: dict[str, object] = {}
        for d in dirs:
            data = load_staged_detections(d)
            all_days[d.name] = data

        current_data = all_days[date_str]
        video_groups = _group_by_video(current_data["detections"])

        days_with_confirmed = [
            name
            for name, data in all_days.items()
            if any(d["section"] == "confirmed" for d in data["detections"])
        ]

        return render_template(
            "review.html",
            current_date=date_str,
            prev_date=prev_date,
            next_date=next_date,
            all_dates=date_names,
            current_data=current_data,
            video_groups=video_groups,
            all_days=all_days,
            days_with_confirmed=days_with_confirmed,
            service_mode=service_mode,
        )

    @app.route("/frame/<date_str>/<filename>")
    def serve_frame(date_str: str, filename: str) -> Response | tuple[str, int]:
        """Serve a staged frame PNG."""
        staging_dir = Path("output") / "staging" / date_str
        frame_path = staging_dir / filename
        if not frame_path.exists():
            return "Frame not found", 404
        return Response(frame_path.read_bytes(), mimetype="image/png")

    @app.route("/crop/<date_str>/<detection_id>")
    def serve_crop(date_str: str, detection_id: str) -> Response | tuple[str, int]:
        """Serve a cropped detection image (generated on-the-fly)."""
        staging_dir = Path("output") / "staging" / date_str
        json_path = staging_dir / "staging.json"
        if not json_path.exists():
            return "Staging data not found", 404

        data = load_staged_detections(staging_dir)
        det_dict = None
        for d in data["detections"]:
            if d["id"] == detection_id:
                det_dict = d
                break

        if det_dict is None:
            return "Detection not found", 404

        frame = load_frame(staging_dir, det_dict["frame_filename"])
        bbox = tuple(det_dict["bbox"])
        crop_padding = data.get("crop_padding", 0.2)
        cropped = crop_to_bbox(frame, bbox, padding=crop_padding)

        # Upscale and encode as JPEG
        h, w = cropped.shape[:2]
        cropped = cv2.resize(cropped, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        _, jpeg_buf = cv2.imencode(".jpg", cropped, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return Response(jpeg_buf.tobytes(), mimetype="image/jpeg")

    @app.route("/video/<path:mp4_path>")
    def serve_video(mp4_path: str) -> Response | tuple[str, int]:
        """Serve an MP4 video file with range request support."""
        video_path = Path("output") / mp4_path
        if not video_path.exists():
            return "Video not found", 404
        return send_file(video_path.resolve(), mimetype="video/mp4", conditional=True)

    @app.route("/submit", methods=["POST"])
    def submit() -> Response | tuple[Response, int]:
        payload = request.get_json()
        if not payload or "days" not in payload:
            return jsonify({"error": "Invalid payload"}), 400
        days = payload["days"]

        if submit_callback is not None:
            try:
                result = submit_callback(days)
            except Exception:
                logger.exception("Submit callback failed")
                return jsonify({"error": "Submit processing failed"}), 500
            response: dict[str, object] = {"status": "ok", "next_url": "/"}
            response.update(result)
            return jsonify(response)

        _shutdown_queue.put({"action": "submit", "days": days})
        return jsonify({"status": "ok"})

    @app.route("/exit", methods=["POST"])
    def exit_review() -> Response:
        if submit_callback is None:
            _shutdown_queue.put({"action": "exit"})
        return jsonify({"status": "ok"})

    return app


def _is_port_available(host: str, port: int) -> bool:
    """Check if a host:port is available for binding."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
        return True
    except OSError:
        return False


def _reachable_url(host: str, port: int) -> str:
    """Return a URL the operator can paste into a browser.

    For non-loopback binds, prefer the machine's hostname so the log line is
    usable from another device on the same network (e.g. tailnet MagicDNS).
    """
    if host in ("127.0.0.1", "localhost"):
        display_host = "127.0.0.1"
    elif host in ("0.0.0.0", "::"):
        display_host = socket.gethostname()
    else:
        display_host = host
    return f"http://{display_host}:{port}/"


def run_review_server(host: str = "127.0.0.1", port: int = 5000) -> dict[str, object] | None:
    """Start the Flask review server and block until user submits or exits.

    One-shot mode: returns the submit payload (with 'action' and 'days') or
    None on exit. Used by `--review` for ad-hoc local review sessions.
    """
    global _shutdown_queue
    _shutdown_queue = Queue()

    if not _is_port_available(host, port):
        logger.info("Review server port %d is already in use; server already running.", port)
        return None

    app = _create_app()

    server_thread = threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False),
        daemon=True,
    )
    server_thread.start()

    url = _reachable_url(host, port)
    logger.info("Review server started at %s", url)
    if host in ("127.0.0.1", "localhost"):
        webbrowser.open(url)

    result = _shutdown_queue.get()

    if result.get("action") == "submit":
        return result
    return None


def run_persistent_review_server(
    host: str,
    port: int,
    submit_callback: SubmitCallback,
) -> None:
    """Run the review server forever, handling submits in-place.

    Service mode: Flask runs in the main thread and never exits. Each
    `/submit` invokes `submit_callback` synchronously (writes fine-tuning
    annotations, sends summary email, cleans up staging) and the client
    redirects back to `/`, which lands on the next unreviewed day or the
    empty state.
    """
    if not _is_port_available(host, port):
        logger.error("Review server port %d is already in use; cannot start service.", port)
        raise SystemExit(1)

    app = _create_app(submit_callback=submit_callback)

    url = _reachable_url(host, port)
    logger.info("Review service started at %s (always-on)", url)

    app.run(host=host, port=port, debug=False, use_reloader=False)
