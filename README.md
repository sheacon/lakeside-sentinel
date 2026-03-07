# Lakeside Sentinel

Motorized vehicle detection and alert system that monitors a Google Nest camera using [YOLO26](https://docs.ultralytics.com/models/yolo26/#overview) object detection, filters with Claude Vision verification, and sends email alerts via Resend. Includes a human-in-the-loop review web app for correcting false positives and collecting YOLO fine-tuning annotations.

## Architecture

```
src/lakeside_sentinel/
├── main.py                # Monitor orchestration & daily run logic
├── cli.py                 # CLI argument parser (present mode default, --review, --debug --veh/--hsp)
├── config.py              # Pydantic settings from .env
├── camera/
│   ├── auth.py            # Google Nest auth via glocaltokens
│   ├── models.py          # CameraEvent dataclass (frozen)
│   └── nest_api.py        # Nest API client, MPEG-DASH XML parsing
├── detection/
│   ├── models.py          # Detection dataclass (frame, bbox, confidence, class_name, verification_response, speed)
│   ├── claude_verifier.py # Claude Vision verification of YOLO detections
│   ├── hsp_detector.py    # Experimental: person tracking + centroid displacement (HSP)
│   └── veh_detector.py    # YOLO vehicle detection (classes 1,3), dynamic imgsz
├── notification/
│   ├── email_sender.py    # Resend email: sends pre-built HTML report
│   └── html_report.py     # Self-contained HTML report; mode-aware (present/veh/hsp)
├── review/
│   ├── staging.py         # Serialize detections to disk for review web app
│   ├── server.py          # Flask web app for human-in-the-loop review
│   ├── fine_tuning.py     # YOLO-format annotation writer for fine-tuning dataset
│   └── templates/
│       └── review.html    # Jinja2 template for review UI
└── utils/
    ├── daylight.py        # Sunrise/sunset filtering & daylight spans via astral
    ├── image.py           # ROI cropping & bounding box cropping with padding
    └── video.py           # MP4 frame extraction via OpenCV
```

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # then fill in credentials
```

## Running

### Present mode (default)

Runs both VEH + HSP detection with Claude verification. Produces a clean report suitable for sharing. Always sends email. Requires `ANTHROPIC_API_KEY`.

```bash
python -m lakeside_sentinel                    # most recent daylight period
python -m lakeside_sentinel --date 2026-02-28  # specific date
```

### Review mode

Launches a local Flask web app for human-in-the-loop review. Backfills up to 14 days of missed detections, stages data, and opens the review UI. On submit: generates per-day reports, sends one combined email, and saves YOLO fine-tuning annotations.

```bash
python -m lakeside_sentinel --review                    # backfill + review all
python -m lakeside_sentinel --review --date 2026-03-05  # analyze specific date + review
```

### Debug mode

Runs a single detector with full diagnostic output (confidence scores, class names, Claude badges).

```bash
python -m lakeside_sentinel --debug --veh              # VEH detection
python -m lakeside_sentinel --debug --veh --date 2026-02-28  # VEH for a specific date
python -m lakeside_sentinel --debug --veh --claude     # VEH with Claude verification
python -m lakeside_sentinel --debug --veh --claude --claude-keep-rejected  # keep rejected
python -m lakeside_sentinel --debug --hsp              # HSP detection
python -m lakeside_sentinel --debug --hsp --claude     # HSP with Claude verification
```

## Scheduling

The repo includes `run.sh`, a self-locating entry point that activates the virtualenv and runs review mode. Hook it into your preferred scheduler:

**cron** (daily at 21:00):
```bash
0 21 * * * /path/to/lakeside-sentinel/run.sh >> /tmp/lakeside-sentinel.log 2>&1
```

**macOS LaunchAgent** (`~/Library/LaunchAgents/com.lakeside-sentinel.worker.plist`):
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.lakeside-sentinel.worker</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/lakeside-sentinel/run.sh</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>21</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
</dict>
</plist>
```

**systemd** (`~/.config/systemd/user/lakeside-sentinel.service` + `.timer`):
```ini
# lakeside-sentinel.service
[Unit]
Description=Lakeside Sentinel detection run

[Service]
Type=oneshot
ExecStart=/path/to/lakeside-sentinel/run.sh
```

```ini
# lakeside-sentinel.timer
[Unit]
Description=Run Lakeside Sentinel daily at 21:00

[Timer]
OnCalendar=*-*-* 21:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

## Track Visualization

Render HSP person tracks as an annotated video and static summary image. Fast tracks (above threshold) are red, slow tracks are green. Multi-value flags create a Cartesian product sweep:

```bash
python scripts/visualize_tracks.py --clip output/video/2026-03-01_16-44-19.mp4
python scripts/visualize_tracks.py --clip a.mp4 b.mp4 --fps 8 --displacement 320.0
python scripts/visualize_tracks.py --clip clip.mp4 --person-confidence 0.3
python scripts/visualize_tracks.py --clip clip.mp4 --max-match-distance 400.0 800.0
```

Output saved to `output/tracks/{clip_stem}/` (single run) or `output/tracks/{clip_stem}/{param_label}/` (sweep).

## Verification Diagnostics

Test Claude Vision verification against a video clip to measure consistency and debug rejections:

```bash
python scripts/test_verification.py --clip output/video/2026-03-01_16-44-19.mp4
python scripts/test_verification.py --clip output/video/clip.mp4 --runs 5 --save-crops
python scripts/test_verification.py --clip output/video/clip.mp4 --temperature 1.0
```

## Testing

```bash
pytest tests/
pytest tests/ -v --cov=src/lakeside_sentinel
```

Tests use mocks for all external services (YOLO, Resend, Nest API). Test fixtures in `tests/fixtures/`.

## Code Quality

```bash
ruff check .    # lint (E, F, I rules)
ruff format .   # format (100 char line length)
mypy .          # type checking
```

## Environment Variables

See `.env.example` for the full list. Key variables:

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_MASTER_TOKEN` | — | Google Nest master token |
| `GOOGLE_USERNAME` | — | Google account email |
| `NEST_DEVICE_ID` | — | Nest camera device ID |
| `RESEND_API_KEY` | — | Resend API key for email alerts |
| `ALERT_EMAIL_TO` | — | Recipient email address |
| `ALERT_EMAIL_FROM` | `alerts@xeroshot.org` | Sender email address |
| `CAMERA_LATITUDE` | — | Camera latitude (daylight filtering) |
| `CAMERA_LONGITUDE` | — | Camera longitude (daylight filtering) |
| `YOLO_MODEL` | `yolo_models/yolo26s.pt` | YOLO model weights file |
| `VEH_CONFIDENCE_THRESHOLD` | `0.4` | Minimum confidence for alerts and report inclusion |
| `YOLO_BATCH_SIZE` | `16` | Frames per YOLO inference batch (prevents GPU OOM) |
| `CROP_PADDING` | `0.2` | Padding around detected bounding box |
| `ROI_Y_START` | `0.0` | Vertical region of interest start (fraction 0.0–1.0) |
| `ROI_Y_END` | `1.0` | Vertical region of interest end (fraction 0.0–1.0) |
| `ROI_X_START` | `0.0` | Horizontal region of interest start (fraction 0.0–1.0) |
| `ROI_X_END` | `1.0` | Horizontal region of interest end (fraction 0.0–1.0) |
| `VEH_FPS_SAMPLE` | `2` | Frames extracted per second for VEH mode |
| `HSP_FPS_SAMPLE` | `4` | Frames per second for HSP mode (higher = better tracking) |
| `HSP_DISPLACEMENT_THRESHOLD` | `240.0` | Min centroid displacement (px/sec) to flag as HSP |
| `HSP_PERSON_CONFIDENCE_THRESHOLD` | `0.4` | Min YOLO person confidence for tracking |
| `HSP_MAX_MATCH_DISTANCE` | `800.0` | Max centroid distance (px) for track matching |
| `ANTHROPIC_API_KEY` | — | API key for Claude Vision verification (required for present/review mode) |
| `CLAUDE_VISION_MODEL` | `claude-sonnet-4-20250514` | Claude model for verification |
| `REVIEW_PORT` | `5000` | Port for the review web app server |
