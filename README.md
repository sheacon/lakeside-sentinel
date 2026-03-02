# Lakeside Sentinel

Vehicle detection and alert system that monitors a Google Nest camera using [YOLO26](https://docs.ultralytics.com/models/yolo26/#overview) object detection and sends email alerts via Resend. Detects bicycles and motorcycles.

![YOLO26 Benchmark](yolo26-benchmark.jpg)

## Architecture

```
src/lakeside_sentinel/
├── main.py                # Monitor orchestration & daily run logic
├── cli.py                 # CLI argument parser (--date, --email, --hsp)
├── config.py              # Pydantic settings from .env
├── camera/
│   ├── auth.py            # Google Nest auth via glocaltokens
│   ├── models.py          # CameraEvent dataclass (frozen)
│   └── nest_api.py        # Nest API client, MPEG-DASH XML parsing
├── detection/
│   ├── models.py          # Detection dataclass (frame, bbox, confidence, class_name)
│   ├── hsp_detector.py    # Experimental: person tracking + centroid displacement (HSP)
│   └── vehicle_detector.py # YOLO vehicle detection (classes 1,3), dynamic imgsz
├── notification/
│   ├── email_sender.py    # Resend email: sends pre-built HTML report
│   └── html_report.py     # Self-contained HTML report generation
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

```bash
python -m lakeside_sentinel              # analyze most recent daylight period
python -m lakeside_sentinel --email      # also send an email report (no embedded videos)
python -m lakeside_sentinel --date 2026-02-28  # analyze a specific date's daylight
python -m lakeside_sentinel --hsp        # experimental HSP detection
python -m lakeside_sentinel --hsp --email  # HSP detection with email report
```

## Scheduling

The repo includes `run.sh`, a self-locating entry point that activates the virtualenv and runs the detector with email reporting. Hook it into your preferred scheduler:

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
| `YOLO_MODEL` | `yolo26s.pt` | YOLO model weights file |
| `YOLO_CONFIDENCE_THRESHOLD` | `0.4` | Minimum confidence for alerts |
| `YOLO_BATCH_SIZE` | `16` | Frames per YOLO inference batch (prevents GPU OOM) |
| `CROP_PADDING` | `0.2` | Padding around detected bounding box |
| `ROI_Y_START` | `0.0` | Vertical region of interest start (fraction 0.0–1.0) |
| `ROI_Y_END` | `1.0` | Vertical region of interest end (fraction 0.0–1.0) |
| `ROI_X_START` | `0.0` | Horizontal region of interest start (fraction 0.0–1.0) |
| `ROI_X_END` | `1.0` | Horizontal region of interest end (fraction 0.0–1.0) |
| `FPS_SAMPLE` | `2` | Frames extracted per second of video |
| `HSP_FPS_SAMPLE` | `4` | Frames per second for HSP mode (higher = better tracking) |
| `HSP_DISPLACEMENT_THRESHOLD` | `60.0` | Min centroid displacement (px/interval) to flag as HSP |
| `HSP_PERSON_CONFIDENCE` | `0.4` | Min YOLO person confidence for tracking |
| `HSP_MAX_MATCH_DISTANCE` | `200.0` | Max centroid distance (px) for track matching |
