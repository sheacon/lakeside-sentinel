# Lakeside Motorbikes

Vehicle detection and alert system that monitors a Google Nest camera using YOLO object detection and sends email alerts via Resend. Detects bicycles and motorcycles.

## Architecture

```
src/lakeside_motorbikes/
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
python -m lakeside_motorbikes              # analyze most recent daylight period
python -m lakeside_motorbikes --email      # also send an email report (no embedded videos)
python -m lakeside_motorbikes --date 2026-02-28  # analyze a specific date's daylight
python -m lakeside_motorbikes --hsp        # experimental HSP detection
python -m lakeside_motorbikes --hsp --email  # HSP detection with email report
```

Deployed as a macOS LaunchAgent via `com.lakeside-motorbikes.worker.plist` (runs daily at 21:00).

## Testing

```bash
pytest tests/                    # tests across 9 modules
pytest tests/ -v --cov=src/lakeside_motorbikes
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
- `GOOGLE_MASTER_TOKEN`, `GOOGLE_USERNAME`, `NEST_DEVICE_ID` - Nest auth
- `RESEND_API_KEY`, `ALERT_EMAIL_TO`, `ALERT_EMAIL_FROM` - email alerts
- `CAMERA_LATITUDE`, `CAMERA_LONGITUDE` - camera location (daylight filtering)
- `YOLO_MODEL` (default `yolo26s.pt`) - YOLO model weights file
- `YOLO_CONFIDENCE_THRESHOLD` (default 0.4), `YOLO_BATCH_SIZE` (default 16), `CROP_PADDING` (default 0.2)
- `ROI_Y_START` (default 0.0), `ROI_Y_END` (default 1.0) - vertical region of interest (fraction 0.0–1.0)
- `ROI_X_START` (default 0.0), `ROI_X_END` (default 1.0) - horizontal region of interest (fraction 0.0–1.0)
- `FPS_SAMPLE` (default 2) - frames extracted per second of video
- `HSP_FPS_SAMPLE` (default 4), `HSP_DISPLACEMENT_THRESHOLD` (default 60.0) - high-speed person mode
- `HSP_PERSON_CONFIDENCE` (default 0.4), `HSP_MAX_MATCH_DISTANCE` (default 200.0) - HSP tracking

## Conventions

- Python 3.12+, PEP 8, full type hints on all signatures
- Conventional commits (`feat:`, `fix:`, `chore:`)
- `@dataclass(frozen=True)` for domain models
- Logging via `logging.getLogger(__name__)`
- External API calls require timeout and error handling
- YOLO model weights (e.g. `yolo26s.pt`) are git-ignored and auto-downloaded
