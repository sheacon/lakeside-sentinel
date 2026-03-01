# Lakeside Motorbikes

Vehicle detection and alert system that monitors a Google Nest camera using YOLO object detection and sends email alerts via Resend. Detects bicycles and motorcycles.

## Architecture

```
src/lakeside_motorbikes/
├── main.py                # Monitor orchestration & polling logic
├── cli.py                 # CLI argument parser (--backfill, --debug-dump)
├── config.py              # Pydantic settings from .env
├── camera/
│   ├── auth.py            # Google Nest auth via glocaltokens
│   ├── models.py          # CameraEvent dataclass (frozen)
│   └── nest_api.py        # Nest API client, MPEG-DASH XML parsing
├── detection/
│   ├── models.py          # Detection dataclass (frame, bbox, confidence, class_name)
│   └── vehicle_detector.py # YOLO vehicle detection (classes 1,3), dynamic imgsz
├── notification/
│   └── email_sender.py    # Resend email: single alerts + backfill summary
└── utils/
    ├── daylight.py        # Sunrise/sunset filtering via astral
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
python -m lakeside_motorbikes              # live monitoring (polls every 120s)
python -m lakeside_motorbikes --backfill   # analyze past 24 hours
python -m lakeside_motorbikes --backfill --debug-dump  # save clips as MP4s (cached)
```

Deployed as a macOS LaunchAgent via `com.lakeside-motorbikes.worker.plist`.

## Testing

```bash
pytest tests/                    # 44 tests across 8 modules
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
- `POLL_INTERVAL_SECONDS` (default 120)

## Conventions

- Python 3.12+, PEP 8, full type hints on all signatures
- Conventional commits (`feat:`, `fix:`, `chore:`)
- `@dataclass(frozen=True)` for domain models
- Logging via `logging.getLogger(__name__)`
- External API calls require timeout and error handling
- YOLO model weights (e.g. `yolo26s.pt`) are git-ignored and auto-downloaded
