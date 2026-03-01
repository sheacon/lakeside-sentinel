# Lakeside Motorbikes

Vehicle detection and alert system that monitors a Google Nest camera using YOLO object detection and sends email alerts via Resend. Detects bicycles, cars, motorcycles, buses, and trucks.

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
│   └── vehicle_detector.py # YOLO11n vehicle detection (classes 1,2,3,5,7)
├── notification/
│   └── email_sender.py    # Resend email: single alerts + backfill summary
└── utils/
    ├── daylight.py        # Sunrise/sunset filtering via astral
    ├── image.py           # Bounding box cropping with padding
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
- `YOLO_CONFIDENCE_THRESHOLD` (default 0.4), `CROP_PADDING` (default 0.2)
- `FPS_SAMPLE` (default 2) - frames extracted per second of video
- `POLL_INTERVAL_SECONDS` (default 120)

## Conventions

- Python 3.12+, PEP 8, full type hints on all signatures
- Conventional commits (`feat:`, `fix:`, `chore:`)
- `@dataclass(frozen=True)` for domain models
- Logging via `logging.getLogger(__name__)`
- External API calls require timeout and error handling
- YOLO model weights (`yolo11n.pt`) are git-ignored and auto-downloaded
