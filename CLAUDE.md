# Lakeside Sentinel

Vehicle detection and alert system that monitors a Google Nest camera using YOLO object detection and sends email alerts via Resend. Detects bicycles and motorcycles.

## Architecture

```
src/lakeside_sentinel/
├── main.py                # Monitor orchestration & run logic
├── cli.py                 # CLI argument parser (--veh, --hsp, --date, --email, --claude)
├── config.py              # Pydantic settings from .env
├── camera/
│   ├── auth.py            # Google Nest auth via glocaltokens
│   ├── models.py          # CameraEvent dataclass (frozen)
│   └── nest_api.py        # Nest API client, MPEG-DASH XML parsing
├── detection/
│   ├── models.py          # Detection dataclass (frame, bbox, confidence, class_name)
│   ├── claude_verifier.py # Claude Vision verification of YOLO detections
│   ├── hsp_detector.py    # Experimental: person tracking + centroid displacement (HSP)
│   └── veh_detector.py    # YOLO vehicle detection (classes 1,3), dynamic imgsz
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

A mode (`--veh` or `--hsp`) must be specified — there is no default.

```bash
python -m lakeside_sentinel --veh              # VEH mode: most recent daylight period
python -m lakeside_sentinel --veh --email      # VEH mode with email report
python -m lakeside_sentinel --veh --date 2026-02-28  # VEH mode for a specific date
python -m lakeside_sentinel --veh --claude     # VEH mode with Claude Vision verification
python -m lakeside_sentinel --veh --claude --claude-keep-rejected  # keep rejected in report
python -m lakeside_sentinel --hsp              # HSP detection mode
python -m lakeside_sentinel --hsp --email      # HSP detection with email report
python -m lakeside_sentinel --hsp --claude     # HSP mode with Claude Vision verification
```

Scheduled via `run.sh` (passes `--veh --email`) — a self-locating entry point for cron, launchd, or systemd. See README.md for scheduler examples.

## Tuning

`scripts/tune_detection.py` sweeps detection parameters on a video clip via CLI flags. Multi-value flags create a Cartesian product sweep.

```bash
# VEH mode — sweep models, FPS, and confidence thresholds
python scripts/tune_detection.py --clip output/video/2026-02-28_12-31-14.mp4 \
    --model yolo26s.pt yolo26m.pt \
    --fps 2 4 8 \
    --confidence 0.3 0.4 0.5 \
    --roi-y-start 0.0 --roi-y-end 0.28 \
    --roi-x-start 0.33 --roi-x-end 1.0

# HSP mode — sweep displacement thresholds and person confidence
python scripts/tune_detection.py --clip output/video/2026-02-28_12-31-14.mp4 \
    --hsp \
    --fps 4 8 \
    --hsp-displacement 40.0 60.0 80.0 \
    --hsp-person-confidence 0.3 0.4 \
    --hsp-max-match-distance 150.0 200.0
```

Annotated images are saved to `output/tune/{clip_stem}/`. Results are printed as a table to stdout.

## Testing

```bash
pytest tests/                    # tests across 10 modules
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
- `GOOGLE_MASTER_TOKEN`, `GOOGLE_USERNAME`, `NEST_DEVICE_ID` - Nest auth
- `RESEND_API_KEY`, `ALERT_EMAIL_TO`, `ALERT_EMAIL_FROM` - email alerts
- `CAMERA_LATITUDE`, `CAMERA_LONGITUDE` - camera location (daylight filtering)
- `YOLO_MODEL` (default `yolo26s.pt`) - YOLO model weights file
- `VEH_CONFIDENCE_THRESHOLD` (default 0.4) - filters alerts and HTML report class detections, `YOLO_BATCH_SIZE` (default 16), `CROP_PADDING` (default 0.2)
- `ROI_Y_START` (default 0.0), `ROI_Y_END` (default 1.0) - vertical region of interest (fraction 0.0–1.0)
- `ROI_X_START` (default 0.0), `ROI_X_END` (default 1.0) - horizontal region of interest (fraction 0.0–1.0)
- `VEH_FPS_SAMPLE` (default 2) - frames extracted per second for VEH mode
- `HSP_FPS_SAMPLE` (default 4), `HSP_DISPLACEMENT_THRESHOLD` (default 60.0) - high-speed person mode
- `HSP_PERSON_CONFIDENCE_THRESHOLD` (default 0.4), `HSP_MAX_MATCH_DISTANCE` (default 200.0) - HSP tracking
- `ANTHROPIC_API_KEY` - API key for Claude Vision verification (optional, required for `--claude`)
- `CLAUDE_VISION_MODEL` (default `claude-sonnet-4-20250514`) - Claude model for verification

## Conventions

- Python 3.12+, PEP 8, full type hints on all signatures
- Conventional commits (`feat:`, `fix:`, `chore:`)
- `@dataclass(frozen=True)` for domain models
- Logging via `logging.getLogger(__name__)`
- External API calls require timeout and error handling
- YOLO model weights (e.g. `yolo26s.pt`) are git-ignored and auto-downloaded
