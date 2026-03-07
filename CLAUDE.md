# Lakeside Sentinel

Vehicle detection and alert system that monitors a Google Nest camera using YOLO object detection and sends email alerts via Resend. Detects bicycles and motorcycles.

## Architecture

```
src/lakeside_sentinel/
├── main.py                # Monitor orchestration & run logic
├── cli.py                 # CLI argument parser (present mode default, --review, --veh/--hsp, --verbose)
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

Runs both VEH + HSP detection with Claude verification. Produces a clean report suitable for sharing (hides confidence scores, class names, and debug info), plus VEH and HSP debug reports with unfiltered detection data. Always sends email. Requires `ANTHROPIC_API_KEY`.

Output files:
- `output/report-{date}.html` — clean present report (opened in browser, emailed)
- `output/report-veh-{date}.html` — VEH debug report with all class detections (not opened/emailed)
- `output/report-hsp-{date}.html` — HSP debug report with fastest track regardless of threshold (not opened/emailed)
- `output/logs/{timestamp}.log` — timestamped log file with full logging output (created each run)

```bash
python -m lakeside_sentinel                    # most recent daylight period
python -m lakeside_sentinel --date 2026-02-28  # specific date
```

### Review mode

Launches a local Flask web app for human-in-the-loop review before sending reports. Backfills up to 14 days of missed detections. On submit: generates per-day reports, sends one combined email, and saves YOLO fine-tuning annotations. Requires `ANTHROPIC_API_KEY`.

```bash
python -m lakeside_sentinel --review                    # backfill + review all
python -m lakeside_sentinel --review --date 2026-03-05  # analyze specific date + review
```

Output files:
- `output/staging/{date}/` — staged detection data (frames + JSON) awaiting review
- `output/fine-tuning/` — YOLO-format annotations (`images/train/`, `labels/train/`, `data.yaml`)
- `output/fine-tuning/other/` — images classified as "other" for later manual review

### Single detector mode

Runs a single detector with full diagnostic output (confidence scores, class names, Claude badges).

```bash
python -m lakeside_sentinel --veh                      # VEH detection
python -m lakeside_sentinel --veh --date 2026-02-28    # VEH for a specific date
python -m lakeside_sentinel --veh --claude             # VEH with Claude verification
python -m lakeside_sentinel --veh --claude --claude-keep-rejected  # keep rejected
python -m lakeside_sentinel --hsp                      # HSP detection
python -m lakeside_sentinel --hsp --claude             # HSP with Claude verification
```

### Verbose mode

Add `--verbose` to any mode for DEBUG-level logging output.

```bash
python -m lakeside_sentinel --verbose                  # present mode + DEBUG logging
python -m lakeside_sentinel --review --verbose         # review mode + DEBUG logging
python -m lakeside_sentinel --veh --verbose            # single detector + DEBUG logging
```

Scheduled via `run.sh` (passes `--review`) — a self-locating entry point for cron, launchd, or systemd. See README.md for scheduler examples.

### Auto-cleanup

On each run, files older than 14 days are automatically deleted from:
- `output/logs/*.log` — log files
- `output/video/*.mp4` — video clips
- `output/staging/*/` — staging directories

## Tuning

`scripts/tune_detection.py` sweeps detection parameters on a video clip via CLI flags. Multi-value flags create a Cartesian product sweep.

```bash
# VEH mode — sweep models, FPS, and confidence thresholds
python scripts/tune_detection.py --clip output/video/2026-02-28_12-31-14.mp4 \
    --model yolo_models/yolo26s.pt yolo_models/yolo26m.pt \
    --fps 2 4 8 \
    --confidence 0.3 0.4 0.5 \
    --roi-y-start 0.0 --roi-y-end 0.28 \
    --roi-x-start 0.33 --roi-x-end 1.0

# HSP mode — sweep displacement thresholds (px/sec) and person confidence
python scripts/tune_detection.py --clip output/video/2026-02-28_12-31-14.mp4 \
    --hsp \
    --fps 4 8 \
    --hsp-displacement 160.0 240.0 320.0 \
    --hsp-person-confidence 0.3 0.4 \
    --hsp-max-match-distance 600.0 800.0
```

Annotated images are saved to `output/tune/{clip_stem}/`. Results are printed as a table to stdout.

## Verification Diagnostics

`scripts/test_verification.py` tests Claude Vision verification against a video clip. Runs YOLO detection, then sends each crop to Claude N times to measure consistency. Uses `temperature=0` by default (matching production).

```bash
# Default: 3 runs at temperature=0
python scripts/test_verification.py --clip output/video/2026-03-01_16-44-19.mp4

# 5 runs, save crop images
python scripts/test_verification.py --clip output/video/2026-03-01_16-44-19.mp4 --runs 5 --save-crops

# Compare with stochastic temperature
python scripts/test_verification.py --clip output/video/2026-03-01_16-44-19.mp4 --temperature 1.0
```

Crop images are saved to `output/verification/{clip_stem}/` when `--save-crops` is used.

## Track Visualization

`scripts/visualize_tracks.py` renders HSP person tracks as an annotated video and a static summary image. Fast tracks (above threshold) are red, slow tracks are green. All CLI params default to `.env.example` values so no `.env` is required. Multi-value flags create a Cartesian product sweep with output per combo.

```bash
# Default settings
python scripts/visualize_tracks.py --clip output/video/2026-03-01_16-44-19.mp4

# Multiple clips with overrides
python scripts/visualize_tracks.py --clip a.mp4 b.mp4 --fps 8 --displacement 320.0

# Multi-value sweep (2x3 = 6 runs per clip)
python scripts/visualize_tracks.py --clip clip.mp4 \
    --max-match-distance 400.0 800.0 \
    --displacement 120.0 240.0 320.0

# With ROI and confidence override
python scripts/visualize_tracks.py --clip clip.mp4 \
    --person-confidence 0.3 --roi-y-start 0.0 --roi-y-end 0.28
```

Output directory: `output/tracks/{clip_stem}/` (single run) or `output/tracks/{clip_stem}/{param_label}/` (sweep)
- `{clip_stem}_tracks.mp4` — annotated video with progressive track visualization
- `{clip_stem}_summary.jpg` — static summary image with all tracks

## Testing

```bash
pytest tests/                    # tests across 17 modules
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
- `YOLO_MODEL` (default `yolo_models/yolo26s.pt`) - YOLO model weights file
- `VEH_CONFIDENCE_THRESHOLD` (default 0.4) - filters alerts and HTML report class detections, `YOLO_BATCH_SIZE` (default 16), `CROP_PADDING` (default 0.2)
- `ROI_Y_START` (default 0.0), `ROI_Y_END` (default 1.0) - vertical region of interest (fraction 0.0–1.0)
- `ROI_X_START` (default 0.0), `ROI_X_END` (default 1.0) - horizontal region of interest (fraction 0.0–1.0)
- `VEH_FPS_SAMPLE` (default 2) - frames extracted per second for VEH mode
- `HSP_FPS_SAMPLE` (default 4), `HSP_DISPLACEMENT_THRESHOLD` (default 240.0, px/sec) - high-speed person mode
- `HSP_PERSON_CONFIDENCE_THRESHOLD` (default 0.4), `HSP_MAX_MATCH_DISTANCE` (default 800.0, px/sec) - HSP tracking (thresholds are FPS-invariant)
- `ANTHROPIC_API_KEY` - API key for Claude Vision verification (required for present mode, review mode, and `--veh/--hsp --claude`)
- `CLAUDE_VISION_MODEL` (default `claude-sonnet-4-20250514`) - Claude model for verification (uses `temperature=0` for deterministic classification; raw response text shown in HTML report)
- `CLAUDE_VISION_PROMPT` - Custom prompt for Claude Vision verification (defaults to built-in motorized vehicle prompt)
- `REVIEW_PORT` (default 5000) - port for the review web app server

## Conventions

- Python 3.12+, PEP 8, full type hints on all signatures
- Conventional commits (`feat:`, `fix:`, `chore:`)
- `@dataclass(frozen=True)` for domain models
- Logging via `logging.getLogger(__name__)`
- External API calls require timeout and error handling
- YOLO model weights live in `yolo_models/` (e.g. `yolo_models/yolo26s.pt`) — git-ignored and auto-downloaded
