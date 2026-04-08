# Lakeside Sentinel

Motorized vehicle detection on a Google Nest camera. YOLO inference + Claude Vision verification, Resend email alerts, Flask review web app for human-in-the-loop corrections and fine-tuning data collection.

All setup, running commands, env vars, scheduling, and per-script usage are in `README.md`. When in doubt, read it first.

## Notes

- **Fine-tuning replaces the class set** (80 COCO → 7 custom), so `veh_detector.py` which hardcodes COCO indices `{1, 3}` needs a follow-up to read `model.names` before production can consume a fine-tuned model. Scripts: `scripts/finetune.py`, `scripts/evaluate_model.py`.
- **Auto-cleanup** runs on each invocation: files older than 14 days in `output/logs/`, `output/video/`, and `output/staging/` are deleted; a warning email fires 3 days before staging cleanup.

## Testing and linting

`uv run pytest tests/` · `uv run ruff check .` · `uv run ruff format .` · `uv run mypy .`. Tests mock all external services (YOLO, Resend, Nest, Anthropic).

## Conventions

- Python 3.12+, PEP 8, full type hints on all signatures
- Conventional commits (`feat:`, `fix:`, `chore:`)
- `@dataclass(frozen=True)` for domain models
- Logging via `logging.getLogger(__name__)`
- External API calls require timeout and error handling
- YOLO model weights live in `yolo_models/` — git-ignored, auto-downloaded
