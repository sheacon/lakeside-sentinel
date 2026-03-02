#!/bin/bash
# Entry point for scheduled execution — no hardcoded paths
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"
source .venv/bin/activate
exec python -m lakeside_sentinel --email
