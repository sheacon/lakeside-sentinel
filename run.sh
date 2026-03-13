#!/bin/bash
# Entry point for scheduled execution — no hardcoded paths
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"
export PATH="$HOME/.local/bin:$PATH"
exec uv run python -m lakeside_sentinel
