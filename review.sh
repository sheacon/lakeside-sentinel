#!/bin/bash
# Launch the review web app for staged detection data
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"
export PATH="$HOME/.local/bin:$PATH"
exec uv run python -m lakeside_sentinel --review
