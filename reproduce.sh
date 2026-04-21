#!/usr/bin/env bash
# reproduce.sh — Reproduce "From Personas to Talks" (EMNLP 2025, Wu et al.)
# Tested on Python 3.10+. Set API keys before running.
#
# Usage:
#   export GEMINI_API_KEY="your-key"
#   export ANTHROPIC_API_KEY="your-key"   # optional fallback
#   bash reproduce.sh [--rq 1|2|3|all]

set -euo pipefail

RQ="${1:-all}"
if [[ "$1" == "--rq" && -n "${2:-}" ]]; then
    RQ="$2"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 1. Environment ────────────────────────────────────────────────────────────
if [[ -z "${GEMINI_API_KEY:-}" && -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ERROR: Set at least one of GEMINI_API_KEY or ANTHROPIC_API_KEY." >&2
    exit 1
fi

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found." >&2
    exit 1
fi

PYTHON="python3"

# ── 2. Virtual environment ────────────────────────────────────────────────────
if [[ ! -d ".venv" ]]; then
    echo "Creating virtual environment..."
    "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# ── 3. Run pipeline ───────────────────────────────────────────────────────────
echo "Running RQ=${RQ}..."
"$PYTHON" main.py --rq "$RQ"

echo "Done. Results are in outputs/."
