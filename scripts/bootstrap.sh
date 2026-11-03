#!/usr/bin/env bash
# FraudShield Bash Bootstrap Wrapper
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

PY_BIN="python3"
if [ -f ".venv/bin/python" ]; then
    PY_BIN=".venv/bin/python"
fi

$PY_BIN scripts/bootstrap.py "$@"

