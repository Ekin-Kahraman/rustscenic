#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "$0")" && pwd)
PYTHON_BIN=${RUSTSCENIC_PYTHON:-"$ROOT/../rustscenic-recovery/work/reference/bin/python"}
exec "$PYTHON_BIN" "$ROOT/orchestrate.py" collect "${RUSTSCENIC_WORK:-$ROOT/work}"
