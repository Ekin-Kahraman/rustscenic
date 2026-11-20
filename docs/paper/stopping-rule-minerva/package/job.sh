#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "$0")" && pwd)
JOB=$1
RECOVERY=$2
WORK=$3
PYTHON_BIN=$4
RESUME=${5:-}
if [[ "$JOB" == collect ]]; then
 exec "$PYTHON_BIN" "$ROOT/orchestrate.py" collect "$WORK"
fi
exec "$PYTHON_BIN" "$ROOT/orchestrate.py" worker "$JOB" "$RECOVERY" "$WORK" ${RESUME:+--resume}
