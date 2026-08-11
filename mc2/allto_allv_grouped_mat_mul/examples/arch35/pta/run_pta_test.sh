#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
PTA_LIBRARY=${PTA_LIBRARY:-$SCRIPT_DIR/build/pta/liballto_allv_grouped_mat_mul_shmem_pta.so}
PTA_TIMEOUT_SECONDS=${PTA_TIMEOUT_SECONDS:-600}

if [[ ! -f "$PTA_LIBRARY" ]]; then
    echo "Missing PTA library: $PTA_LIBRARY" >&2
    echo "Run $SCRIPT_DIR/build_pta.sh first or set PTA_LIBRARY." >&2
    exit 1
fi
if ! command -v timeout >/dev/null 2>&1; then
    echo "GNU timeout is required to bound SHMEM collective failures." >&2
    exit 1
fi

timeout --signal=TERM --kill-after=30 "${PTA_TIMEOUT_SECONDS}s" \
    "$PYTHON_BIN" "$SCRIPT_DIR/test_allto_allv_grouped_mat_mul_shmem.py" \
    --library "$PTA_LIBRARY" "$@"
