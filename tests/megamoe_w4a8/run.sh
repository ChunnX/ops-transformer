#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

export PYTHONPATH="${REPO_ROOT}/torch_extension${PYTHONPATH:+:${PYTHONPATH}}"
export MEGAMOE_ENABLE_TIMER="${MEGAMOE_ENABLE_TIMER:-1}"
export MEGAMOE_TIMER_DIR="${MEGAMOE_TIMER_DIR:-${SCRIPT_DIR}/timer}"

cd "${SCRIPT_DIR}"
python3 -B mega_moe.py
