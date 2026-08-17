#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

if (( $# > 1 )); then
    echo "Usage: $0 [visible_device_ids]" >&2
    exit 2
fi
if (( $# == 1 )); then
    export ASCEND_RT_VISIBLE_DEVICES="$1"
fi

export PYTHONPATH="${REPO_ROOT}/torch_extension${PYTHONPATH:+:${PYTHONPATH}}"
export MEGAMOE_ENABLE_TIMER="${MEGAMOE_ENABLE_TIMER:-1}"
export MEGAMOE_TIMER_DIR="${MEGAMOE_TIMER_DIR:-${SCRIPT_DIR}/timer}"

cd "${SCRIPT_DIR}"
python3 -B mega_moe.py
