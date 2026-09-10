#!/usr/bin/env bash
# Run from the TTK root so CANN and TTK use the same kernel_meta directory.
#
# By default all cases in the CSV are executed.
# Set TTK_TESTCASE=<case_name> to run a single case.
set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OP_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"


TTK_ROOT=/tmp/ttk_root


CASE_FILE="$OP_DIR/tests/st/arch35/ttk_kernel_cla_gate_quant_general.csv"
RESULT_FILE="$OP_DIR/tests/st/arch35/cla_gate_result.csv"

export PYTHONPATH="$TTK_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$TTK_ROOT"

TESTCASE_ARGS=()
if [ -n "${TTK_TESTCASE:-}" ]; then
    TESTCASE_ARGS=(-t "$TTK_TESTCASE")
fi

python3 -m ttk kernel \
    -i "$CASE_FILE" \
    "${TESTCASE_ARGS[@]}" \
    -o "$RESULT_FILE" \
    --plugin "${TTK_GOLDEN_PLUGIN:-$OP_DIR/tests/assets/golden_npu.py}" \
    --run=10 \
    -d=true \
    "$@"
