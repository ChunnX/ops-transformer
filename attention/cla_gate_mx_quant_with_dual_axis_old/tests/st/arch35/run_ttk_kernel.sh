#!/usr/bin/env bash
# Run from the TTK root so CANN and TTK use the same kernel_meta directory.
#
# By default all cases in the CSV are executed.
# Set TTK_TESTCASE=<case_name> to run a single case.
set -euo pipefail

OPS_NN_ROOT=/home/g00940185/work/ops-nn
TTK_ROOT=/home/g00940185/ops-test-kit
CASE_FILE="$OPS_NN_ROOT/quant/cla_gate_mx_quant_with_dual_axis/tests/st/arch35/ttk_kernel_cla_gate_mx_quant_with_dual_axis.csv"
RESULT_FILE="$OPS_NN_ROOT/quant/cla_gate_mx_quant_with_dual_axis/tests/st/arch35/cla_gate_result.csv"

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
    --plugin "${TTK_GOLDEN_PLUGIN:-$OPS_NN_ROOT/quant/cla_gate_mx_quant_with_dual_axis/tests/assets/golden_npu.py}" \
    --warmup=false \
    -d=true \
    "$@"
