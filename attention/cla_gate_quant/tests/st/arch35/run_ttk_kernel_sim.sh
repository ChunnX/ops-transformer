#!/usr/bin/env bash
# TTK NPUSim simulation for the CLA gate forward TND case.
#
# Usage:
#   source /home/g00940185/work/CANN9.2.0/cann-9.2.0/set_env.sh
#   bash attention/cla_gate_quant/tests/st/arch35/run_ttk_kernel_sim.sh
#
# By default only the fp16 + scale_alg=1 case is executed.
# Override with: TTK_TESTCASE=ClaGateQuant_bf16_cublas_random
# Additional TTK args can still be appended after the script name.
set -euo pipefail

TESTCASE="${TTK_TESTCASE:-ClaGateQuant_fp16_cublas_random}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OP_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# TTK source root; can be overridden with TTK_ROOT=/path/to/ops-test-kit.
TTK_ROOT="${TTK_ROOT:-/tmp/ttk_root}"
if [ ! -d "$TTK_ROOT" ]; then
    TTK_ROOT=/home/g00940185/ops-test-kit
fi

CASE_DIR="$OP_DIR/tests/st/arch35"
CASE_FILE="$CASE_DIR/ttk_kernel_cla_gate_quant.csv"
RESULT_FILE="$CASE_DIR/cla_gate_sim_result.csv"
SIM_OUTPUT_DIR="$CASE_DIR/sim_output"

# TTK's npusim backend calls $ASCEND_TOOLKIT_HOME/bin/cannsim. If the CANN
# environment is not sourced yet, source the fixed toolkit path used here.
if ! command -v cannsim >/dev/null 2>&1; then
    if [ -f /home/g00940185/work/CANN9.2.0/cann-9.2.0/set_env.sh ]; then
        # shellcheck disable=SC1091
        source /home/g00940185/work/CANN9.2.0/cann-9.2.0/set_env.sh
    fi
fi

export PYTHONPATH="$TTK_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$TTK_ROOT"

python3 -m ttk kernel \
    --backend npusim \
    -i "$CASE_FILE" \
    -t "$TESTCASE" \
    -o "$RESULT_FILE" \
    --plugin "${TTK_GOLDEN_PLUGIN:-$OP_DIR/tests/assets/golden_small_op.py}" \
    --sim-soc Ascend950 \
    --sim-cores 0 \
    --sim-output "$SIM_OUTPUT_DIR" \
    --sim-report \
    --warmup=false \
    --proc-timeout 3600 \
    -d=true \
    "$@"
