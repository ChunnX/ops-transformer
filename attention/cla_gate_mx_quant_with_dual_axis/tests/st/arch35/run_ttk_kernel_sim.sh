#!/usr/bin/env bash
# TTK NPUSim simulation for the fixed CLA gate forward case.
#
# Usage:
#   source /home/g00940185/work/CANN9.2.0/cann-9.2.0/set_env.sh
#   bash quant/cla_gate_mx_quant_with_dual_axis/tests/st/arch35/run_ttk_kernel_sim.sh
#
# By default only the second case (fp16 + scale_alg=1) is executed.
# Override with: TTK_TESTCASE=ClaGateMxQuantWithDualAxis_bf16_cublas_random
# Additional TTK args can still be appended after the script name.
set -euo pipefail

TESTCASE="${TTK_TESTCASE:-ClaGateMxQuantWithDualAxis_fp16_cublas_random}"

OPS_NN_ROOT=/home/g00940185/work/ops-nn
TTK_ROOT=/home/g00940185/ops-test-kit
CASE_DIR="$OPS_NN_ROOT/quant/cla_gate_mx_quant_with_dual_axis/tests/st/arch35"
CASE_FILE="$CASE_DIR/ttk_kernel_cla_gate_mx_quant_with_dual_axis.csv"
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
    --plugin "${TTK_GOLDEN_PLUGIN:-$OPS_NN_ROOT/quant/cla_gate_mx_quant_with_dual_axis/tests/assets/golden_small_op.py}" \
    --sim-soc Ascend950 \
    --sim-cores 0 \
    --sim-output "$SIM_OUTPUT_DIR" \
    --sim-report \
    --warmup=false \
    --proc-timeout 3600 \
    -d=true \
    "$@"
