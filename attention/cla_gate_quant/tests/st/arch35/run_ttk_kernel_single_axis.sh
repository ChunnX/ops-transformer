#!/usr/bin/env bash
# Run the single-axis (quantMode=0) TTK cases on the real-device backend.
#
# Usage:
#   source /home/g00940185/work/CANN9.2.0/cann-9.2.0/set_env.sh
#   bash tests/st/arch35/run_ttk_kernel_single_axis.sh
#
# Override a single case:
#   TTK_TESTCASE=ClaGateQuantSingleAxis_bf16_T13_N11_D128_e4m3fn_alg0_rint \
#       bash tests/st/arch35/run_ttk_kernel_single_axis.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OP_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

TTK_ROOT="${TTK_ROOT:-//home/g00940185/ops-test-kit}"

CASE_FILE="$OP_DIR/tests/st/arch35/ttk_kernel_cla_gate_quant_single_axis.csv"
RESULT_FILE="$OP_DIR/tests/st/arch35/cla_gate_single_axis_result.csv"

export PYTHONPATH="$TTK_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$TTK_ROOT"

TESTCASE_ARGS=()
if [ -n "${TTK_TESTCASE:-}" ]; then
    TESTCASE_ARGS=(-t "$TTK_TESTCASE")
fi

# If the user passes --device-whitelist=6, pass the same device to the NPU
# golden through a dedicated variable without changing TTK's device enumeration.
for arg in "$@"; do
    case "$arg" in
        --device-whitelist=*)
            export TTK_GOLDEN_DEVICE_ID="${arg#*=}"
            ;;
    esac
done

python3 -m ttk kernel \
    -i "$CASE_FILE" \
    "${TESTCASE_ARGS[@]}" \
    -o "$RESULT_FILE" \
    --plugin "${TTK_GOLDEN_PLUGIN:-$OP_DIR/tests/assets/golden_npu.py}" \
    --run=10 \
    -d=true \
    "$@"
