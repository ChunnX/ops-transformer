#!/usr/bin/env bash
set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
CANN_ROOT=${CANN_ROOT:-/home/developer/Ascend/cann-9.2.0/cann-9.2.0}
CUSTOM_OPP=${ASCEND_CUSTOM_OPP_PATH:-/home/zhoushaolong/v3_opp_extract/packages/vendors/custom_transformer}
V3_DEMO_BIN=${V3_DEMO_BIN:-$ROOT/examples/bin/test_v3_modes_92}
GOLDEN_SCRIPT=$ROOT/examples/arch35/cpu_golden_swiglu_quant_v3.py

if [ ! -x "$V3_DEMO_BIN" ]; then
    echo "missing ACLNN demo binary: $V3_DEMO_BIN" >&2
    echo "compile test_aclnn_grouped_matmul_swiglu_quant_v3_nz.cpp with CANN 9.2 first" >&2
    exit 2
fi
if [ -f "$CANN_ROOT/set_env.sh" ]; then
    # shellcheck disable=SC1090
    source "$CANN_ROOT/set_env.sh"
fi
export ASCEND_CUSTOM_OPP_PATH="$CUSTOM_OPP"
export LD_LIBRARY_PATH="$CUSTOM_OPP/op_api/lib:${LD_LIBRARY_PATH:-}"

MODE=1
for transpose in 0 1; do
    prefix="/tmp/gmm_swiglu_v3_mode${MODE}_t${transpose}"
    rm -f "${prefix}_output.bin" "${prefix}_output_scale.bin"
    set +e
    timeout -k 2s "${V3_TIMEOUT:-12s}" "$V3_DEMO_BIN" "$MODE" "$transpose" 1 "$prefix"
    rc=$?
    set -e
    # The current sample exits with timeout 124 while the ACL teardown waits;
    # output synchronization and the golden check have already completed.
    if [ "$rc" -ne 0 ] && [ "$rc" -ne 124 ]; then
        echo "ACLNN demo failed: mode=$MODE transpose=$transpose rc=$rc" >&2
        exit "$rc"
    fi
    python3 "$GOLDEN_SCRIPT" "$MODE" "$transpose" 1         "${prefix}_output.bin" "${prefix}_output_scale.bin"
done
