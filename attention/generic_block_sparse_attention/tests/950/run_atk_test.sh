#!/bin/bash
# Arch35 / Ascend 950 ATK entry for normal and Metadata-v8 FD coverage.

set -eo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

TEST_MODE=${TEST_MODE:-normal}
DEVICE_ID=${DEVICE_ID:-1}
ATK_WHITE_LIST=${ATK_WHITE_LIST:-}
RUN_NONCONTIG=${RUN_NONCONTIG:-0}
ATK_LOOP_NUMS=${ATK_LOOP_NUMS:-1}
ATK_UNCON_WHITE_LIST=${ATK_UNCON_WHITE_LIST:-'[178]'}
DEFAULT_CUSTOM_OPP_ENV=/home/npu_user1/l00937279/package/package_custom_personal/wjc/msa_new_dev_llw2_c36ad0218c/vendors/custom_transformer/bin/set_env.bash
CUSTOM_OPP_ENV=${CUSTOM_OPP_ENV:-${DEFAULT_CUSTOM_OPP_ENV}}

if [[ "${TEST_MODE}" != "normal" && "${TEST_MODE}" != "fd" ]]; then
    echo "TEST_MODE must be normal or fd, got: ${TEST_MODE}" >&2
    exit 2
fi
if [[ ! "${DEVICE_ID}" =~ ^[0-9]+$ ]]; then
    echo "DEVICE_ID must be a non-negative integer, got: ${DEVICE_ID}" >&2
    exit 2
fi
if [[ ! "${ATK_LOOP_NUMS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ATK_LOOP_NUMS must be a positive integer, got: ${ATK_LOOP_NUMS}" >&2
    exit 2
fi

if [[ -f /usr/local/Ascend/cann/set_env.sh ]]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/cann/set_env.sh
elif [[ -f /usr/local/Ascend/cann-9.1.0/set_env.sh ]]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/cann-9.1.0/set_env.sh
fi

# Preserve the already-installed package as the default.  A newly installed
# package can be selected without editing this script by exporting either
# CUSTOM_OPP_PATH=<install>/vendors/custom_transformer or CUSTOM_OPP_ENV=...
if [[ -n "${CUSTOM_OPP_PATH:-}" ]]; then
    export ASCEND_CUSTOM_OPP_PATH=${CUSTOM_OPP_PATH}
elif [[ -f "${CUSTOM_OPP_ENV}" ]]; then
    # shellcheck disable=SC1090
    source "${CUSTOM_OPP_ENV}"
else
    echo "custom OPP environment not found: ${CUSTOM_OPP_ENV}" >&2
    exit 2
fi

# ATK_CUSTOM_OPP_PATH expects the ACLNN shared library itself.  Custom OPP
# set_env scripts expose ASCEND_CUSTOM_OPP_PATH at the vendor directory, while
# ATK otherwise treats it as the parent of vendors/ and fails to locate ACLNN.
if [[ -z "${ATK_CUSTOM_OPP_PATH:-}" ]]; then
    IFS=':' read -r -a custom_opp_roots <<< "${ASCEND_CUSTOM_OPP_PATH:-}"
    for custom_opp_root in "${custom_opp_roots[@]}"; do
        [[ -z "${custom_opp_root}" ]] && continue
        if [[ -f "${custom_opp_root}/op_api/lib/libcust_opapi.so" ]]; then
            export ATK_CUSTOM_OPP_PATH="${custom_opp_root}/op_api/lib/libcust_opapi.so"
            break
        fi
        if [[ -f "${custom_opp_root}/vendors/custom_transformer/op_api/lib/libcust_opapi.so" ]]; then
            export ATK_CUSTOM_OPP_PATH="${custom_opp_root}/vendors/custom_transformer/op_api/lib/libcust_opapi.so"
            break
        fi
    done
fi
if [[ ! -f "${ATK_CUSTOM_OPP_PATH:-}" ]]; then
    echo "ATK ACLNN library not found; set ATK_CUSTOM_OPP_PATH to libcust_opapi.so" >&2
    exit 2
fi

# Expose exactly one physical card so both ACL and Torch NPU consistently use
# logical device 0.  Mixing a physical ID in ACL with Torch's default logical
# device can block during PyAclnn input initialization.
export ASCEND_RT_VISIBLE_DEVICES=${DEVICE_ID}
ATK_LOGICAL_DEVICE_ID=0

PYTHON_BIN=${PYTHON_BIN:-/usr/local/python3.12.13/bin/python3.12}
if command -v atk >/dev/null 2>&1; then
    ATK_CMD=(atk)
elif [[ -x "${PYTHON_BIN}" ]] && "${PYTHON_BIN}" -c 'import atk' >/dev/null 2>&1; then
    ATK_CMD=("${PYTHON_BIN}" -m atk)
elif [[ -d /home/npu_user1/l00937279/ATK/atk ]]; then
    export PYTHONPATH=/home/npu_user1/l00937279/ATK:${PYTHONPATH:-}
    ATK_CMD=("${PYTHON_BIN}" -m atk)
else
    echo "ATK is not installed and /home/npu_user1/l00937279/ATK is unavailable" >&2
    exit 2
fi

CASE_FILE="all_op_generic_block_sparse_attention_${TEST_MODE}.json"
RUNTIME_NODES=".nodes_device_${DEVICE_ID}.yaml"
sed -E "s/devices: \[[0-9]+\]/devices: [${ATK_LOGICAL_DEVICE_ID}]/g" nodes.yaml > "${RUNTIME_NODES}"

function atk_task_gen() {
    echo "[ATK] generate TEST_MODE=${TEST_MODE}"
    "${ATK_CMD[@]}" case -f op_generic_block_sparse_attention.yaml \
        -p generator_genericblocksparseattention.py
    local generated_case=result/op_generic_block_sparse_attention/json/all_op_generic_block_sparse_attention.json
    if [[ ! -f "${generated_case}" ]]; then
        echo "ATK did not generate the expected Case file: ${generated_case}" >&2
        exit 1
    fi
    cp -f "${generated_case}" "${CASE_FILE}"
    echo "[ATK] generated ${CASE_FILE}"
}

function run_one_plugin() {
    local plugin=$1
    local white_list=${2:-}
    local args=(task -c "${CASE_FILE}" -n "${RUNTIME_NODES}" -p "${plugin}" -to 9999 --loop_nums "${ATK_LOOP_NUMS}")
    if [[ -n "${white_list}" ]]; then
        args+=(-wl "${white_list}")
    fi
    echo "[ATK] run TEST_MODE=${TEST_MODE}, physicalDevice=${DEVICE_ID}, logicalDevice=${ATK_LOGICAL_DEVICE_ID}, plugin=${plugin}, whiteList=${white_list:-all}"
    "${ATK_CMD[@]}" "${args[@]}"
}

function atk_execute() {
    if [[ ! -f "${CASE_FILE}" ]]; then
        echo "case file does not exist: ${CASE_FILE}; run generate first" >&2
        exit 2
    fi
    run_one_plugin aclnn_genericblocksparseattention.py "${ATK_WHITE_LIST}"
    if [[ "${TEST_MODE}" == "fd" && "${RUN_NONCONTIG}" == "1" ]]; then
        run_one_plugin aclnn_genericblocksparseattention_uncon.py "${ATK_UNCON_WHITE_LIST}"
    fi
}

function usage() {
    cat <<'EOF'
Usage: TEST_MODE=normal|fd [DEVICE_ID=N] bash ./run_atk_test.sh all|generate|run

Optional:
  ATK_WHITE_LIST='[0,1]'       Run selected generated Case IDs.
  RUN_NONCONTIG=1              In FD mode, also run dim0-strided KV coverage.
  ATK_LOOP_NUMS=100            Repeat each selected Case for stability coverage.
  ATK_UNCON_WHITE_LIST='[178]' Select the non-contiguous KV Case IDs.
  CUSTOM_OPP_PATH=<vendor-dir> Select a newly installed custom OPP package.
EOF
}

case "${1:-all}" in
    all)
        atk_task_gen
        atk_execute
        ;;
    generate)
        atk_task_gen
        ;;
    run)
        atk_execute
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
