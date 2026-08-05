#!/usr/bin/env bash
set -e

ROOT=/home/matmul/z00933385/ops-transformer
GEN_DIR=${ROOT}/build/binary/ascend950/gen
OP_IMPL=${ROOT}/build/binary/ascend950/src/grouped_matmul_apt/GroupedMatmul.py
COMPILE_SH=${GEN_DIR}/GroupedMatmul-grouped_matmul_apt-155.sh
STOP_FILE=${GEN_DIR}/compile_stop.flag
OUT_DIR=/usr/local/Ascend/cann/opp/vendors/s8s4test_transformer/op_impl/ai_core/tbe/kernel/ascend950/grouped_matmul_apt
KERNEL=GroupedMatmul_d68006bdc9220e3c49bb6c18ad1d8c97
STAGE_DIR=$(mktemp -d "${GEN_DIR}/.${KERNEL}.XXXXXX")

cleanup() {
    rm -rf "${STAGE_DIR}"
}
trap cleanup EXIT

run_privileged() {
    if [[ ${EUID} -eq 0 ]]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo -E "$@"
    else
        echo "root permission is required to deploy into ${OUT_DIR}" >&2
        exit 1
    fi
}

source /usr/local/Ascend/cann/set_env.sh

export ASCEND_CUSTOM_OPP_PATH=${ROOT}/build/custom
export HI_PYTHON=python3
export TILINGKEY_PAR_COMPILE=1
export BIN_FILENAME_HASHED=1

echo "kernel source: ${ROOT}/build/binary/ascend950/src/grouped_matmul_apt/op_kernel/grouped_matmul_apt.cpp"
echo "ops-tensor headers: /home/matmul/z00933385/ops-tensor/include"
rm -f "${STOP_FILE}"

cd "${GEN_DIR}"
bash "${COMPILE_SH}" "${OP_IMPL}" "${STAGE_DIR}"

test -s "${STAGE_DIR}/${KERNEL}.o"
test -s "${STAGE_DIR}/${KERNEL}.json"
run_privileged install -m 0640 "${STAGE_DIR}/${KERNEL}.o" "${OUT_DIR}/${KERNEL}.o.new"
run_privileged install -m 0640 "${STAGE_DIR}/${KERNEL}.json" "${OUT_DIR}/${KERNEL}.json.new"
run_privileged mv -f "${OUT_DIR}/${KERNEL}.o.new" "${OUT_DIR}/${KERNEL}.o"
run_privileged mv -f "${OUT_DIR}/${KERNEL}.json.new" "${OUT_DIR}/${KERNEL}.json"

ls -l "${OUT_DIR}/${KERNEL}.o" "${OUT_DIR}/${KERNEL}.json"
