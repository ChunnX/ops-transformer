#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd -- "${PROJECT_DIR}/../../.." && pwd)
CATCCOS_REPOSITORY=https://gitcode.com/cann/catccos.git
CATCCOS_COMMIT=878c2e0a504be8c49c03b5e1157a7c714a628e29
CATCCOS_DIR=${CATCCOS_SOURCE_PATH:-"${REPO_ROOT}/3rdparty/catccos"}

if [[ ! -d "${CATCCOS_DIR}/.git" ]]; then
    if [[ -e "${CATCCOS_DIR}" ]]; then
        echo "error: ${CATCCOS_DIR} exists but is not a Git checkout" >&2
        exit 1
    fi
    git clone "${CATCCOS_REPOSITORY}" "${CATCCOS_DIR}"
    git -C "${CATCCOS_DIR}" checkout --detach "${CATCCOS_COMMIT}"
else
    ACTUAL_COMMIT=$(git -C "${CATCCOS_DIR}" rev-parse HEAD)
    if [[ "${ACTUAL_COMMIT}" != "${CATCCOS_COMMIT}" ]]; then
        echo "error: CATCCOS_SOURCE_PATH is at ${ACTUAL_COMMIT}, expected ${CATCCOS_COMMIT}" >&2
        echo "Use a clean checkout at the pinned commit or unset CATCCOS_SOURCE_PATH." >&2
        exit 1
    fi
fi

git -C "${CATCCOS_DIR}" submodule update --init --recursive 3rdparty/catlass 3rdparty/shmem

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "error: ASCEND_HOME_PATH is not set; source the CANN toolkit set_env.sh first" >&2
    exit 1
fi

SHMEM_DIR="${CATCCOS_DIR}/3rdparty/shmem"
TORCH_ABI=$(python3 -c 'import torch; f=getattr(torch, "compiled_with_cxx11_abi", None); print(int(f() if f else torch._C._GLIBCXX_USE_CXX11_ABI))')
if [[ "${TORCH_ABI}" == "1" ]]; then
    ABI_ARG=-use_cxx11_abi1
else
    ABI_ARG=-use_cxx11_abi0
fi
bash "${SHMEM_DIR}/scripts/build.sh" -soc_type Ascend950 "${ABI_ARG}"

echo "CatCCOS: ${CATCCOS_DIR}"
echo "SHMEM_HOME_PATH: ${SHMEM_DIR}/install"
echo "Run: export CATCCOS_SOURCE_PATH='${CATCCOS_DIR}'"
echo "Run: export SHMEM_HOME_PATH='${SHMEM_DIR}/install'"
