#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd -- "${PROJECT_DIR}/../../.." && pwd)
export CATCCOS_SOURCE_PATH=${CATCCOS_SOURCE_PATH:-"${REPO_ROOT}/3rdparty/catccos"}
export SHMEM_HOME_PATH=${SHMEM_HOME_PATH:-"${CATCCOS_SOURCE_PATH}/3rdparty/shmem/install"}

if [[ ! -f "${SHMEM_HOME_PATH}/shmem/lib/libshmem.so" ]]; then
    echo "error: SHMEM UDMA library was not found; run scripts/prepare_dependencies.sh first" >&2
    exit 1
fi

cd "${PROJECT_DIR}"
python3 -m build --wheel --no-isolation
echo "Wheel written under ${PROJECT_DIR}/dist"
