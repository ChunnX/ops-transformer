#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd -- "${PROJECT_DIR}/../../.." && pwd)
export CATCCOS_SOURCE_PATH=${CATCCOS_SOURCE_PATH:-"${REPO_ROOT}/3rdparty/catccos"}
export SHMEM_SOURCE_PATH=${SHMEM_SOURCE_PATH:-"${REPO_ROOT}/3rdparty/shmem"}
export SHMEM_HOME_PATH=${SHMEM_HOME_PATH:-"${SHMEM_SOURCE_PATH}/install"}
EXPECTED_SHMEM_COMMIT=aee374a396b389626f68fab74ec8c555e18978a7

if [[ ! -f "${SHMEM_SOURCE_PATH}/include/shmem.h" ||
      ! -f "${SHMEM_HOME_PATH}/shmem/lib/libshmem.so" ||
      ! -f "${SHMEM_HOME_PATH}/shmem/.ops_transformer_source_commit" ]]; then
    echo "error: SHMEM UDMA library was not found; run scripts/prepare_dependencies.sh first" >&2
    exit 1
fi
ACTUAL_SHMEM_COMMIT=$(git -C "${SHMEM_SOURCE_PATH}" rev-parse HEAD 2>/dev/null || true)
BUILT_SHMEM_COMMIT=$(tr -d '[:space:]' < "${SHMEM_HOME_PATH}/shmem/.ops_transformer_source_commit")
if [[ "${ACTUAL_SHMEM_COMMIT}" != "${EXPECTED_SHMEM_COMMIT}" ||
      "${BUILT_SHMEM_COMMIT}" != "${EXPECTED_SHMEM_COMMIT}" ]]; then
    echo "error: SHMEM source or library is not the pinned official v1.6.0 commit" >&2
    echo "expected: ${EXPECTED_SHMEM_COMMIT}" >&2
    echo "source:   ${ACTUAL_SHMEM_COMMIT:-missing}" >&2
    echo "library:  ${BUILT_SHMEM_COMMIT:-missing}" >&2
    echo "run scripts/prepare_dependencies.sh to rebuild the dependency" >&2
    exit 1
fi

cd "${PROJECT_DIR}"
python3 -m build --wheel --no-isolation
echo "Wheel written under ${PROJECT_DIR}/dist"
