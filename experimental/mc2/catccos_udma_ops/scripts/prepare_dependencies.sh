#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
set -euo pipefail

BACKEND=${CATCCOS_BACKEND:-udma}
WITH_TUNING=${CATCCOS_MTE_WITH_TUNING:-0}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            [[ $# -ge 2 ]] || { echo "error: --backend requires udma or mte" >&2; exit 2; }
            BACKEND=$2; shift 2 ;;
        --with-tuning) WITH_TUNING=1; shift ;;
        --help|-h) echo "Usage: $0 [--backend udma|mte] [--with-tuning]"; exit 0 ;;
        *) echo "error: unknown argument $1" >&2; exit 2 ;;
    esac
done
[[ "$BACKEND" == "udma" || "$BACKEND" == "mte" ]] || { echo "error: invalid backend $BACKEND" >&2; exit 2; }
case "${WITH_TUNING,,}" in
    1|on|true|yes) WITH_TUNING=1 ;;
    ""|0|off|false|no) WITH_TUNING=0 ;;
    *) echo "error: CATCCOS_MTE_WITH_TUNING must be boolean" >&2; exit 2 ;;
esac
if [[ "$WITH_TUNING" != "0" && "$BACKEND" != "mte" ]]; then
    echo "error: --with-tuning requires --backend mte" >&2; exit 2
fi
export CATCCOS_BACKEND=$BACKEND
export CATCCOS_MTE_WITH_TUNING=$WITH_TUNING

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd -- "${PROJECT_DIR}/../../.." && pwd)
CATCCOS_REPOSITORY=https://gitcode.com/cann/catccos.git
CATCCOS_COMMIT=878c2e0a504be8c49c03b5e1157a7c714a628e29
CATCCOS_DIR=${CATCCOS_SOURCE_PATH:-"${REPO_ROOT}/3rdparty/catccos"}
SHMEM_REPOSITORY=https://gitcode.com/cann/shmem.git
# Official release/v1.6.0 maintenance snapshot (2026-09-04).
SHMEM_COMMIT=7fa19662da0093587d9a332e740b91d6c929cdcf
if [[ "$BACKEND" == "mte" ]]; then
    SHMEM_COMMIT=aee374a396b389626f68fab74ec8c555e18978a7
    SHMEM_DIR=${SHMEM_SOURCE_PATH:-"${REPO_ROOT}/3rdparty/shmem-mte"}
else
    SHMEM_DIR=${SHMEM_SOURCE_PATH:-"${REPO_ROOT}/3rdparty/shmem"}
fi

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

# CatCCOS keeps its complete source tree. The Wheel deliberately consumes the
# independent SHMEM checkout below instead of this revision's SHMEM gitlink.
git -C "${CATCCOS_DIR}" submodule update --init --recursive 3rdparty/catlass 3rdparty/shmem

CLONED_SHMEM=0
if [[ ! -e "${SHMEM_DIR}/.git" ]]; then
    if [[ -e "${SHMEM_DIR}" ]]; then
        echo "error: ${SHMEM_DIR} exists but is not a Git checkout" >&2
        exit 1
    fi
    git clone "${SHMEM_REPOSITORY}" "${SHMEM_DIR}"
    CLONED_SHMEM=1
fi

# Keep user edits intact and never label a modified checkout as the official
# binary. Gitignored build/install products do not make the checkout dirty.
SHMEM_WORKTREE_STATUS=$(git -C "${SHMEM_DIR}" status --porcelain --untracked-files=normal)
if [[ -n "${SHMEM_WORKTREE_STATUS}" ]]; then
    echo "error: SHMEM checkout has local changes; commit or move them before preparing dependencies" >&2
    printf '%s\n' "${SHMEM_WORKTREE_STATUS}" >&2
    exit 1
fi
ACTUAL_SHMEM_COMMIT=$(git -C "${SHMEM_DIR}" rev-parse HEAD)
if [[ "${ACTUAL_SHMEM_COMMIT}" != "${SHMEM_COMMIT}" ]]; then
    if [[ "${BACKEND:-udma}" == "mte" && "${CLONED_SHMEM:-0}" == "0" ]]; then
        echo "error: existing SHMEM source is not the MTE baseline ${SHMEM_COMMIT}; use a separate checkout" >&2
        exit 1
    fi
    if ! git -C "${SHMEM_DIR}" cat-file -e "${SHMEM_COMMIT}^{commit}" 2>/dev/null; then
        git -C "${SHMEM_DIR}" fetch --no-tags "${SHMEM_REPOSITORY}" "${SHMEM_COMMIT}"
    fi
    git -C "${SHMEM_DIR}" checkout --detach "${SHMEM_COMMIT}"
fi

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "error: ASCEND_HOME_PATH is not set; source the CANN toolkit set_env.sh first" >&2
    exit 1
fi

TORCH_ABI=$(python3 -c 'import torch; f=getattr(torch, "compiled_with_cxx11_abi", None); print(int(f() if f else torch._C._GLIBCXX_USE_CXX11_ABI))')
if [[ "${TORCH_ABI}" == "1" ]]; then
    ABI_ARG=-use_cxx11_abi1
else
    ABI_ARG=-use_cxx11_abi0
fi
# Invalidate the previous install before rebuilding, including on build failure.
rm -f "${SHMEM_DIR}/install/shmem/.ops_transformer_source_commit"
VERSION=$(tr -d '[:space:]' < "${SHMEM_DIR}/VERSION") \
    bash "${SHMEM_DIR}/scripts/build.sh" -soc_type Ascend950 "${ABI_ARG}"
for SHMEM_LIBRARY in libshmem.so libshmem_utils.so; do
    if [[ ! -f "${SHMEM_DIR}/install/shmem/lib/${SHMEM_LIBRARY}" ]]; then
        echo "error: SHMEM build did not install ${SHMEM_LIBRARY}; no provenance marker was written" >&2
        exit 1
    fi
done
printf '%s\n' "${SHMEM_COMMIT}" > "${SHMEM_DIR}/install/shmem/.ops_transformer_source_commit"

echo "CatCCOS: ${CATCCOS_DIR}"
echo "SHMEM source: ${SHMEM_DIR} (${SHMEM_COMMIT})"
echo "SHMEM_HOME_PATH: ${SHMEM_DIR}/install"
echo "Run: export CATCCOS_SOURCE_PATH='${CATCCOS_DIR}'"
echo "Run: export SHMEM_SOURCE_PATH='${SHMEM_DIR}'"
echo "Run: export SHMEM_HOME_PATH='${SHMEM_DIR}/install'"
