#!/usr/bin/env bash
# Sourced only by the repository-local diagnostic entry points.
set -Eeuo pipefail
SLIG_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SLIG_WORK_DIR="${SLIG_WORK_DIR:-${SLIG_REPO_ROOT}/.klloss_repro}"
SLIG_PYTHON="${SLIG_PYTHON:-python3}"
SLIG_CANN_PATH="${SLIG_CANN_PATH:-${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.1.0}}"
mkdir -p "${SLIG_WORK_DIR}"
SLIG_WORK_DIR="$(cd -- "${SLIG_WORK_DIR}" && pwd)"
export SLIG_REPO_ROOT SLIG_WORK_DIR SLIG_PYTHON SLIG_CANN_PATH
[[ -f "${SLIG_CANN_PATH}/set_env.sh" ]] || { echo "Set SLIG_CANN_PATH to CANN 9.1.0" >&2; return 2; }
set +u
source "${SLIG_CANN_PATH}/set_env.sh"
set -u
export PYTHONUNBUFFERED=1
export MAX_JOBS="${SLIG_EXTENSION_JOBS:-2}"
# Never use an unrelated globally installed OPP or torch extension as the test artifact.
unset ASCEND_CUSTOM_OPP_PATH
export TORCH_EXTENSIONS_DIR="${SLIG_WORK_DIR}/torch_extensions"
mkdir -p "${SLIG_WORK_DIR}/logs"
