#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
DEVICE_ID=${DEVICE_ID:-0}

set +u
source /data/wnc/cann/ascend-toolkit/set_env.sh
source "${REPO_ROOT}/install_custom_ops/vendors/custom_transformer/bin/set_env.bash"
source /data/miniconda3/etc/profile.d/conda.sh
set -u
conda activate yyd

export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/data/wnc/cann/cann-9.1.0-beta.1}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions_sparse_flash_mla}

python "${REPO_ROOT}/attention/sparse_flash_mla/tests/pytest/sparse_flash_mla_validation.py" \
  --device "${DEVICE_ID}" --json
