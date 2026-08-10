#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
export UDMA_MASTER_PORT=${UDMA_MASTER_PORT:-29601}
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

torchrun --standalone --nproc-per-node="${NPROC_PER_NODE}" \
    "${PROJECT_DIR}/tests/test_all_to_allv_grouped_mat_mul.py" "$@"
