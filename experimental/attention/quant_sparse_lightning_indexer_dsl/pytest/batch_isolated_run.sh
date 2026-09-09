#!/usr/bin/env bash
set -euo pipefail

PT_PATH=${1:-./pt_path}
DEVICE_ID=${2:-0}
failed=0
for file in "${PT_PATH}"/*.pt; do
  echo "===== $(basename "${file}") ====="
  if ! QSLI_TESTCASE_PATH=${file} QSLI_TESTCASE_DIR=${PT_PATH} QSLI_DEVICE_ID=${DEVICE_ID} \
    python3 -m pytest -rA -s test_quant_sparse_lightning_indexer_batch.py -v -m ci; then
    failed=$((failed + 1))
  fi
done
exit ${failed}
