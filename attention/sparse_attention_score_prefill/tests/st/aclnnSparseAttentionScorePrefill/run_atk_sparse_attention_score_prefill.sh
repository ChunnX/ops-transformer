#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASE_FILE="${SCRIPT_DIR}/atk_aclnnSparseAttentionScorePrefill.json"
EXECUTOR_FILE="${SCRIPT_DIR}/executor_aclnnSparseAttentionScorePrefill.py"
NODE_FILE="${SCRIPT_DIR}/nodes_a2.yaml"

# ATK's pyaclnn loader uses this library-specific variable.  Derive it from
# the active custom OPP when the caller has not set it explicitly.
if [[ -z "${ATK_CUSTOM_OPP_PATH:-}" && -n "${ASCEND_CUSTOM_OPP_PATH:-}" ]]; then
  _custom_opapi="${ASCEND_CUSTOM_OPP_PATH}/vendors/custom_transformer/op_api/lib/libcust_opapi.so"
  if [[ -f "${_custom_opapi}" ]]; then
    export ATK_CUSTOM_OPP_PATH="${_custom_opapi}"
  fi
fi

python "${SCRIPT_DIR}/generate_atk_sparse_attention_score_prefill.py" \
  --output "${CASE_FILE}" "$@"

atk task \
  -n "${NODE_FILE}" \
  -p "${EXECUTOR_FILE}" \
  -c "${CASE_FILE}" \
  -tk accuracy \
  -sp \
  -to 180
