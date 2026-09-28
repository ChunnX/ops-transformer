#!/usr/bin/env bash
# End-to-end: real host metadata -> linked sparse_flash_mla CAMODEL case.
#
# Steps:
#   1. build the host driver (host g++ only; no AscendC / simulator)
#   2. build the MLA cases without metadata (so the int32 input .npy exist)
#   3. run the real SparseFlashMlaMetadata kernel once per delay case
#   4. rebuild the MLA cases with the per-delay metadata tensors
#   5. run the linked cases in CAMODEL (unless PRO0921_SKIP_SIM=1)
#
# Steps 1 and 5 do real work; run the whole script under the shared
# local-resource-runner lock.  Everything here is host-only until step 5.
#
# Usage:
#   bash gen_smla_metadata.sh [mla-root]        # default /tmp/pro0921_latent_mla
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_DIR="$(cd "${HERE}/.." && pwd)"
MLA_ROOT="${1:-/tmp/pro0921_latent_mla}"
META_DIR="${PRO0921_META_DIR:-${MLA_ROOT}/meta}"
PYTHON_BIN="${PRO0921_PYTHON:-/home/jinxh/.pyenv/versions/3.11.13/bin/python3}"
SKIP_SIM="${PRO0921_SKIP_SIM:-0}"
SELECTOR="${PRO0921_SMLA_SELECTOR:-sparse_mla}"

echo "[1/5] build host metadata driver"
bash "${HERE}/build_metadata_host.sh"

echo "[2/5] build MLA cases (inputs + CPU golden, metadata still missing)"
"${PYTHON_BIN}" "${SMOKE_DIR}/build_smla_linked_cases.py" --out "${MLA_ROOT}"

echo "[3/5] run the real AICPU metadata kernel for each delay case"
"${PYTHON_BIN}" "${HERE}/run_metadata_host.py" --mla-root "${MLA_ROOT}" --out "${META_DIR}"

echo "[4/5] rebuild the MLA cases with per-delay metadata"
"${PYTHON_BIN}" "${SMOKE_DIR}/build_smla_linked_cases.py" --out "${MLA_ROOT}" \
  --metadata-dir "${META_DIR}"

if [[ "${SKIP_SIM}" == "1" ]]; then
  echo "[5/5] skipped (PRO0921_SKIP_SIM=1); cases ready under ${MLA_ROOT}"
  exit 0
fi

echo "[5/5] run the linked CAMODEL case(s)"
bash "${SMOKE_DIR}/run_tiny_sim.sh" "${MLA_ROOT}" "${SELECTOR}"
