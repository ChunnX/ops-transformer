#!/usr/bin/env bash
# Minimal JIT host build for the formal Pro 0921 latent operators.
#
# Run this ONLY inside the shared local-resource-runner lock. It builds the
# formal (non-experimental) attention operators with --jit so the package
# carries the host tiling library and the AscendC kernel sources; the kernel
# binary is then JIT-compiled by the simulator/toolchain on first use instead
# of shipping every SoC binary. That is why no --experimental and no full
# business build is used.
#
# Usage:
#   bash compile_formal_ops.sh [ops-transformer-checkout]
#
# Environment overrides:
#   CANN_ROOT        CANN toolkit root (default /home/jinxh/Ascend_0722/cann-9.2.0)
#   OPS_THIRD_PARTY  independent third-party dependency copy (default
#                    <task-root>/build-deps, NEVER the shared third_party)
#   VENDOR_NAME      independent vendor name (default pro0921latent)
#   OPS_LIST         operators to build
#
# Note: build.sh installs the vendor as "<VENDOR_NAME>_transformer", so
# --vendor_name=pro0921latent yields vendors/pro0921latent_transformer.  This
# script and install_formal_opp.sh / run_tiny_sim.sh agree on that name; do not
# rename it in only one place.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# tests/pro0921_latent_smoke -> attention/compressor/tests/pro0921_latent_smoke
TASK_ROOT="$(cd "${HERE}/../../../../.." && pwd)"
CHECKOUT="${1:-${HERE}/../../../..}"
CANN_ROOT="${CANN_ROOT:-/home/jinxh/Ascend_0722/cann-9.2.0}"
OPS_THIRD_PARTY="${OPS_THIRD_PARTY:-${TASK_ROOT}/build-deps}"
VENDOR_NAME="pro0921latent"
OPS_LIST="${OPS_LIST:-compressor,quant_compressor,sparse_flash_mla,sparse_flash_mla_metadata}"

echo "checkout        : ${CHECKOUT}"
echo "CANN_ROOT       : ${CANN_ROOT}"
echo "OPS_THIRD_PARTY : ${OPS_THIRD_PARTY}"
echo "vendor name     : ${VENDOR_NAME}"
echo "installed vendor: ${VENDOR_NAME}_transformer (build.sh appends _transformer)"
echo "operators       : ${OPS_LIST}"

if [[ ! -f "${CANN_ROOT}/set_env.sh" ]]; then
  echo "FATAL: ${CANN_ROOT}/set_env.sh not found" >&2
  exit 2
fi
if [[ ! -d "${OPS_THIRD_PARTY}/opbase" || ! -d "${OPS_THIRD_PARTY}/ops-tensor" ]]; then
  echo "FATAL: ${OPS_THIRD_PARTY} is not an independent third-party copy" >&2
  exit 2
fi

# Explicit toolkit path only. Do NOT unset the proxy here: a blanket unset was
# not justified for this step (the direct connection already timed out at ~45 s).
export ASCEND_CANN_PACKAGE_PATH="${CANN_ROOT}"
# shellcheck disable=SC1091
source "${CANN_ROOT}/set_env.sh"

cd "${CHECKOUT}"
bash build.sh --incremental --pkg --jit --soc=ascend950 --vendor_name="${VENDOR_NAME}" \
  --ops="${OPS_LIST}" --cann_3rd_lib_path="${OPS_THIRD_PARTY}" -j4

echo "artifacts:"
ls -1 "${CHECKOUT}/build" 2>/dev/null | grep -E 'cann-ops-transformer.*\.run$' || true
