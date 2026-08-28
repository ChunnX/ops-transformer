#!/usr/bin/env bash
# Install AICore custom package, then merge AICPU metadata kernel artifacts.
set -euo pipefail

INSTALL_PATH="${1:-/data/cjz/add_gsag_custom}"
OPS_ROOT="/data/cjz/ops-transformer_2"
RUN_PKG="${OPS_ROOT}/build_out/cann-ops-transformer-custom_linux-x86_64.run"

if [[ ! -f "${RUN_PKG}" ]]; then
  RUN_PKG="${OPS_ROOT}/build/cann-ops-transformer-custom_linux-x86_64.run"
fi

# --force: skip shared-lib validation when optional deps (e.g. libopapi_math.so) are absent.
"${RUN_PKG}" --quiet --force --install-path="${INSTALL_PATH}"

CPU_IMPL="${INSTALL_PATH}/vendors/custom_transformer/op_impl/cpu"
mkdir -p "${CPU_IMPL}/aicpu_kernel/impl" "${CPU_IMPL}/config"

AICPU_SO="${OPS_ROOT}/build/libtransformer_aicpu_kernels.so"
AICPU_JSON="${OPS_ROOT}/build/cust_aicpu_kernel.json"
if [[ -f "${AICPU_SO}" && -f "${AICPU_JSON}" ]]; then
  cp -f "${AICPU_SO}" "${CPU_IMPL}/aicpu_kernel/impl/"
  cp -f "${AICPU_JSON}" "${CPU_IMPL}/config/"
  echo "Merged AICPU metadata kernel into ${CPU_IMPL}"
else
  echo "Warning: AICPU artifacts not found under ${OPS_ROOT}/build; metadata op may fail."
fi
