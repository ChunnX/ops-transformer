#!/usr/bin/env bash
# Build the host driver that calls the real SparseFlashMlaMetadata AICPU kernel.
#
# This is a *host* g++ compile only: no bisheng, no AscendC, no simulator, no
# NPU, no board.  Run compilation and execution inside the shared local-resource-runner lock.
#
# What it links (all shipped by the CANN toolkit):
#   libaicpu_context_host.a   x86-64 CpuKernelContext/NodeDef/Tensor +
#                             CpuKernelUtils::CreateNodeDef, ABI0
#   libaicpu_nodedef_host.a   x86-64 aicpu::NodeDefBuilder, ABI0
#   libhost_ascend_protobuf.a x86-64 protobuf/abseil built with the
#                             `ascend_private` namespace the context layer uses
#
# Usage:
#   bash build_metadata_host.sh [out-dir]      # default ./build
#
# Environment overrides:
#   CANN_ROOT   toolkit root (default /home/jinxh/Ascend_0722/cann-9.2.0)
#   CXX         compiler (default g++)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ROOT="$(cd "${HERE}/../../../../.." && pwd)"
OPS_ROOT="$(cd "${HERE}/../../../../.." && pwd)"          # .../ops-transformer
CANN_ROOT="${CANN_ROOT:-/home/jinxh/Ascend_0722/cann-9.2.0}"
CXX="${CXX:-g++}"
OUT_DIR="${1:-${HERE}/build}"
SYS="x86_64-linux"

CTX="${CANN_ROOT}/${SYS}/pkg_inc/aicpu_common/context"
PKG_INC="${CANN_ROOT}/${SYS}/pkg_inc"
LIB64="${CANN_ROOT}/${SYS}/lib64"

KERNEL_DIR="${OPS_ROOT}/attention/sparse_flash_mla_metadata/op_kernel_aicpu"
KERNEL_CPP="${KERNEL_DIR}/sparse_flash_mla_metadata_aicpu.cpp"

CTX_LIB="${LIB64}/libaicpu_context_host.a"
NODEDEF_LIB="${LIB64}/libaicpu_nodedef_host.a"
PROTO_LIB="${LIB64}/libhost_ascend_protobuf.a"

missing=0
for path in \
    "${CTX}/common/cpu_context.h" \
    "${CTX}/common/node_def_builder.h" \
    "${CTX}/utils/log.h" \
    "${PKG_INC}/aicpu/aicpu_task_struct.h" \
    "${PKG_INC}/base/dlog_pub.h" \
    "${KERNEL_CPP}" \
    "${KERNEL_DIR}/sparse_flash_mla_metadata_aicpu.h" \
    "${CTX_LIB}" "${NODEDEF_LIB}" "${PROTO_LIB}"; do
  if [[ ! -e "${path}" ]]; then
    echo "MISSING: ${path}" >&2
    missing=1
  fi
done
if [[ "${missing}" -ne 0 ]]; then
  echo "FATAL: build inputs missing; set CANN_ROOT / check the ops-transformer checkout" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

echo "compiler      : ${CXX}"
echo "CANN_ROOT     : ${CANN_ROOT}"
echo "kernel source : ${KERNEL_CPP}"
echo "context libs  : ${CTX_LIB}"
echo "                ${NODEDEF_LIB}"
echo "                ${PROTO_LIB}"

# _GLIBCXX_USE_CXX11_ABI=0  : match the toolkit's host context libraries
# -DRUN_TEST                : KERNEL_LOG_* print to stdout (log.h) instead of
#                             requiring the AICPU dlog runtime
# -no-pie                   : the toolkit static archives are not all PIC
"${CXX}" -std=c++17 -O2 -g \
  -D_GLIBCXX_USE_CXX11_ABI=0 -DRUN_TEST -no-pie \
  -I"${HERE}" \
  -I"${KERNEL_DIR}" \
  -I"${CTX}" -I"${CTX}/common" -I"${CTX}/cpu_proto" -I"${CTX}/utils" \
  -I"${PKG_INC}" -I"${PKG_INC}/base" -I"${PKG_INC}/aicpu" \
  -o "${OUT_DIR}/metadata_host" \
  "${HERE}/metadata_host_main.cpp" \
  "${KERNEL_CPP}" \
  -Wl,--start-group "${CTX_LIB}" "${NODEDEF_LIB}" "${PROTO_LIB}" -Wl,--end-group \
  -L"${LIB64}" -Wl,-rpath,"${LIB64}" -lc_sec -ldl -lpthread -lm

echo "built: ${OUT_DIR}/metadata_host"
