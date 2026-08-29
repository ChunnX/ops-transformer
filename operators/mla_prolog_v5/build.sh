#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

set -e

# mla_prolog_v5：A2（ascend910b/ascend910_93）独立算子工程构建脚本
SUPPORT_COMPUTE_UNITS=("ascend910b" "ascend910_93")

export BASE_PATH=$(
  cd "$(dirname $0)"
  pwd
)
export BUILD_PATH="${BASE_PATH}/build"
export BUILD_OUT_PATH="${BASE_PATH}/build_out"

CORE_NUMS=$(cat /proc/cpuinfo | grep "processor" | wc -l)
if [ ${CORE_NUMS} -gt 8 ]; then
  CORE_NUMS=8
fi

usage() {
  echo "Build script for mla_prolog_v5 operator"
  echo "Usage: bash build.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  -h, --help              Print this help message"
  echo "  --list-socs             List supported SoC versions"
  echo "  -j[n]                   Compile thread nums, default is ${CORE_NUMS}, eg: -j8"
  echo "  --soc=soc_version       Compile for specified Ascend SoC (ascend910b, ascend910_93)"
  echo "  --make_clean            Clean build artifacts"
  echo "  --install               Install the built .run operator package after build"
  echo "  --torch                 Build torch_extension (pybind) after package build"
  echo ""
  echo "Examples:"
  echo "  bash build.sh --soc=ascend910b -j8"
  echo "  bash build.sh --soc=ascend910b --install --torch"
}

check_compute_unit() {
  local unit="$1"
  for support_unit in "${SUPPORT_COMPUTE_UNITS[@]}"; do
    if [[ "$unit" == "$support_unit" ]]; then
      return 0
    fi
  done
  return 1
}

# Default values
THREAD_NUM=${CORE_NUMS}
COMPUTE_UNIT=""
ENABLE_CLEAN=FALSE
RUN_INSTALL=FALSE
BUILD_TORCH=FALSE

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --list-socs)
      echo "Supported SoC versions:"
      for soc in "${SUPPORT_COMPUTE_UNITS[@]}"; do
        echo "  - ${soc}"
      done
      exit 0
      ;;
    -j*)
      THREAD_NUM="${1:2}"
      if [ -z "$THREAD_NUM" ]; then
        THREAD_NUM=${CORE_NUMS}
      fi
      shift
      ;;
    --make_clean)
      ENABLE_CLEAN=TRUE
      shift
      ;;
    --install)
      RUN_INSTALL=TRUE
      shift
      ;;
    --torch)
      BUILD_TORCH=TRUE
      shift
      ;;
    --soc=*)
      COMPUTE_UNIT="${1#*=}"
      shift
      ;;
    -*)
      echo "[ERROR] Invalid option: $1"
      usage
      exit 1
      ;;
    *)
      echo "[ERROR] Unexpected argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [ "$ENABLE_CLEAN" = "TRUE" ]; then
  rm -rf ${BUILD_PATH}
  rm -rf ${BUILD_OUT_PATH}
  echo "[INFO] Build directories cleaned."
  exit 0
fi

if [ -z "$COMPUTE_UNIT" ]; then
  echo "[ERROR] --soc parameter is required"
  usage
  exit 1
fi

COMPUTE_UNIT=$(echo "$COMPUTE_UNIT" | tr '[:upper:]' '[:lower:]')
if ! check_compute_unit "$COMPUTE_UNIT"; then
  echo "[ERROR] Invalid SoC version: ${COMPUTE_UNIT}"
  echo "[INFO] Supported SoC versions: ${SUPPORT_COMPUTE_UNITS[@]}"
  exit 1
fi
echo "[INFO] Compile for SoC: ${COMPUTE_UNIT}"

CMAKE_ARGS="-DASCEND_COMPUTE_UNIT=${COMPUTE_UNIT}"

mkdir -p "${BUILD_PATH}"
[ -f "${BUILD_PATH}/CMakeCache.txt" ] && rm -f ${BUILD_PATH}/CMakeCache.txt

echo "----------------------------------------------------------------"
echo "[INFO] Configuring project..."
cd "${BUILD_PATH}" && cmake ${CMAKE_ARGS} ..

echo "----------------------------------------------------------------"
echo "[INFO] Building project with ${THREAD_NUM} threads..."
cmake --build . --target all binary package -- -j ${THREAD_NUM}

KERNEL_O=$(find ${BUILD_PATH}/op_kernel -path "*binary*${COMPUTE_UNIT}*" -name "*.o" 2>/dev/null | head -1)
if [ -z "$KERNEL_O" ]; then
  echo "[ERROR] Kernel binary not found"
  exit 1
fi

PKG_PATH="${BUILD_PATH}/custom_opp_ubuntu_aarch64.run"
if [ ! -f "$PKG_PATH" ] || [ ! -s "$PKG_PATH" ]; then
  echo "[ERROR] Package not found or empty: ${PKG_PATH}"
  exit 1
fi

echo "----------------------------------------------------------------"
echo "[INFO] Build completed successfully!"
echo "[INFO] Kernel binary: ${KERNEL_O}"
echo "[INFO] Package: ${PKG_PATH}"

if [ "$RUN_INSTALL" = "TRUE" ]; then
  echo "----------------------------------------------------------------"
  echo "[INFO] Installing operator package..."
  ${PKG_PATH}
  if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install operator package"
    exit 1
  fi
  echo "[INFO] Operator package installed successfully!"
fi

if [ "$BUILD_TORCH" = "TRUE" ]; then
  echo "----------------------------------------------------------------"
  echo "[INFO] Building torch_extension (pybind)..."
  bash "${BASE_PATH}/torch_extension/build.sh"
  if [ $? -ne 0 ]; then
    echo "[ERROR] torch_extension build failed"
    exit 1
  fi
fi
