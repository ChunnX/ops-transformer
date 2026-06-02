#!/bin/bash
# Build run_aclnn standalone benchmark binary for qgmmd.
#
# Expects ASCEND_TOOLKIT_HOME (set by source set_env.sh) and the custom
# operator package installed under $HOME/.local/ascend-toolkit/vendors/
# custom_transformer/ (for libcust_opapi.so providing
# aclnnQuantGroupedMatmulDequant + aclnnQuantGroupedMatmulDequantWeightNZ).
#
# Adapted from QBMv3 device_tests/aclnn_bench/build.sh (2026-05-21).
set -e
cd "$(dirname "$0")"

if [ -z "$ASCEND_TOOLKIT_HOME" ]; then
    echo "ERROR: source CANN set_env.sh first (ASCEND_TOOLKIT_HOME unset)."
    exit 1
fi

ARCH=$(uname -m)  # x86_64 or aarch64
TOOLKIT_LIB="$ASCEND_TOOLKIT_HOME/${ARCH}-linux/lib64"
TOOLKIT_INC="$ASCEND_TOOLKIT_HOME/${ARCH}-linux/include"
# qgmmd vendor name is custom_transformer (NOT customize)
CUSTOM_LIB="$HOME/.local/ascend-toolkit/vendors/custom_transformer/op_api/lib"
CUSTOM_INC="$HOME/.local/ascend-toolkit/vendors/custom_transformer/op_api/include"

if [ ! -f "$CUSTOM_LIB/libcust_opapi.so" ]; then
    echo "ERROR: $CUSTOM_LIB/libcust_opapi.so missing — install custom op pkg first."
    exit 1
fi

CXX=${CXX:-g++}
mkdir -p build
$CXX -std=c++17 -O2 -Wall \
    -I"$TOOLKIT_INC" \
    -I"$TOOLKIT_INC/aclnnop" \
    -I"$CUSTOM_INC" \
    -I"$CUSTOM_INC/aclnnop" \
    run_aclnn.cpp \
    -L"$TOOLKIT_LIB" -L"$CUSTOM_LIB" \
    -Wl,-rpath,"$TOOLKIT_LIB" -Wl,-rpath,"$CUSTOM_LIB" \
    -lascendcl -lnnopbase -lcust_opapi \
    -o build/run_aclnn

echo "Built: $(realpath build/run_aclnn)"
