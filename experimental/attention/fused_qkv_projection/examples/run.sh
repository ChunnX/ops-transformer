#!/bin/bash
# ============================================================
# fused_qkv_projection 算子 ACLNN 调用示例运行脚本
# ============================================================

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    _ASCEND_INSTALL_PATH="/ddhome/qkv_xpb/Ascend/cann-9.1.0-beta.1"
fi

source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

rm -rf ${SCRIPT_DIR}/build
mkdir -p ${SCRIPT_DIR}/build
cd ${SCRIPT_DIR}/build
cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
make
cd bin
./test_aclnn_fused_qkv_projection
