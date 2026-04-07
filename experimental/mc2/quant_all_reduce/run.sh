#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)
# 默认参数
INSTALL_PREFIX="${CURRENT_DIR}/out"
CMAKE_REBUILD="ON"

SHORT=c:,i:,p:,r:
LONG=cmake-rebuild:,install-path:,install-prefix:,run-test:
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"

while :; do
    case "$1" in
    -c | --cmake-rebuild)
        CMAKE_REBUILD="$2"
        shift 2
        ;;
    -i | --install-path)
        ASCEND_INSTALL_PATH="$2"
        shift 2
        ;;
    -p | --install-prefix)
        INSTALL_PREFIX="$2"
        shift 2
        ;;
    -r | --run-test)
        RUN_TEST="$2"
        shift 2
        ;;
    --)
        shift
        break
        ;;
    *)
        echo "[ERROR] Unexpected option: $1"
        break
        ;;
    esac
done

# 获取环境变量ASCEND_HOME_PATH
if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi

# 设置环境变量
export ASCEND_TOOLKIT_HOME=${_ASCEND_INSTALL_PATH}
export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}

set -e

# 编译
if [[ "$CMAKE_REBUILD" == "ON" ]]; then
    rm -rf build
    mkdir -p build
    cmake -B build \
        -DSOC_VERSION=ascend950pr_9599 \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
        -DASCEND_CANN_PACKAGE_PATH=${_ASCEND_INSTALL_PATH}
    cmake --build build -j
    cmake --install build
fi

# 运行
export LD_LIBRARY_PATH=$(pwd)/out/lib:$(pwd)/out/lib64:${_ASCEND_INSTALL_PATH}/lib64:$LD_LIBRARY_PATH
echo "========== 生成Golden数据 =========="
python3 ${CURRENT_DIR}/generate_golden.py
echo "========== 开始执行程序 =========="
./out/bin/test_quant_all_reduce
echo "========== 程序执行完成 =========="