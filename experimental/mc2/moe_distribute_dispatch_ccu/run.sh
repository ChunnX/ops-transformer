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
# dispatch-ccu 编译 / 运行脚本（P0 骨架）
#   ./run.sh -c ON                       仅编译
#   ./run.sh -c ON -r ON                 编译并运行（需先 export RANK_TABLE_FILE）
CURRENT_DIR=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)

BUILD_TYPE="Debug"
INSTALL_PREFIX="${CURRENT_DIR}/out"
SOC_VERSION="ascend950pr_9599"   # A6 仿真器沿用 arch35 工具链
CMAKE_REBUILD="OFF"
RUN_TEST="OFF"

SHORT=c:,v:,b:,r:,
LONG=cmake-rebuild:,soc-version:,build-type:,run-test:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"
while :; do
    case "$1" in
    -c | --cmake-rebuild) CMAKE_REBUILD="$2"; shift 2 ;;
    -v | --soc-version)   SOC_VERSION="$2";   shift 2 ;;
    -b | --build-type)    BUILD_TYPE="$2";    shift 2 ;;
    -r | --run-test)      RUN_TEST="$2";      shift 2 ;;
    --) shift; break ;;
    *)  echo "[ERROR] Unexpected option: $1"; break ;;
    esac
done

# 获取 CANN 安装路径
if [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
elif [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
    _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
else
    _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
fi
export ASCEND_TOOLKIT_HOME=${_ASCEND_INSTALL_PATH}
export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}

set -e
# 编译
if [[ "$CMAKE_REBUILD" == "ON" ]]; then
    rm -rf build
    mkdir -p build
    cmake -B build \
        -DSOC_VERSION=${SOC_VERSION} \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
        -DASCEND_CANN_PACKAGE_PATH=${_ASCEND_INSTALL_PATH}
    cmake --build build -j
    cmake --install build
fi

# 运行（P0 骨架：单卡占位；多卡需配合 RANK_TABLE_FILE 与多进程拉起）
if [[ "$RUN_TEST" == "ON" ]]; then
    if [ -z "$RANK_TABLE_FILE" ]; then
        echo "[ERROR] 请先 export RANK_TABLE_FILE=<rank_table.json>"
        exit 1
    fi
    "${INSTALL_PREFIX}/bin/test_moe_distribute_dispatch_ccu" \
        --rank_id 0 --rank_size 8 --bs 16 --hidden 4096 --topk 8 --expert_num 64
fi
