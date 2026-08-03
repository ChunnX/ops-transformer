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
# 用法:
#   bash run_spr_rw.sh -v ascend950pr_9599 -b Release -c ON -r ON -t 18 -n 36 -a 0 -d 1
#   -v soc-version        芯片版本 (ascend950pr_9599)
#   -b build-type         Release or Debug
#   -c cmake-rebuild      ON=重新编译, OFF=不编译
#   -r run-test           ON=运行用例, OFF=不运行
#   -t target_core_id     要访问的目标核 res_id (默认 18 = DIE0 AIV0)
#   -n res_id_count       halResMap 映射的核总数 (默认 36)
#   -a rank_id            rank ID (默认 0)
#   -d ranksize           卡数 (默认 1)
# 环境变量:
#   RANK_TABLE_FILE       rank table 文件路径
# -----------------------------------------------------------------------------------------------------------
CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)

BUILD_TYPE="Debug"
INSTALL_PREFIX="${CURRENT_DIR}/out"
SOC_VERSION="ascend950pr_9599"
CMAKE_REBUILD="OFF"
RUN_TEST="OFF"
TARGET_CORE_ID=18
RES_ID_COUNT=36
RANKSIZE=1
RANK_ID=0

SHORT=v:,b:,c:,r:,t:,n:,a:,d:,
LONG=soc-version:,build-type:,cmake-rebuild:,run-test:,target-core-id:,res-id-count:,rank-id:,ranksize:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"

while :; do
    case "$1" in
    -v | --soc-version)
        SOC_VERSION="$2"
        shift 2
        ;;
    -b | --build-type)
        BUILD_TYPE="$2"
        shift 2
        ;;
    -c | --cmake-rebuild)
        CMAKE_REBUILD="$2"
        shift 2
        ;;
    -r | --run-test)
        RUN_TEST="$2"
        shift 2
        ;;
    -t | --target-core-id)
        TARGET_CORE_ID="$2"
        shift 2
        ;;
    -n | --res-id-count)
        RES_ID_COUNT="$2"
        shift 2
        ;;
    -a | --rank-id)
        RANK_ID="$2"
        shift 2
        ;;
    -d | --ranksize)
        RANKSIZE="$2"
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

# 获取环境变量 ASCEND_HOME_PATH
if [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
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

# 运行
if [[ "$RUN_TEST" == "ON" ]]; then
    export LD_LIBRARY_PATH=${INSTALL_PREFIX}/lib:${INSTALL_PREFIX}/lib64:${_ASCEND_INSTALL_PATH}/lib64:$LD_LIBRARY_PATH
    echo "========== 开始执行 SPR 读写测试 =========="
    echo "target_core_id:                     $TARGET_CORE_ID"
    echo "res_id_count (halResMap 核总数):    $RES_ID_COUNT"
    echo "rank_id:                            $RANK_ID"
    echo "ranksize:                           $RANKSIZE"
    echo "=========================================="
    ./out/bin/test_spr_rw \
        --rank_id ${RANK_ID} \
        --ranksize ${RANKSIZE} \
        --target_core_id ${TARGET_CORE_ID} \
        --res_id_count ${RES_ID_COUNT}
    echo "========== 程序执行完成 =========="
fi
