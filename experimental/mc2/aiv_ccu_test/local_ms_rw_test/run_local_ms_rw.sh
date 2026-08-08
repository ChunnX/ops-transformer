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
#   bash run_local_ms_rw.sh -v ascend950pr_9599 -b Release -c ON -r ON -m 0 -w 0 -u 0 -l 100 -s 4096 -n 1
#   -v soc-version     芯片版本 (ascend950pr_9599)
#   -b build-type      Release or Debug
#   -c cmake-rebuild   ON=重新编译, OFF=不编译
#   -r run-test        ON=运行用例, OFF=不运行
#   -m mode0           0=写, 1=读, 2=读写混合, 3=先写后读
#   -w warmup          0=不预热, 1=正式打点前执行一次相同操作
#   -u ms_id           选择测试的 MS ID，范围 0..3
#   -l loop_count      正式执行循环次数
#   -s data_size       数据量(字节), 4KB=4096 ~ 120KB=122880 (mode0=3时上限120KB)
#   -n ranksize        卡数(默认1)
#   -a rank_id         rank ID(默认0)
# 环境变量:
#   RANK_TABLE_FILE    rank table 文件路径(多卡时必须设置)
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
MODE0=0
WARMUP=0
MS_ID=0
LOOP_COUNT=100
DATA_SIZE=4096
RANKSIZE=1
RANK_ID=0

SHORT=v:,b:,c:,r:,m:,w:,u:,l:,s:,n:,a:,
LONG=soc-version:,build-type:,cmake-rebuild:,run-test:,mode0:,warmup:,ms-id:,loop-count:,data-size:,ranksize:,rank-id:,
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
    -m | --mode0)
        MODE0="$2"
        shift 2
        ;;
    -w | --warmup)
        WARMUP="$2"
        shift 2
        ;;
    -u | --ms-id)
        MS_ID="$2"
        shift 2
        ;;
    -l | --loop-count)
        LOOP_COUNT="$2"
        shift 2
        ;;
    -s | --data-size)
        DATA_SIZE="$2"
        shift 2
        ;;
    -n | --ranksize)
        RANKSIZE="$2"
        shift 2
        ;;
    -a | --rank-id)
        RANK_ID="$2"
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
    echo "========== 开始执行本地 MS 读写测试 =========="
    echo "mode0(WRITE=0/READ=1/READ_WRITE=2/WRITE_THEN_READ=3): $MODE0"
    echo "warmup(OFF=0/ON=1):                 $WARMUP"
    echo "ms_id:                              $MS_ID"
    echo "loop_count:                         $LOOP_COUNT"
    echo "data_size:                          $DATA_SIZE bytes ($(($DATA_SIZE / 1024)) KB)"
    echo "rank_id:                            $RANK_ID"
    echo "ranksize:                           $RANKSIZE"
    echo "================================================="
    # for ((rank_id=0; rank_id<RANKSIZE; rank_id++))
    # do
    #     (
    #         ./out/bin/test_local_ms_rw \
    #             --rank_id ${rank_id} \
    #             --ranksize ${RANKSIZE} \
    #             --mode0 ${MODE0} \
    #             --data_size ${DATA_SIZE}
    #     ) &
    # done
    # wait
    ./out/bin/test_local_ms_rw \
        --rank_id ${RANK_ID} \
        --ranksize ${RANKSIZE} \
        --mode0 ${MODE0} \
        --warmup ${WARMUP} \
        --ms_id ${MS_ID} \
        --loop_count ${LOOP_COUNT} \
        --data_size ${DATA_SIZE}
    echo "========== 程序执行完成 =========="
fi
