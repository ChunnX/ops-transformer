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
#   bash run_local_xncke_rw.sh -v ascend950pr_9599 -b Release -c ON -r ON -m 2 -k 2 -w 1 -x 0 -e 0 -l 100
#   -v soc-version     芯片版本
#   -b build-type      Release or Debug
#   -c cmake-rebuild   ON=重新编译, OFF=不编译
#   -r run-test        ON=运行用例, OFF=不运行
#   -m mode            0=DataCopy写, 1=DataCopy读, 2=DataCopy先写后读, 3=SetValue写常数
#   -k res_kind        0=XN, 1=CKE, 2=BOTH
#   -w warmup          0=不预热, 1=正式打点前执行一次相同操作
#   -x xn_id           XN 资源 ID
#   -e cke_id          CKE 资源 ID
#   -l loop_count      循环次数
#   -p pattern         写入 pattern，支持 0x 前缀
#   -n ranksize        卡数(默认1)
# 环境变量:
#   RANK_TABLE_FILE    rank table 文件路径
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
MODE=2
RES_KIND=2
WARMUP=0
XN_ID=0
CKE_ID=0
LOOP_COUNT=100
PATTERN=0x5A5A000000000001
RANKSIZE=1
RANK_ID=0

SHORT=v:,b:,c:,r:,m:,k:,w:,x:,e:,l:,p:,n:,a:,
LONG=soc-version:,build-type:,cmake-rebuild:,run-test:,mode:,res-kind:,warmup:,xn-id:,cke-id:,loop-count:,pattern:,ranksize:,rank-id:,
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
    -m | --mode)
        MODE="$2"
        shift 2
        ;;
    -k | --res-kind)
        RES_KIND="$2"
        shift 2
        ;;
    -w | --warmup)
        WARMUP="$2"
        shift 2
        ;;
    -x | --xn-id)
        XN_ID="$2"
        shift 2
        ;;
    -e | --cke-id)
        CKE_ID="$2"
        shift 2
        ;;
    -l | --loop-count)
        LOOP_COUNT="$2"
        shift 2
        ;;
    -p | --pattern)
        PATTERN="$2"
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

if [[ "$RUN_TEST" == "ON" ]]; then
    export LD_LIBRARY_PATH=${INSTALL_PREFIX}/lib:${INSTALL_PREFIX}/lib64:${_ASCEND_INSTALL_PATH}/lib64:$LD_LIBRARY_PATH
    echo "========== 开始执行本地 XN/CKE 读写测试 =========="
    echo "mode(DATACOPY_WRITE=0/DATACOPY_READ=1/DATACOPY_WTR=2/SET_VALUE=3): $MODE"
    echo "res_kind(XN=0/CKE=1/BOTH=2):                                      $RES_KIND"
    echo "warmup(OFF=0/ON=1):                                                $WARMUP"
    echo "xn_id:                                                            $XN_ID"
    echo "cke_id:                                                           $CKE_ID"
    echo "loop_count:                                                       $LOOP_COUNT"
    echo "pattern:                                                          $PATTERN"
    echo "rank_id:                                                          $RANK_ID"
    echo "ranksize:                                                         $RANKSIZE"
    echo "=================================================="
    ./out/bin/test_local_xncke_rw \
        --rank_id ${RANK_ID} \
        --ranksize ${RANKSIZE} \
        --mode ${MODE} \
        --res_kind ${RES_KIND} \
        --warmup ${WARMUP} \
        --xn_id ${XN_ID} \
        --cke_id ${CKE_ID} \
        --loop_count ${LOOP_COUNT} \
        --pattern ${PATTERN}
    echo "========== 程序执行完成 =========="
fi
