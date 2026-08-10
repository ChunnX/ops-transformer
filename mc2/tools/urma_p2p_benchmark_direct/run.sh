#!/bin/bash
set -euo pipefail

CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")" && pwd)

BUILD_TYPE="Release"
INSTALL_PREFIX="${CURRENT_DIR}/out"
SOC_VERSION="ascend950pr_9599"
CMAKE_REBUILD="OFF"
RUN_TEST="OFF"
NUM_SQE=16
SQE_BYTES=65536
BUF_BYTES=104857600
DB_INTERVAL=1
LOOP_COUNT=100
WARMUP=10
RANKSIZE=2
RANK_ID=0
SQE_MODE=0

SHORT=v:,b:,c:,r:,n:,s:,k:,l:,w:,d:,a:,m:
LONG=soc-version:,build-type:,cmake-rebuild:,run-test:,num-sqe:,sqe-bytes:,buf-bytes:,db-interval:,loop-count:,warmup:,ranksize:,rank-id:,sqe-mode:
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"

while :; do
    case "$1" in
    -v | --soc-version)     SOC_VERSION="$2"; shift 2 ;;
    -b | --build-type)      BUILD_TYPE="$2"; shift 2 ;;
    -c | --cmake-rebuild)   CMAKE_REBUILD="$2"; shift 2 ;;
    -r | --run-test)        RUN_TEST="$2"; shift 2 ;;
    -n | --num-sqe)         NUM_SQE="$2"; shift 2 ;;
    -s | --sqe-bytes)       SQE_BYTES="$2"; shift 2 ;;
    -k | --db-interval)     DB_INTERVAL="$2"; shift 2 ;;
    -l | --loop-count)      LOOP_COUNT="$2"; shift 2 ;;
    -w | --warmup)          WARMUP="$2"; shift 2 ;;
    -d | --ranksize)        RANKSIZE="$2"; shift 2 ;;
    -a | --rank-id)         RANK_ID="$2"; shift 2 ;;
    -m | --sqe-mode)        SQE_MODE="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "[ERROR] Unexpected option: $1"; break ;;
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
    echo "========== URMA P2P Bench =========="
    echo "num_sqe=$NUM_SQE sqe_bytes=$SQE_BYTES buf_bytes=$BUF_BYTES db_interval=$DB_INTERVAL"
    echo "loop_count=$LOOP_COUNT warmup=$WARMUP ranksize=$RANKSIZE rank_id=$RANK_ID sqe_mode=$SQE_MODE"
    echo "====================================="
    ./out/bin/test_urma_p2p_bench \
        --rank_id ${RANK_ID} \
        --ranksize ${RANKSIZE} \
        --num_sqe ${NUM_SQE} \
        --sqe_bytes ${SQE_BYTES} \
        --buf_bytes ${BUF_BYTES} \
        --db_interval ${DB_INTERVAL} \
        --loop_count ${LOOP_COUNT} \
        --warmup ${WARMUP} \
        --sqe_mode ${SQE_MODE}
    echo "========== Done =========="
fi
