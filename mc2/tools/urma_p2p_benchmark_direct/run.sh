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
SQE_MODE=1
DEV_COUNT=2
DEV_IDS=""
PEER=""

SHORT=v:,b:,c:,r:,n:,s:,k:,l:,w:,m:,d:,i:,p:
LONG=soc-version:,build-type:,cmake-rebuild:,run-test:,num-sqe:,sqe-bytes:,buf-bytes:,db-interval:,loop-count:,sqe-mode:,dev-count:,dev-ids:,peer:
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
    -m | --sqe-mode)        SQE_MODE="$2"; shift 2 ;;
    -d | --dev-count)       DEV_COUNT="$2"; shift 2 ;;
    -i | --dev-ids)         DEV_IDS="$2"; shift 2 ;;
    -p | --peer)            PEER="$2"; shift 2 ;;
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
    echo "dev_count=$DEV_COUNT dev_ids=$DEV_IDS num_sqe=$NUM_SQE sqe_bytes=$SQE_BYTES buf_bytes=$BUF_BYTES db_interval=$DB_INTERVAL"
    echo "loop_count=$LOOP_COUNT sqe_mode=$SQE_MODE peer=$PEER"
    echo "====================================="
    RUN_ARGS="--dev_count ${DEV_COUNT} --num_sqe ${NUM_SQE} --sqe_bytes ${SQE_BYTES} --buf_bytes ${BUF_BYTES} --db_interval ${DB_INTERVAL} --loop_count ${LOOP_COUNT} --sqe_mode ${SQE_MODE}"
    if [ -n "$DEV_IDS" ]; then RUN_ARGS="$RUN_ARGS --dev_ids=$DEV_IDS"; fi
    if [ -n "$PEER" ]; then RUN_ARGS="$RUN_ARGS --peer=$PEER"; fi
    ./out/bin/test_urma_p2p_bench $RUN_ARGS
    echo "========== Done =========="
fi
