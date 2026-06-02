#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0.
# -----------------------------------------------------------------------------------------------------------
# A6 CCU MoeDistributeCombineCcuSetup 独立验证工程（v21）：一键编译 + 多进程（默认 2p）运行
#   - 只跑 combine_setup（CCU 0.5RTT 反向通信）+ host commWindow 校验；与 dispatch 用例完全解耦。
#   - 不依赖 dispatch 源（combine_setup 输入由 driver host 自行 mock）。
#   - 用法：bash run.sh [RANK_SIZE]    RANK_SIZE 缺省 2
#
# 注意：纯 host g++ 可编译；链接 / 运行需 A6（0.5RTT 上层 API 由 A6 hcomm 提供），A5 跑不动。
# -----------------------------------------------------------------------------------------------------------
CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")" && pwd)
CANN_ENV="/home/0513/env/set_env.sh"
RANK_SIZE=${1:-2}

source "${CANN_ENV}"
export ASCEND_TOOLKIT_HOME="${ASCEND_HOME_PATH}"

set -e
rm -rf "${CURRENT_DIR}/build" "${CURRENT_DIR}/out"
mkdir -p "${CURRENT_DIR}/build"
cmake -B "${CURRENT_DIR}/build" -S "${CURRENT_DIR}" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_INSTALL_PREFIX="${CURRENT_DIR}/out" \
    -DASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}"
cmake --build "${CURRENT_DIR}/build" -j
cmake --install "${CURRENT_DIR}/build"
set +e

export LD_LIBRARY_PATH="${CURRENT_DIR}/out/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}"
echo "========== 运行 A6 CCU combine_setup 0.5RTT 通信验证：${RANK_SIZE}p (MPI) =========="
# A6 范式：MPI 多进程拉起（rank/rankSize 由 MPI 给、rootinfo 由 MPI_Bcast 交换）。
#   mpirun 需在 PATH（CANN/MPI 环境已含）。每 rank 末尾自打印 PASS/FAIL（本卡 commWindow 校验）。
mpirun --allow-run-as-root -n ${RANK_SIZE} "${CURRENT_DIR}/out/bin/test_moe_distribute_combine_ccu"
exit $?
