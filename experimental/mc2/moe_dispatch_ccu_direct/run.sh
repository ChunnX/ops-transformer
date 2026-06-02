#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0.
# -----------------------------------------------------------------------------------------------------------
# A6 CCU MoeDistributeDispatchCcu 算子工程：一键编译 + 多进程（默认 2p）运行。
#
# 注意：A5 远程环境跑不动 A6 0.5RTT 特性（驱动 countXn / 上层 API 当前 hcomm 仓未实现）。
#   本脚本仅供 A6 真机/仿真使用；A5 上可编译验证骨架，但链接 / 运行需 SE 提供 0.5RTT
#   API 实现及 ChannelDesc.channelGroupTag 字段补全。
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
echo "========== 运行 A6 CCU dispatch_setup 0.5RTT 通信验证：${RANK_SIZE}p (MPI) =========="
# v20.8 ★ A6 范式：MPI 多进程拉起（rank/rankSize 由 MPI 给、rootinfo 由 MPI_Bcast 交换），
#   替代旧的手工 fork + --rank_id + 文件 rootinfo。mpirun 需在 PATH（CANN/MPI 环境已含）。
mpirun --allow-run-as-root -n ${RANK_SIZE} "${CURRENT_DIR}/out/bin/test_moe_distribute_dispatch_ccu"
exit $?
