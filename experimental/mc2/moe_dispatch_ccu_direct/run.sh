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
rm -f /tmp/ccu_a6_rootinfo*
echo "========== 运行 A6 CCU MoeDispatch：${RANK_SIZE}p =========="
pids=()
for ((rank_id = 0; rank_id < RANK_SIZE; rank_id++)); do
    "${CURRENT_DIR}/out/bin/test_moe_distribute_dispatch_ccu" \
        --rank_id ${rank_id} --rank_size ${RANK_SIZE} &
    pids+=($!)
done
FAIL=0
for ((i = 0; i < RANK_SIZE; i++)); do
    if wait "${pids[$i]}"; then echo "[rank $i] exit 0"; else echo "[rank $i] exit non-zero"; FAIL=1; fi
done
[ ${FAIL} -eq 0 ] && echo "全部 ${RANK_SIZE}p 通过" || echo "存在失败 rank"
exit ${FAIL}
