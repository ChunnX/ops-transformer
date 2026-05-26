#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0.
# -----------------------------------------------------------------------------------------------------------
# A6 CCU MoeDistributeCombine 算子工程（v16）：一键编译 + 多进程（默认 2p）运行
#   - 跑完整 dispatch + combine 链路 + 数据闭环校验（identity expert + last-write-wins）
#   - 依赖 ../moe_dispatch_ccu_direct/ 下的 dispatch 源（CMakeLists 跨目录引用）
#   - 用法：bash run.sh [RANK_SIZE]    RANK_SIZE 缺省 2
#
# 注意：A5 远程环境跑不动 A6 0.5RTT；本脚本仅供 A6 真机/仿真使用。
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
rm -f /tmp/ccu_a6_combine_rootinfo*
echo "========== 运行 A6 CCU MoeDistribute dispatch+combine 完整链路：${RANK_SIZE}p =========="
pids=()
for ((rank_id = 0; rank_id < RANK_SIZE; rank_id++)); do
    "${CURRENT_DIR}/out/bin/test_moe_distribute_combine_ccu" \
        --rank_id ${rank_id} --rank_size ${RANK_SIZE} &
    pids+=($!)
done
FAIL=0
for ((i = 0; i < RANK_SIZE; i++)); do
    if wait "${pids[$i]}"; then echo "[rank $i] exit 0"; else echo "[rank $i] exit non-zero"; FAIL=1; fi
done
[ ${FAIL} -eq 0 ] && echo "全部 ${RANK_SIZE}p 通过：xOut == x 闭环成立" || echo "存在失败 rank"
exit ${FAIL}
