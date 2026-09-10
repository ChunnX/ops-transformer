#!/bin/bash
# run.sh — CCU AllGather 第一阶段（通信验证）
# 流程: 设置环境 → 编译 → 运行（单进程多线程起 2 rank）→ Host 读回 recvBuf 校验
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ---- 环境检查 ----
if [ -n "${ASCEND_HOME_PATH:-}" ] && [ -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
    source "${ASCEND_HOME_PATH}/set_env.sh"
else
    echo "ERROR: ASCEND_HOME_PATH not set or set_env.sh missing"; exit 1
fi

# CCU 调度模式必须设置
export HCCL_OP_EXPANSION_MODE=CCU_SCHED

# ---- 编译 ----
echo "=== [1/2] Build ==="
mkdir -p build
cmake -S "${SCRIPT_DIR}" -B "${SCRIPT_DIR}/build" || { echo "ERROR: cmake failed"; exit 1; }
cmake --build "${SCRIPT_DIR}/build" --target ccu_all_gather_matmul --parallel 4 || { echo "ERROR: build failed"; exit 1; }

# ---- 运行 ----
echo "=== [2/2] Run (2 ranks, M=256, K=256, FP16) ==="
"${SCRIPT_DIR}/build/ccu_all_gather_matmul"
