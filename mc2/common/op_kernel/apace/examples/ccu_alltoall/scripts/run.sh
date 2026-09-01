#!/bin/bash
# run.sh — CCU AllToAll 阶段1（纯通信验证，FP16）
# 流程: 设置环境 → 编译 → 运行（单进程多线程起 R rank）→ Host 读回 recvBuf 校验
#
# 目录结构（本脚本位于 scripts/ 下，工程根为其上一级）：
#   <工程根>/
#   ├── CMakeLists.txt / README.md
#   ├── docs/（DESIGN_ANALYSIS 方案 / CODE_GUIDE 导读 / CODE_REVIEW 检视记录）
#   ├── scripts/run.sh        ← 本脚本
#   └── src/                  ← 源码（单翻译单元，CMake 只编译 .asc）
#
# 注意: 本脚本在用户自己的编译/运行服务器上执行（agent 不执行编译）
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"  # output/<rank>/ 等运行产物统一落地在工程根

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
cmake -S "${PROJECT_DIR}" -B "${PROJECT_DIR}/build" || { echo "ERROR: cmake failed"; exit 1; }
cmake --build "${PROJECT_DIR}/build" --target ccu_alltoall --parallel 4 || { echo "ERROR: build failed"; exit 1; }

# ---- 运行（默认 precision 模式：每块 65536 元素 FP16 = 128KB，commTurn=1）----
# 退出码：全部 rank 校验 PASS → 0；任一 rank FAIL → 非 0（本脚本 set -e 据此闭环）
echo "=== [2/2] Run (precision, ranks=NPU卡数, FP16, 128KB block, commTurn=1) ==="
if "${PROJECT_DIR}/build/ccu_alltoall" precision; then
    echo "=== Run PASS ==="
else
    run_ret=$?
    echo "=== Run FAIL (exit code ${run_ret}) ==="
    exit ${run_ret}
fi

# 附加用例（阶段2 精度验证矩阵，可按需手动打开，在工程根执行）：
#   commTurn=2 多轮事件桥:
#     ./build/ccu_alltoall precision 65536 2
#   非 loopSize 整数倍 blockSize（触发 goSize 尾部分支，16385 元素 = 32770 字节）:
#     ./build/ccu_alltoall precision 16385 1
#   perf 模式（循环 10 次不落地）:
#     ./build/ccu_alltoall perf 65536 1 10
