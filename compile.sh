#!/bin/bash

set -o pipefail

# 定义全局变量：编译子进程PID、编译是否成功标识
BUILD_CHILD_PID=0
BUILD_SUCCESS=0  # 0=未完成/失败，1=成功

# ===================== 核心清理函数（仅异常时执行） =====================
cleanup() {
    # 仅当编译未成功时，才执行清理逻辑
    if [ $BUILD_SUCCESS -ne 1 ]; then
        echo -e "\n[$(date)] 检测到终止信号/编译失败，开始强制清理所有编译进程..."

        # 1. 终止编译子进程（如果存在）
        if [ $BUILD_CHILD_PID -ne 0 ]; then
            kill -9 $BUILD_CHILD_PID 2>/dev/null
            echo "[$(date)] 终止编译子进程: $BUILD_CHILD_PID"
        fi

        # 2. 执行兜底杀进程命令
        kill -9 $(ps -ef | grep -E 'run.sh|build.sh --pkg --soc=ascend960dt' | grep -v grep | awk '{print $2}') 2>/dev/null
        kill -9 $(ps -ef | grep bisheng | grep -v grep | awk '{print $2}') 2>/dev/null

        echo "[$(date)] 所有编译进程已清理完毕！"
        exit 1  # 异常退出，返回错误码
    fi
    # 编译成功时，此函数无操作
}

# ===================== 捕获信号（仅捕获手动终止信号，去掉EXIT） =====================
trap cleanup SIGINT SIGTERM  # 仅捕获Ctrl+C(SIGINT)、强制终止(SIGTERM)
trap '' SIGTSTP  # 屏蔽Ctrl+Z，避免卡死

# ===================== 执行编译逻辑 =====================
echo "[$(date)] 开始执行编译流程（按Ctrl+C可强制终止并清理）..."
rm -rf build
source /home/z00939165/Ascend96/cann/set_env.sh
bash build.sh --pkg --soc=ascend960dt --ops="allto_allv_quant_grouped_mat_mul" 2>&1 | tee 1.log &
BUILD_CHILD_PID=$!

# 等待子进程执行，并捕获编译返回码
wait "$BUILD_CHILD_PID"
BUILD_EXIT_CODE=$?  # 获取build.sh的退出码（0=成功，非0=失败）
BUILD_CHILD_PID=0

# ===================== 根据编译结果处理 =====================
if [ $BUILD_EXIT_CODE -eq 0 ]; then
    BUILD_SUCCESS=1  # 标记编译成功
   # echo "[$(date)] 编译流程正常执行完毕！"
    exit 0  # 正常退出，不触发cleanup
else
    BUILD_SUCCESS=0  # 编译失败，触发cleanup
    echo "[$(date)][ERROR] 编译流程执行失败，退出码: $BUILD_EXIT_CODE"
    cleanup
fi
