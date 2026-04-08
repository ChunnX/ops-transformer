#!/bin/bash
# ============================================================================
# MTE AllToAll Kernel 直调运行脚本
# 编译 → 运行（多卡多线程测试）
#
# 前置条件：
#   - ASCEND_HOME_PATH 已设置
#   - 多卡环境可用（至少 4 卡）
#   - HCCL 已安装
#
# 用法：
#   bash run.sh              # 完整流程（含编译）
#   bash run.sh --skip-build # 跳过编译，复用已有产物
# ============================================================================

CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)

set -e

SOC_VERSION=${SOC_VERSION:-ascend950pr_9599}

SKIP_BUILD=0
if [ "${1:-}" == "--skip-build" ]; then
    SKIP_BUILD=1
fi

if [ "${SKIP_BUILD}" -eq 1 ]; then
    if [ ! -f "${CURRENT_DIR}/out/bin/test_mte_alltoall" ]; then
        echo "ERROR: --skip-build 但编译产物不存在，请先完整编译"
        exit 1
    fi
    echo "=== 跳过编译（复用已有产物）==="
else
    echo "=== 编译 ==="
    rm -rf build
    mkdir -p build
    cmake -B build \
        -DSOC_VERSION=${SOC_VERSION} \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_INSTALL_PREFIX=${CURRENT_DIR}/out \
        -DASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}
    cmake --build build -j
    cmake --install build
fi

echo "=== 运行 ==="
export LD_LIBRARY_PATH=$(pwd)/out/lib:$(pwd)/out/lib64:${ASCEND_HOME_PATH}/lib64:$LD_LIBRARY_PATH
echo "========== Running test =========="
./out/bin/test_mte_alltoall
echo "========== Done =========="
