#!/bin/bash
# ============================================================================
# probe5_norm_fp16_skeleton 一键运行：编译 -> 生成数据 -> NPU 运行 -> 精度验证
#   迭代一联调：A5 arch35 NORM/100/fp16 单核骨架路径 NPU 二进制一致验证
#   bash run.sh              # 完整流程
#   bash run.sh --skip-build # 复用已编译产物
# 退出码：0 成功；非 0 某步失败
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="probe_ffn_wb"
SKIP_BUILD=0
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
    esac
done
die() { echo "ERROR: $*" >&2; exit 1; }

echo "=== [1/4] 设置 CANN 环境 ==="
source /home/zl/Ascend/ascend-toolkit/set_env.sh || die "set_env.sh 执行失败"
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH 未设置"
# find_package(ASC) 需要 asc-config.cmake 所在前缀
export CMAKE_PREFIX_PATH="${ASCEND_HOME_PATH}/x86_64-linux/lib64/cmake:${ASCEND_HOME_PATH}/x86_64-linux/tikcpp/ascendc_kernel_cmake:${CMAKE_PREFIX_PATH:-}"

if [ "${SKIP_BUILD}" -eq 1 ]; then
    [ -f "build/${OP_NAME}" ] || die "--skip-build 但 build/${OP_NAME} 不存在"
    echo "=== [2/4] 跳过编译 ==="
else
    echo "=== [2/4] 编译 (dav-3510) ==="
    mkdir -p build
    cd build
    cmake .. || die "cmake 配置失败"
    make -j4 || die "make 编译失败"
    cd ..
fi

echo "=== [3/4] 生成测试数据 + golden ==="
cd build
python3 ../scripts/gen_data.py || die "gen_data.py 失败"

echo "=== [4/4] NPU 运行 Kernel ==="
rm -f output/y.bin output/group_list.bin output/actual_token_num.bin
"./${OP_NAME}" || die "Kernel 运行失败 (exit $?)"
[ -f output/y.bin ] || die "运行后 output/y.bin 不存在（静默失败）"

echo "=== 精度验证 ==="
python3 ../scripts/verify_result.py || die "精度验证失败"

echo "=== 完成 ==="
exit 0
