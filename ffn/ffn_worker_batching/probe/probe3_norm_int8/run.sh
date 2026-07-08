#!/bin/bash
# ============================================================================
# probe3_norm_int8 一键运行：设置环境 → 编译(dav-3510) → 生成数据+golden → NPU 实跑 → 二进制校验
# 用法：
#   bash run.sh              完整流程
#   bash run.sh --skip-build 复用已编译产物
# 退出码：0 全部成功；非 0 某步失败
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="ffn_wb_probe"
SKIP_BUILD=0
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
    esac
done
die() { echo "ERROR: $*" >&2; exit 1; }

echo "=== [1/4] 设置 CANN 环境 ==="
source /root/Ascend/ascend-toolkit/set_env.sh || die "set_env.sh 执行失败"
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH 未设置"

if [ "${SKIP_BUILD}" -eq 1 ]; then
    [ -f "build/${OP_NAME}" ] || die "--skip-build 指定但 build/${OP_NAME} 不存在"
    echo "=== [2/4] 跳过编译（复用已有产物）==="
else
    echo "=== [2/4] 编译 (dav-3510) ==="
    rm -rf build
    mkdir -p build
    cd build
    cmake .. || die "cmake 配置失败"
    make -j8  || die "make 编译失败"
    cd ..
fi

echo "=== [3/4] 生成测试数据 + CPU golden ==="
cd build
mkdir -p input output
python3 ../scripts/gen_data.py || die "gen_data.py 执行失败"

echo "=== [4/4] NPU 实跑 Kernel ==="
rm -f output/y.bin
"./${OP_NAME}" || die "Kernel 运行失败（exit code $?）"
[ -f output/y.bin ] || die "运行后 output/y.bin 不存在（静默失败）"

echo "=== 精度验证（二进制一致）==="
python3 ../scripts/verify_result.py || die "精度验证失败"

echo "=== 完成 ==="
exit 0
