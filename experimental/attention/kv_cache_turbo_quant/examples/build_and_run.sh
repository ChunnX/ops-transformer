#!/usr/bin/env bash
# 编译 KvCacheTurboQuant 的 aclnn 示例 / 自测驱动器（examples/test_aclnn_kv_cache_turbo_quant.cpp）。
#
# 依赖一个已构建的算子包（提供 op_api 头与 libcust_opapi.so）：
#   bash build.sh --pkg --experimental --soc=ascend910b --ops=kv_cache_turbo_quant
#   ./build/cann-ops-transformer-custom_linux-aarch64.run --extract=<dir> --noexec
#
# 用法:
#   bash examples/build_and_run.sh <算子包解包目录>/packages [输出目录]
# 环境变量:
#   KCTQ_PKG  等价于第一个参数；KCTQ_OUT 等价于第二个参数（默认 /tmp/kctq_example）
#   ASCEND_HOME_PATH  CANN 安装根（默认 /usr/local/Ascend/cann-9.1.0）
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
PKG=${1:-${KCTQ_PKG:-}}
OUT=${2:-${KCTQ_OUT:-/tmp/kctq_example}}
ASCEND=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.1.0}

if [[ -z "$PKG" ]]; then
    echo "用法: bash examples/build_and_run.sh <算子包解包目录>/packages [输出目录]" >&2
    echo "（包由 build.sh --pkg ... 产出，再用 ./cann-ops-transformer-custom_linux-aarch64.run --extract=<dir> --noexec 解包）" >&2
    exit 2
fi

mkdir -p "$OUT"
g++ -std=c++17 -O2 -o "$OUT/test_aclnn_kv_cache_turbo_quant" \
    "$HERE/test_aclnn_kv_cache_turbo_quant.cpp" \
    -I"$PKG/vendors/custom_transformer/op_api/include" \
    -I"$ASCEND/include" \
    -L"$PKG/vendors/custom_transformer/op_api/lib" -lcust_opapi \
    -L"$ASCEND/lib64" -lascendcl -lnnopbase -lpthread

echo "built: $OUT/test_aclnn_kv_cache_turbo_quant"
echo "运行示例（需先加载算子包环境）:"
echo "  source <算子包安装目录>/vendors/custom_transformer/bin/set_env.bash"
echo "  $OUT/test_aclnn_kv_cache_turbo_quant --kv kv.bin --rot rotation_matrix.bin \\"
echo "      --qjl qjl_matrix.bin --out <输出目录> --tokens 2048 --heads 8 --bits 3"
