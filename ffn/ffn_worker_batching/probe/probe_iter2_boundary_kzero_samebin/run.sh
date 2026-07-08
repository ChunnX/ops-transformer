#!/bin/bash
# ============================================================================
# probe_iter2_boundary_kzero_samebin 一键运行：编译 -> 两场景(NPU 实跑 + 精度验证 + 3x 确定性)
#   迭代二穿刺3：A5 arch35 NORM/100/fp16 单核 —— 边界 全 mask(k=0) + same_bin(expert_num=8192)
#   bash run.sh              # 完整流程
#   bash run.sh --skip-build # 复用已编译产物
# 退出码：0 全部场景通过；非 0 某步失败
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="probe_ffn_wb"
SCENARIOS=("kzero" "samebin")
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

cd build
for sc in "${SCENARIOS[@]}"; do
    echo ""
    echo "############################################################"
    echo "### 场景: ${sc}"
    echo "############################################################"

    echo "=== [3/4] 生成测试数据 + golden (${sc}) ==="
    python3 ../scripts/gen_data.py "${sc}" || die "gen_data.py ${sc} 失败"

    echo "=== [4/4] NPU 运行 Kernel (${sc}) ==="
    rm -f "output/${sc}/y.bin" "output/${sc}/group_list.bin" "output/${sc}/actual_token_num.bin"
    "./${OP_NAME}" "${sc}" || die "Kernel 运行失败 ${sc} (exit $?)"
    [ -f "output/${sc}/y.bin" ] || die "运行后 output/${sc}/y.bin 不存在（静默失败）"

    echo "=== 精度验证 (${sc}) ==="
    python3 ../scripts/verify_result.py "${sc}" || die "精度验证失败 ${sc}"

    echo "=== 确定性验证 3x (${sc}) ==="
    prev=""
    for run in 1 2 3; do
        "./${OP_NAME}" "${sc}" >/dev/null || die "确定性运行 ${sc}#${run} 失败"
        cur=$(cat "output/${sc}/y.bin" "output/${sc}/group_list.bin" \
                  "output/${sc}/session_ids.bin" "output/${sc}/micro_batch_ids.bin" \
                  "output/${sc}/token_ids.bin" "output/${sc}/expert_offsets.bin" \
                  "output/${sc}/dynamic_scale.bin" "output/${sc}/actual_token_num.bin" | sha256sum | awk '{print $1}')
        echo "  run#${run} merged SHA256 = ${cur}"
        if [ -n "${prev}" ] && [ "${prev}" != "${cur}" ]; then
            die "确定性失败 ${sc}：run#${run} SHA256 与前次不一致"
        fi
        prev="${cur}"
    done
    echo "=== [${sc}] bitwise reproducible OK (SHA256=${prev}) ==="
done

echo ""
echo "=== 全部场景完成 (kzero + samebin) ==="
exit 0
