#!/bin/bash
# ==============================================================================================
#  chain_e2e/run.sh —— 端到端 4 算子链驱动一键构建（+ a6 运行提示）。
#    ① 编两个 teardown aclnn 算子包（dispatch_teardown + combine_teardown，registry，--experimental）
#    ② a6 装包（运行 aclnn 需要）
#    ③ cmake 编 chain_e2e_driver（TARGET=a6 开 CHAIN_CCU_A6，编入两个 CCU kernel）
#    ④ 运行提示（仅 a6：清场 + mpirun -n 2）
#
#  用法：
#    TARGET=a5 ./run.sh    # 默认。a5 shell：CCU 关、不装包、直链散装 cust_opapi（只验驱动本体编译/链接）
#    TARGET=a6 ./run.sh    # a6 全功能：开 CHAIN_CCU_A6 + 装两个 teardown 包
#  env 覆盖：SOC_VERSION / BUILD_SOC / ASCEND_HOME_PATH / MPI_HOME / CONDA_ENV / JOBS
# ==============================================================================================
set -euo pipefail

TARGET="${TARGET:-a5}"
SOC_VERSION="${SOC_VERSION:-ascend950pr_9599}"
BUILD_SOC="${BUILD_SOC:-ascend950}"
OPS_ROOT="${OPS_TRANSFORMER_ROOT:-/home/developer/ops-transformer}"
CANN_PATH="${ASCEND_HOME_PATH:-/home/0617/env/ascend-toolkit/latest}"
CONDA_ENV="${CONDA_ENV:-mc2_py310}"
JOBS="${JOBS:-16}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==== config: TARGET=${TARGET} SOC_VERSION=${SOC_VERSION} BUILD_SOC=${BUILD_SOC} CANN=${CANN_PATH} ===="

# ---- 0. 环境 ----
if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate "${CONDA_ENV}" || true
fi
for se in "${CANN_PATH}/set_env.sh" "${CANN_PATH}/../set_env.sh"; do
    [ -f "${se}" ] && { source "${se}"; break; }
done

# ---- 1. 编两个 teardown aclnn 算子包（registry，--experimental）----
echo "==== [1/3] build teardown aclnn ops (dispatch_teardown + combine_teardown) ===="
( cd "${OPS_ROOT}" && bash build.sh --pkg --experimental --soc="${BUILD_SOC}" --ops=dispatch_teardown,combine_teardown -j"${JOBS}" )
RUN_PKG="$(ls "${OPS_ROOT}"/build_out/cann-ops-transformer-custom_linux-*.run 2>/dev/null | head -1)"
CUST_OPAPI="${OPS_ROOT}/build/libcust_opapi.so"
echo "  pkg : ${RUN_PKG:-<none>}"

# a6：装包（运行时 aclnn 才能加载 kernel）
if [ "${TARGET}" = "a6" ]; then
    echo "  -> install op package (--force)  [a6 运行所需]"
    "${RUN_PKG}" --force
fi

# ---- 2. cmake 编链驱动 ----
echo "==== [2/3] build chain_e2e_driver ===="
CMAKE_EXTRA=()
if [ "${TARGET}" = "a6" ]; then
    CMAKE_EXTRA+=( -DCHAIN_CCU_A6=ON )
else
    # a5：直链散装 opapi（不装包）。★动态定位真实 .so，写死路径不符时不硬传（否则 -D 绕过 find_library
    #   存在性校验 → 链接期 undefined aclnnDispatchTeardown）；找不到就交给 CMake find_library 自搜。
    REAL_OPAPI="$(ls ${OPS_ROOT}/build/libcust_opapi.so ${OPS_ROOT}/build_out/*/libcust_opapi.so 2>/dev/null | head -1)"
    if [ -n "${REAL_OPAPI}" ] && [ -f "${REAL_OPAPI}" ]; then
        echo "  cust_opapi: ${REAL_OPAPI}"
        CMAKE_EXTRA+=( -DLIB_cust_opapi="${REAL_OPAPI}" )
    else
        echo "  [WARN] 未找到 libcust_opapi.so（${OPS_ROOT}/build/ 下）——交给 CMake find_library 自搜；仍缺则链接会报 undefined aclnn*Teardown，先跑步骤①装 teardown 包"
    fi
fi
cd "${HERE}"
rm -rf build && mkdir -p build
cmake -B build \
    -DSOC_VERSION="${SOC_VERSION}" \
    -DASCEND_CANN_PACKAGE_PATH="${CANN_PATH}" \
    -DCMAKE_BUILD_TYPE=Debug \
    "${CMAKE_EXTRA[@]}"
cmake --build build -j"${JOBS}"
echo "==== build OK ===="
ls -la "${HERE}/build/chain_e2e_driver"

# ---- 3. 运行提示（仅 a6）----
cat <<EOF

==== 运行（仅 a6 且 TARGET=a6；2 die / mpirun -n 2）====
  # (1) §8 清场（跑前必做）
  npu-smi info; pkill -9 chain_e2e_driver; pkill -9 python; sleep 3; npu-smi info
  # (2) 运行库路径 + teardown 设备侧二进制路径
  export LD_LIBRARY_PATH=${CANN_PATH}/lib64:${OPS_ROOT}/build:\$LD_LIBRARY_PATH
  export ASCEND_CUSTOM_OPP_PATH=\${ASCEND_OPP_PATH}/vendors/custom_transformer   # teardown aclnn 需要（161002 时查此）
  # (3) 跑全链：默认 bs=8 topk=8 h=7168 E=4；可传参 [bs topk h E [seed]]
  mpirun --allow-run-as-root -n 2 ${HERE}/build/chain_e2e_driver
  # 期望: 每 rank 打印 ①②③④ 各段 launched，末尾 [rank r] PASS: 链端到端 golden ... bad=0

  # ★ 若 [HcclCcuKernelRegister] ccuContainer is nullptr：
  #   (a) -DCHAIN_CCU_A6=ON 是否生效；(b) 是否 mpirun -n 2；(c) CCU 扩展模式（默认 6，基线用 ccu_ms 则 -DCHAIN_CCU_EXPANSION_MODE=5）
  # ★ 若 fp8/e8m0 dtype 编译报错：改 -DCHAIN_ACL_FP8_E4M3=<a6 实名> -DCHAIN_ACL_E8M0=<...>（[C1]）
EOF
