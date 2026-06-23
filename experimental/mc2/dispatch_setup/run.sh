#!/bin/bash
# ===========================================================================================================
# dispatch_min 一键构建脚本 —— dispatch_setup(CCU 写状态位) + dispatch_teardown(AIV aclnn 等状态位)
#   2 die / 2 rank（MPI）、每 rank 两条流、写/等状态位串行同步。本机为 a5(仅编译验证)；最终目标 a6 编译跑通。
#   2-die 理由：CCU 容器(ccuContainer)由库内部在【CCU 扩展模式 + rank_size>=2】下创建，单 rank 不建容器。
#
# 用法：
#   TARGET=a5 ./run.sh      # 默认。a5 骨架验证：CCU 围栏关、不装包、链散装 libcust_opapi.so（已实测 build OK）
#   TARGET=a6 ./run.sh      # a6 全功能：开 -DDISPATCH_CCU_A6=ON + 安装算子包（运行所需）
# 可用 env 覆盖：SOC_VERSION / BUILD_SOC / ASCEND_HOME_PATH(CANN_PATH) / OPS_TRANSFORMER_ROOT / CONDA_ENV / JOBS
# 详细说明见同目录 BUILD_RUN_GUIDE.html
# ===========================================================================================================
set -euo pipefail

TARGET="${TARGET:-a5}"                                              # a5 | a6
SOC_VERSION="${SOC_VERSION:-ascend950pr_9599}"                      # 驱动 cmake 的 -DSOC_VERSION（[T8] a6 确认真值）
BUILD_SOC="${BUILD_SOC:-ascend950}"                                 # build.sh 的 --soc（算子包编译目标）
OPS_ROOT="${OPS_TRANSFORMER_ROOT:-/home/developer/ops-transformer}"
CANN_PATH="${ASCEND_HOME_PATH:-/home/0617/env/ascend-toolkit/latest}"   # [T8] a6 改成 a6 的 CANN 路径
CONDA_ENV="${CONDA_ENV:-mc2_py310}"
JOBS="${JOBS:-16}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==== config: TARGET=${TARGET} SOC_VERSION=${SOC_VERSION} BUILD_SOC=${BUILD_SOC} CANN=${CANN_PATH} ===="

# ---- 0. 环境：conda + CANN set_env（build.sh 的算子代码生成依赖 python 环境）----
if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate "${CONDA_ENV}" || true
fi
for se in "${CANN_PATH}/set_env.sh" "${CANN_PATH}/../set_env.sh"; do
    [ -f "${se}" ] && { source "${se}"; break; }
done

# ---- 1. teardown aclnn 注册算子（experimental/mc2 下，必须带 --experimental）----
echo "==== [1/2] build dispatch_teardown aclnn op (registry, --experimental) ===="
( cd "${OPS_ROOT}" && bash build.sh --pkg --experimental --soc="${BUILD_SOC}" --ops=dispatch_teardown -j"${JOBS}" )
RUN_PKG="$(ls "${OPS_ROOT}"/build_out/cann-ops-transformer-custom_linux-*.run 2>/dev/null | head -1)"
CUST_OPAPI="${OPS_ROOT}/build/libcust_opapi.so"                     # 散装库（导出 aclnnDispatchTeardown），a5 直链用
echo "  pkg : ${RUN_PKG:-<none>}"
echo "  lib : ${CUST_OPAPI}"

# a6：安装算子包（运行时 aclnn 才能加载并拉起 kernel；a5 仅编译验证则不装，避免动共享 opp）
if [ "${TARGET}" = "a6" ]; then
    echo "  -> install op package (--force)  [a6 运行所需]"
    "${RUN_PKG}" --force
fi

# ---- 2. 集成驱动（standalone cmake；CCU 路径由 TARGET 决定）----
echo "==== [2/2] build dispatch_min_driver ===="
CMAKE_EXTRA=()
if [ "${TARGET}" = "a6" ]; then
    CMAKE_EXTRA+=( -DDISPATCH_CCU_A6=ON )                           # 编入 CCU kernel + 拉起 + 段10 LocalCopyNb 真实写
    # a6 装包后 find_library 通常能自动定位 cust_opapi；如未装可再加 -DLIB_cust_opapi=...
else
    CMAKE_EXTRA+=( -DLIB_cust_opapi="${CUST_OPAPI}" )               # a5：直链散装 opapi（不装包）
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
ls -la "${HERE}/build/dispatch_min_driver"

# ---- 3. 运行提示（仅 a6 + TARGET=a6 有意义）----
cat <<EOF

==== 运行（仅 a6 且 TARGET=a6；2 die / mpirun -n 2；a5 骨架 CCU 关、teardown 会死等状态位，勿运行）====
  # (1) §8 清场（2 die 用 dev0/dev1）
  npu-smi info; pkill -9 python; sleep 3; npu-smi info
  # (2) 运行库路径（CANN lib64 + 散装 opapi 目录；装了包则只需 CANN lib64）
  export LD_LIBRARY_PATH=${CANN_PATH}/lib64:${OPS_ROOT}/build:\$LD_LIBRARY_PATH
  # (2.5) ★ teardown 自定义算子【设备侧二进制】路径——必须 .run --force 装包，且让 runtime 找到，否则
  #       报 errno 161002 "basePath.empty / May not set ASCEND_CUSTOM_OPP_PATH"（libcust_opapi.so 只是 host 入口）：
  export ASCEND_CUSTOM_OPP_PATH=\${ASCEND_OPP_PATH}/vendors/custom_transformer   # 路径以 .run --force 实际落点为准
  # (3) 跑：2 die / 2 rank，MPI 起 2 进程（root 取 rootInfo→MPI_Bcast 广播；每 rank dev=rank、各自本地写+轮询）
  mpirun --allow-run-as-root -n 2 ${HERE}/build/dispatch_min_driver
  # 期望: [rank 0] PASS: teardown observed status==1 (done=1)  且  [rank 1] PASS: ...(done=1)  两 rank 各自 PASS

  # ★ 若仍报 [HcclCcuKernelRegister] ccuContainer is nullptr，按序排查：
  #   (a) -DDISPATCH_CCU_A6=ON 是否生效（config.hcclOpExpansionMode 那行才会编进去）；
  #   (b) 是否 2 rank（mpirun -n 2）——单 rank 不建 CCU 容器；
  #   (c) CCU 扩展模式值（默认 6=ccu_sched；基线若用 ccu_ms 则 cmake ... -DDISPATCH_CCU_EXPANSION_MODE=5）；
  #   取值按 a6 基线数组约定（hccl_test_common.cc:36/:867），非 a5 头枚举（CCU_MS=4/CCU_SCHED=5）。
EOF
