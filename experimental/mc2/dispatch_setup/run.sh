#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# dispatch_min 集成构建 + 运行（最小功能：CCU 写状态位 / AIV aclnn 等状态位，单 die 两条流）。
# 本机为 a5、仅参考；目标是 a6 上编译跑通。运行前务必走 §8 NPU 清场。
# -----------------------------------------------------------------------------------------------------------
set -e

SOC_VERSION="${SOC_VERSION:-ascend950pr_9599}"           # [T8] a6 目标 SoC 串
OPS_TRANSFORMER_ROOT="${OPS_TRANSFORMER_ROOT:-/home/developer/ops-transformer}"
CANN_PATH="${ASCEND_HOME_PATH:-/home/0617/env/ascend-toolkit/latest}"   # [T8] a6 改 CANN 路径

# 0) CANN 环境（确保 $ASCEND_HOME_PATH 已 set_env）
if [ -f "${CANN_PATH}/../set_env.sh" ]; then source "${CANN_PATH}/../set_env.sh"; fi

# 1) 先构建并安装 teardown 的 aclnn 注册算子（注册框架路径，产出自定义算子包）。
#    它会扫描 experimental/mc2 下带注册 CMake 的算子（含本 dispatch_teardown）。
echo "==== [1/2] build + install dispatch_teardown aclnn op (registry) ===="
( cd "${OPS_TRANSFORMER_ROOT}" \
  && bash build.sh --pkg --soc=ascend950 --ops=dispatch_teardown -j16 \
  && ./build_out/cann-ops-transformer-custom_linux-*.run --force )
# 装包后 aclnnDispatchTeardown 由自定义算子包对外库提供；把它的 lib 路径/库名填进
# dispatch_setup/CMakeLists.txt 的 [T9] target_link_libraries（如 cust_opapi）。

# 2) 构建集成驱动（standalone cmake）。
echo "==== [2/2] build dispatch_min_driver (CCU + aclnn driver) ===="
rm -rf build && mkdir -p build
cmake -B build \
    -DSOC_VERSION="${SOC_VERSION}" \
    -DASCEND_CANN_PACKAGE_PATH="${CANN_PATH}" \
    -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j

echo "==== build done. 运行（先 §8 清场：npu-smi info; pkill -9 python; sleep 3; npu-smi info）===="
echo "  # [T7] 单 die rank table：cp rank_table.json.tmpl rank_table.json，把 device_ip 填成本卡 NPU ip"
echo "  #      （device_ip 取自 hccn_tool -i 0 -ip -g，device_id/rank_id 单 die 用 0）"
echo "  export LD_LIBRARY_PATH=${CANN_PATH}/lib64:\$LD_LIBRARY_PATH"
echo "  ./build/dispatch_min_driver ./rank_table.json"
