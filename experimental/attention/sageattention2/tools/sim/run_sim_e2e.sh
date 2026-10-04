#!/usr/bin/env bash
# B2 门禁端到端仿真一键脚本（CANN 容器内执行，经 ./docker/sage run 进入）。
# 用法：bash run_sim_e2e.sh <算子仓根> <工作目录>
#   <算子仓根> ops-transformer 仓根，缺省 /tmp/sa2_sim/ops-transformer（开发副本）；
#              门禁容器形态传 /work/ops-transformer（仓库根挂载, 见 docker/docker-compose.yml）。
#   <工作目录> bin 落点，须可写；门禁容器形态传 /work/out/sa2_sim_e2e（挂载自
#              算子 docker/out/），日志/产物目录经 SAGE_ATTN_SIM_OUT
#              覆盖（缺省 /tmp/sa2_logs）。
# 前置：算子门禁容器（experimental/attention/sageattention2/docker/，CANN 9.2.0-beta.1），
# 本脚本自行 source 环境；全部编译统一采用 BiSheng 编译器。
set -uo pipefail

REPO="${1:-/tmp/sa2_sim/ops-transformer}"
WORK="${2:-/tmp/sa2_work/e2e}"
SAGE_ATTN="$REPO/experimental/attention/sageattention2"
SIMDIR="$SAGE_ATTN/tools/sim"
OUT="${SAGE_ATTN_SIM_OUT:-/tmp/sa2_logs}"
mkdir -p "$WORK" "$OUT/camel_logs"

# ---- CANN 环境 ----
ASC="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
# shellcheck disable=SC1091
source "$ASC/bin/set_env.bash" 2>/dev/null || source "$ASC/bin/set_env.sh" 2>/dev/null || true
export ASCEND_HOME_PATH="$ASC"

# camodel 运行栈（cann-9.2.0-beta.1 侧 camodel 目录，内含 libruntime_camodel/
# libascend_hal/libstars_wrapper 全家桶；ascend-toolkit 侧 simulator 目录仅可链接，
# 两栈混用会在 libstars_wrapper.so 报 STARS_TOP 符号缺失）。
CAMODEL_STACK="/usr/local/Ascend/cann-9.2.0-beta.1/tools/simulator/Ascend950PR_9589/camodel"
export CAMODEL_LOG_PATH="$OUT/camel_logs"
export LD_LIBRARY_PATH="$CAMODEL_STACK:/usr/local/Ascend/cann-9.2.0-beta.1/lib64:/usr/local/Ascend/cann-9.2.0-beta.1/aarch64-linux/devlib:${LD_LIBRARY_PATH:-}"

SIMLIB="$ASC/aarch64-linux/simulator/Ascend950PR_9599/lib"
RC=0
# CATLASS 头（sage_tile.hpp/sage_attn_host_tiling.h 链必需；约定与 run.sh 一致）
CATLASS_INC="${CATLASS_ROOT:-${SAGE_ATTN}/docker/third_party/catlass}/include"

echo "== [1/4] golden 参考程序（bisheng 宿主模式，SAGE_ATTN_DEVICE_COMPILE=0 路径）=="
bisheng -std=c++17 -O2 -DSAGE_ATTN_DEVICE_COMPILE=0 \
    -I"$SAGE_ATTN/sageattn_fp8/ascendc" -I"$SAGE_ATTN/aclnn" -I"$SAGE_ATTN/quant" -I"$SIMDIR" \
    -I"$CATLASS_INC" \
    "$SIMDIR/sim_reference.cpp" "$SAGE_ATTN/quant/quant_int8.cpp" "$SAGE_ATTN/quant/quant_fp8.cpp" \
    -lstdc++ -lm \
    -o "$OUT/sim_reference" || { echo "[FAIL] 参考程序编译失败"; exit 1; }
(cd "$WORK" && "$OUT/sim_reference" "$WORK") || RC=1

echo "== [2/4] 设备 sim 编译与 camodel 仿真运行（已按指令禁用，直接使用真机验证）=="
# camodel 纯软件逐时钟周期仿真极度消耗 CPU 且耗时长达数小时，已全局禁用，直接使用真机 Atlas 950PR 验证
echo "[INFO] camodel 仿真已禁用，跳过 camodel 步骤。"
exit 0

