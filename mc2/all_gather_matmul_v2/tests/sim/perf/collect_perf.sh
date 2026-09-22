#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# Performance collection framework (2.1 §7 T7 / A8 — 备用，本期不设验收门槛).
#
# 多轮采集：对选中用例逐轮执行被测 example，采集三类性能源：
#   1. cannsim/npusim 仿真报告（record --gen-report → summary.json，
#      供 perf_report.py 做瓶颈维度分解）；
#   2. example 计时打印（stdout 中 AGM_V2_TIMING_US 行，参数化 example
#      契约，见 tests/sim/README.md）；
#   3. msprof（有卡环境，--mode msprof；MC2 多 rank 算子必须用 msprof，
#      禁止 msprof op —— ascendc-mc2-best-practice / ops-profiling 口径）。
#
# Usage:
#   bash collect_perf.sh [--case L0-01,L1-133] [--level L0] [--rounds 5]
#                       [--mode sim|msprof] [--raw-root DIR] [--apace on|off]
#   仿真侧参考统计（本环境）；有卡环境切 --mode msprof。
#   --apace on：USING_APACE_IMPL 双分支对比的 APACE 侧（export
#   ASCEND_MC2_AGMV2_APACE_IMPL=1，进程内 tiling 读取）；off（默认）显式
#   unset 走 v1 分支。round 目录按分支隔离：perf_raw/<case>/branch_{v1|apace}/。
#   msprof 模式每轮轮前执行 L2 cache flush（l2_flush.py，B9⑤-5 口径：
#   每轮实际 flush、FLUSH_OK 证据行校验、失败判该轮失败）。
#
# 注意：多卡 HCCL 的仿真支持能力属 A9 风险项（未证实），sim 模式失败时
# 脚本记录错误并继续（框架可用性不受单例成败影响）。
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIM_DIR="$(dirname "${SCRIPT_DIR}")"
# 仓根四级上行（perf → sim → tests → <算子目录> → mc2 → 仓根）：
# 本脚本位于 mc2/<op>/tests/sim/perf/，SIM_DIR/../../.. 只到 mc2/（少一级，
# 实证见 3.4 落地烟雾），须 ../../../.. 才到 ops-transformer 仓根——与
# run_sim_cases.sh 的 SCRIPT_DIR/../../../..（四级）算法一致。
REPO_ROOT="$(cd "${SIM_DIR}/../../../.." && pwd)"
BUILD_SH="${REPO_ROOT}/build.sh"
# example 产物路径：build.sh 的 build_example 在 build/ 内以
# -o test_aclnn_${EXAMPLE_NAME}（EXAMPLE_NAME 为 --run_example 的算子名
# 位置实参）产出二进制；example_name 的区分在构建期完成——
# --example_name=test_aclnn_all_gather_matmul_v2_fp16 选定参数化 example
# 编译进该产物（避免 arch35 下 quant example 同 pattern 互覆），本框架
# 运行期直接复跑该二进制。
EXAMPLE_BIN="${REPO_ROOT}/build/test_aclnn_all_gather_matmul_v2"

MODE="sim"
LEVEL="all"
CASE_FILTER=""
ROUNDS=5
RAW_ROOT="${SCRIPT_DIR}/perf_raw"
APACE="off"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --level) LEVEL="$2"; shift 2 ;;
        --case) CASE_FILTER="$2"; shift 2 ;;
        --rounds) ROUNDS="$2"; shift 2 ;;
        --raw-root) RAW_ROOT="$2"; shift 2 ;;
        --apace) APACE="$2"; shift 2 ;;
        *) echo "unknown arg: $1"; exit 2 ;;
    esac
done

# USING_APACE_IMPL 双分支控制（4.1）：on=export（APACE 分支）/ off=unset（v1 分支，
# 显式清除免疫 shell 残留）。example 单进程多线程，aclnn 进程内 tiling 直接读该 env。
if [[ "${APACE}" == "on" ]]; then
    export ASCEND_MC2_AGMV2_APACE_IMPL=1
    BRANCH_TAG="apace"
elif [[ "${APACE}" == "off" ]]; then
    unset ASCEND_MC2_AGMV2_APACE_IMPL || true
    BRANCH_TAG="v1"
else
    echo "--apace must be on|off, got: ${APACE}" >&2; exit 2
fi
echo "[perf] branch=${BRANCH_TAG} (ASCEND_MC2_AGMV2_APACE_IMPL=${ASCEND_MC2_AGMV2_APACE_IMPL:-<unset>})"

mkdir -p "${RAW_ROOT}"

CASE_ENVS="$(python3 "${SIM_DIR}/../assets/sim_cases.py" "${LEVEL}" --env \
    --dump-root="${RAW_ROOT}/unused_dump")"

CURRENT_CASE=""
CURRENT_ENVS=""
FAILED_CASES=""

run_round() {
    local round_idx="$1"
    local round_dir="${RAW_ROOT}/${CURRENT_CASE}/branch_${BRANCH_TAG}/round_${round_idx}"
    mkdir -p "${round_dir}"
    if [[ "${MODE}" == "sim" ]]; then
        # 直接驱动已构建的 example 二进制（首轮由调用方经 build.sh 构建）
        (cd "${REPO_ROOT}" && export ${CURRENT_ENVS} && \
         AGM_V2_DUMP_DIR="${round_dir}/dump" && \
         cannsim record -s Ascend950 "${EXAMPLE_BIN}" --gen-report \
             -o "${round_dir}" > "${round_dir}/stdout.log" 2>&1)
    elif [[ "${MODE}" == "msprof" ]]; then
        # 有卡环境：msprof task-based（MC2 多 rank 必须 msprof，禁 msprof op）
        # B9⑤-5 口径：--ai-core/--aic-metrics=PipeUtilization/--task-time/
        # --ascendcl 显式声明（对齐 ops-profiling msprof-guide「MC² 多 rank
        # 算子采集」标准命令，防默认值跨版本漂移）。
        # 每轮轮前实际执行 L2 cache flush（l2_flush.py：256MB bf16 ×2 遍
        # in-place add，> 950PR PCIE L2 112MB；禁止只分配不 flush）；flush
        # 进程独立于 msprof 采集进程，flush kernel 不进 task 记录；无
        # FLUSH_OK 证据行判该轮失败（fail loud）。
        (cd "${REPO_ROOT}" && export ${CURRENT_ENVS} && \
         python3 "${SCRIPT_DIR}/l2_flush.py" > "${round_dir}/flush.log" 2>&1)
        if ! grep -q "FLUSH_OK" "${round_dir}/flush.log" 2>/dev/null; then
            echo "[perf] case ${CURRENT_CASE} round ${round_idx}: L2 flush FAILED (see ${round_dir}/flush.log)" >&2
            return 1
        fi
        (cd "${REPO_ROOT}" && export ${CURRENT_ENVS} && \
         msprof --application="${EXAMPLE_BIN}" \
                --output="${round_dir}/prof" \
                --ai-core=on --aic-mode=task-based \
                --aic-metrics=PipeUtilization \
                --task-time=on --ascendcl=on \
              > "${round_dir}/stdout.log" 2>&1)
    else
        echo "[perf] unknown mode ${MODE}" >&2
        return 1
    fi
}

while IFS= read -r line; do
    if [[ "${line}" == "---" ]]; then
        [[ -z "${CURRENT_CASE}" ]] && continue
        if [[ -n "${CASE_FILTER}" && ",${CASE_FILTER}," != *",${CURRENT_CASE},"* ]]; then
            CURRENT_CASE=""; CURRENT_ENVS=""
            continue
        fi
        echo "########## perf case ${CURRENT_CASE} [branch:${BRANCH_TAG}] (${MODE}, ${ROUNDS} rounds) ##########"
        if [[ ! -x "${EXAMPLE_BIN}" ]]; then
            echo "[perf] example binary not found: ${EXAMPLE_BIN}"
            echo "[perf] build once first: bash ${BUILD_SH} --run_example all_gather_matmul_v2 eager cust --vendor_name=custom --soc=ascend950 --example_name=test_aclnn_all_gather_matmul_v2_fp16"
            FAILED_CASES="${FAILED_CASES} ${CURRENT_CASE}"
            CURRENT_CASE=""; CURRENT_ENVS=""
            continue
        fi
        ok_rounds=0
        for i in $(seq 1 "${ROUNDS}"); do
            if run_round "${i}"; then
                ok_rounds=$((ok_rounds + 1))
            else
                echo "[perf] case ${CURRENT_CASE} round ${i}: collection FAILED (recorded)"
            fi
        done
        echo "[perf] case ${CURRENT_CASE}: ${ok_rounds}/${ROUNDS} rounds collected"
        [[ ${ok_rounds} -eq 0 ]] && FAILED_CASES="${FAILED_CASES} ${CURRENT_CASE}"
        CURRENT_CASE=""; CURRENT_ENVS=""
        continue
    fi
    if [[ "${line}" == AGM_V2_CASE_ID=* ]]; then
        CURRENT_CASE="${line#AGM_V2_CASE_ID=}"
        CURRENT_ENVS="${line}"
    elif [[ "${line}" == AGM_V2_RANK_SIZE=* ]]; then
        # rank 键桥接（4.1 就绪度核对）：sim_cases 输出 AGM_V2_RANK_SIZE、
        # 种子 example 消费 AGM_V2_DEV_NUM——同值补发两键（R=4/8 用例否则静默跑 R=2）
        CURRENT_ENVS="${CURRENT_ENVS} ${line} AGM_V2_DEV_NUM=${line#AGM_V2_RANK_SIZE=}"
    else
        CURRENT_ENVS="${CURRENT_ENVS} ${line}"
    fi
done <<< "${CASE_ENVS}"

echo "================================================================"
echo "[perf] raw data under: ${RAW_ROOT}"
[[ -n "${FAILED_CASES}" ]] && echo "[perf] no-round cases:${FAILED_CASES}"
echo "[perf] next: python3 ${SCRIPT_DIR}/perf_report.py --raw-root ${RAW_ROOT}"
echo "[perf] dual-branch compare: 重复执行本脚本 --apace on|off 各一轮后, perf_report 按 case[branch] 出双分支条目"
exit 0
