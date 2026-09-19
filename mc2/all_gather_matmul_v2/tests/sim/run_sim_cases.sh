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
# Channel-A simulation case driver (2.1 §7 T3).
#
# 逐例驱动：参数化 example（build.sh --run_example --simulator，env 注入
# case 参数）→ dump 落盘 → compare_dump.py 按 §2.3 口径判定 → 汇总。
#
# Usage:
#   bash run_sim_cases.sh [--level L0|L1|all] [--case L0-01,L1-101]
#                        [--dump-root DIR] [--report-root DIR]
#                        [--dry-run] [--include-conditional]
#   默认执行 L0+L1 非条件例（L1-142 8 卡条件例需 --include-conditional）。
#
# 前置条件：
#   * 参数化 example（examples/arch35/test_aclnn_all_gather_matmul_v2.cpp）
#     已支持本脚本的 env 契约（AGM_V2_*，见 tests/sim/README.md §接口契约）；
#   * build 目录空闲（与其他构建任务互斥）；
#   * CANN 环境已 source。
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"   # ops-transformer 仓根
BUILD_SH="${REPO_ROOT}/build.sh"

LEVEL="all"
CASE_FILTER=""
DUMP_ROOT="${SCRIPT_DIR}/runs"
REPORT_ROOT="${SCRIPT_DIR}/reports"
DRY_RUN=0
INCLUDE_CONDITIONAL=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --level) LEVEL="$2"; shift 2 ;;
        --case) CASE_FILTER="$2"; shift 2 ;;
        --dump-root) DUMP_ROOT="$2"; shift 2 ;;
        --report-root) REPORT_ROOT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --include-conditional) INCLUDE_CONDITIONAL=1; shift ;;
        *) echo "unknown arg: $1"; exit 2 ;;
    esac
done

mkdir -p "${DUMP_ROOT}" "${REPORT_ROOT}"
SUMMARY="${REPORT_ROOT}/summary.jsonl"
: > "${SUMMARY}"

# 从用例表取参数（tests/assets/sim_cases.py 为唯一用例源）
CASE_ENVS="$(python3 "${SCRIPT_DIR}/../assets/sim_cases.py" "${LEVEL}" --env \
    --dump-root="${DUMP_ROOT}")"
if [[ -z "${CASE_ENVS}" ]]; then
    echo "[driver] no case selected (level=${LEVEL})"
    exit 2
fi

TOTAL=0
PASS=0
FAIL=0
CURRENT_CASE=""

while IFS= read -r line; do
    if [[ "${line}" == "---" ]]; then
        [[ -z "${CURRENT_CASE}" ]] && continue
        if [[ -n "${CASE_FILTER}" && ",${CASE_FILTER}," != *",${CURRENT_CASE},"* ]]; then
            CURRENT_CASE=""
            continue
        fi
        TOTAL=$((TOTAL + 1))
        echo "================================================================"
        echo "[driver] case ${CURRENT_CASE} (${TOTAL})"
        echo "----------------------------------------------------------------"
        if [[ ${DRY_RUN} -eq 1 ]]; then
            echo "[driver] dry-run: would execute build.sh --run_example --simulator with:"
            echo "${CURRENT_ENVS}" | sed 's/^/    export /'
            CURRENT_CASE=""; CURRENT_ENVS=""
            continue
        fi
        # 1) 参数化 example：cannsim 仿真执行 + dump 落盘（env 契约注入）
        if ! (export ${CURRENT_ENVS}; \
              bash "${BUILD_SH}" --ops=all_gather_matmul_v2 --module=mc2 \
                   --soc=ascend950 --run_example --simulator); then
            echo "[driver] case ${CURRENT_CASE}: example/simulator FAILED"
            FAIL=$((FAIL + 1))
            CURRENT_CASE=""; CURRENT_ENVS=""
            continue
        fi
        # 2) 比对（§2.3 口径表硬断言 + 交叉核对，报告落盘）
        if (cd "${REPO_ROOT}" && python3 "${SCRIPT_DIR}/compare_dump.py" \
                --dump-dir "${DUMP_ROOT}/${CURRENT_CASE}" \
                --out-dir "${REPORT_ROOT}/${CURRENT_CASE}" \
                --summary "${SUMMARY}"); then
            PASS=$((PASS + 1))
        else
            FAIL=$((FAIL + 1))
        fi
        CURRENT_CASE=""; CURRENT_ENVS=""
        continue
    fi
    if [[ "${line}" == AGM_V2_CASE_ID=* ]]; then
        CURRENT_CASE="${line#AGM_V2_CASE_ID=}"
        CURRENT_ENVS="${line}"
    else
        CURRENT_ENVS="${CURRENT_ENVS} ${line}"
    fi
done <<< "${CASE_ENVS}"

echo "================================================================"
echo "[driver] total=${TOTAL} pass=${PASS} fail=${FAIL}"
echo "[driver] summary: ${SUMMARY}"
[[ ${DRY_RUN} -eq 1 ]] && echo "[driver] (dry-run: no execution)"

exit $((FAIL > 0 ? 1 : 0))
