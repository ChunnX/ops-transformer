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
# 逐例驱动：参数化 example（build.sh --run_example，env 注入 case 参数；
# 有卡默认直跑，--sim 回退 cannsim 仿真）→ dump 落盘 → compare_dump.py
# 按 §2.3 口径判定 → 汇总。
#
# P2 处置（2026-09-20，B9⑤ 报告 v1.1 §7-P2）：
#   * 权威判定 = compare_dump.py（§2.3 口径表）；example 退出码不再跳过权威
#     比对——dump 完整（meta.json + output_rank0.bin）即比对。仅当 example
#     异常退出且 dump 不完整（崩溃/未到 kernel 输出）时直接计 FAIL，不掩盖
#     真实崩溃；
#   * 逐例一行 driver_summary.jsonl（用例号/判定/关键指标/失败原因摘要），
#     失败例不再缺行（L1-112 教训：example 内嵌校验失败曾致 summary 缺行）；
#   * informational 用例（sim_cases.py INFO_CASES，大幅值原幅值记录，
#     AGM_V2_INFORMATIONAL=1 透传）判定加 INFO_ 前缀，不计入 PASS/FAIL
#     门禁（--include-informational 启用，默认排除）。
#
# Usage:
#   bash run_sim_cases.sh [--level L0|L1|all] [--case L0-01,L1-101]
#                        [--dump-root DIR] [--report-root DIR]
#                        [--dry-run] [--sim] [--include-conditional]
#                        [--include-informational]
#   默认执行 L0+L1 非条件例（L1-142 8 卡条件例需 --include-conditional；
#   INFO_CASES 大幅值 informational 变体需 --include-informational）。
#
# 前置条件：
#   * 参数化 example（examples/arch35/test_aclnn_all_gather_matmul_v2_fp16.cpp）
#     已支持本脚本的 env 契约（AGM_V2_*，见 tests/sim/README.md §接口契约）；
#   * build 目录空闲（与其他构建任务互斥）；
#   * CANN 环境已 source，且自定义算子包已出包安装（cust 模式依赖
#     vendors/<vendor>_transformer 或 ASCEND_CUSTOM_OPP_PATH）。
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
INCLUDE_INFORMATIONAL=0
USE_SIMULATOR=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --level) LEVEL="$2"; shift 2 ;;
        --case) CASE_FILTER="$2"; shift 2 ;;
        --dump-root) DUMP_ROOT="$2"; shift 2 ;;
        --report-root) REPORT_ROOT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --sim) USE_SIMULATOR=1; shift ;;
        --include-conditional) INCLUDE_CONDITIONAL=1; shift ;;
        --include-informational) INCLUDE_INFORMATIONAL=1; shift ;;
        *) echo "unknown arg: $1"; exit 2 ;;
    esac
done

# build.sh --run_example 调用形态（实证：A9 探针证据）：
#   * build.sh 仅接受 --simulator=<value>（camodel 经 cannsim 仿真），
#     裸 --simulator 形式 exit 1；--run_example 需位置实参 <op> <mode> [pkg_mode]；
#   * 有卡（默认）：不带 --simulator，example 直接在 NPU 上执行；
#   * --sim（无卡回退）：--simulator=camodel 仿真执行；
#   * --example_name 锁定参数化 example，避免 arch35 下 quant example
#     （test_aclnn_all_gather_matmul_v2.cpp）被同 pattern 匹配而互覆。
SIM_ARGS=""
if [[ ${USE_SIMULATOR} -eq 1 ]]; then
    SIM_ARGS="--simulator=camodel"
fi
RUN_EXAMPLE_ARGS="--run_example all_gather_matmul_v2 eager cust --vendor_name=custom --soc=ascend950"
if [[ -n "${SIM_ARGS}" ]]; then
    RUN_EXAMPLE_ARGS="${RUN_EXAMPLE_ARGS} ${SIM_ARGS}"
fi
RUN_EXAMPLE_ARGS="${RUN_EXAMPLE_ARGS} --example_name=test_aclnn_all_gather_matmul_v2_fp16"

mkdir -p "${DUMP_ROOT}" "${REPORT_ROOT}"
SUMMARY="${REPORT_ROOT}/summary.jsonl"
DRIVER_SUMMARY="${REPORT_ROOT}/driver_summary.jsonl"
: > "${SUMMARY}"
: > "${DRIVER_SUMMARY}"

# 从用例表取参数（tests/assets/sim_cases.py 为唯一用例源）。
# --include-conditional / --include-informational 透传给用例表枚举
# （L1-142 条件例 / INFO_CASES informational 变体默认排除）。
CASE_ENUM_ARGS=""
if [[ ${INCLUDE_CONDITIONAL} -eq 1 ]]; then
    CASE_ENUM_ARGS="${CASE_ENUM_ARGS} --include-conditional"
fi
if [[ ${INCLUDE_INFORMATIONAL} -eq 1 ]]; then
    CASE_ENUM_ARGS="${CASE_ENUM_ARGS} --include-informational"
fi
CASE_ENVS="$(python3 "${SCRIPT_DIR}/../assets/sim_cases.py" "${LEVEL}" --env \
    --dump-root="${DUMP_ROOT}"${CASE_ENUM_ARGS})"
if [[ -z "${CASE_ENVS}" ]]; then
    echo "[driver] no case selected (level=${LEVEL})"
    exit 2
fi

TOTAL=0
PASS=0
FAIL=0
INFO_TOTAL=0
INFO_PASS=0
INFO_FAIL=0
CURRENT_CASE=""
CURRENT_ENVS=""
FAILED_LINES=()   # 非 PASS 例（FAIL / INFO_FAIL）逐例一行，末尾汇总补全

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
            echo "[driver] dry-run: would execute: bash ${BUILD_SH} ${RUN_EXAMPLE_ARGS}"
            echo "${CURRENT_ENVS}" | sed 's/^/    export /'
            CURRENT_CASE=""; CURRENT_ENVS=""
            continue
        fi
        # informational 用例属性（AGM_V2_INFORMATIONAL 由用例表透传；example 不消费）
        IS_INFORMATIONAL=0
        if [[ "${CURRENT_ENVS}" == *"AGM_V2_INFORMATIONAL=1"* ]]; then
            IS_INFORMATIONAL=1
            echo "[driver] case ${CURRENT_CASE}: informational (verdict excluded from gate)"
        fi
        # 1) 参数化 example 执行 + dump 落盘（env 契约注入；有卡直跑 / --sim 仿真）
        #    P2：退出码仅记录，不跳过权威比对（崩溃与否由 dump 完整性区分）
        EXAMPLE_RC=0
        if ! (export ${CURRENT_ENVS}; \
              bash "${BUILD_SH}" ${RUN_EXAMPLE_ARGS}); then
            EXAMPLE_RC=1
            echo "[driver] case ${CURRENT_CASE}: example/simulator FAILED (exit != 0)"
        fi
        # 2) 权威比对（§2.3 口径表硬断言 + 交叉核对，报告落盘）
        #    dump 完整（meta.json + output_rank0.bin）即比对；不完整 = example
        #    崩溃/未到 kernel 输出，直接 FAIL（不掩盖真实崩溃）
        DUMP_DIR="${DUMP_ROOT}/${CURRENT_CASE}"
        COMPARE_RC=2
        if [[ -f "${DUMP_DIR}/meta.json" && -f "${DUMP_DIR}/output_rank0.bin" ]]; then
            if (cd "${REPO_ROOT}" && python3 "${SCRIPT_DIR}/compare_dump.py" \
                    --dump-dir "${DUMP_DIR}" \
                    --out-dir "${REPORT_ROOT}/${CURRENT_CASE}" \
                    --summary "${SUMMARY}"); then
                COMPARE_RC=0
            else
                COMPARE_RC=1
            fi
        else
            echo "[driver] case ${CURRENT_CASE}: dump incomplete (meta.json/output missing) — no authoritative compare"
        fi
        # 3) 逐例汇总（P2：每例一行 driver_summary.jsonl + 日志行；失败例不缺行）
        CASE_LINE="$(python3 "${SCRIPT_DIR}/driver_case_summary.py" \
            --case-id "${CURRENT_CASE}" \
            --informational "${IS_INFORMATIONAL}" \
            --example-rc "${EXAMPLE_RC}" \
            --compare-rc "${COMPARE_RC}" \
            --report "${REPORT_ROOT}/${CURRENT_CASE}/report_${CURRENT_CASE}.json" \
            --summary-out "${DRIVER_SUMMARY}")"
        echo "${CASE_LINE}"
        CASE_VERDICT="$(printf '%s' "${CASE_LINE}" | awk -F'|' '{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}')"
        if [[ "${CASE_VERDICT}" == "INFO_PASS" ]]; then
            INFO_TOTAL=$((INFO_TOTAL + 1)); INFO_PASS=$((INFO_PASS + 1))
        elif [[ "${CASE_VERDICT}" == "INFO_FAIL" ]]; then
            INFO_TOTAL=$((INFO_TOTAL + 1)); INFO_FAIL=$((INFO_FAIL + 1))
            FAILED_LINES+=("${CASE_LINE}")
        elif [[ "${CASE_VERDICT}" == "PASS" ]]; then
            PASS=$((PASS + 1))
        else
            FAIL=$((FAIL + 1))
            FAILED_LINES+=("${CASE_LINE}")
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
echo "[driver] total=${TOTAL} pass=${PASS} fail=${FAIL}" \
     "info_total=${INFO_TOTAL} info_pass=${INFO_PASS} info_fail=${INFO_FAIL}"
echo "[driver] summary: ${SUMMARY}"
echo "[driver] driver summary: ${DRIVER_SUMMARY}"
if [[ ${#FAILED_LINES[@]} -gt 0 ]]; then
    echo "[driver] ---- non-PASS case summary (case | verdict | metrics | reason) ----"
    for ln in "${FAILED_LINES[@]}"; do
        echo "${ln}"
    done
fi
[[ ${DRY_RUN} -eq 1 ]] && echo "[driver] (dry-run: no execution)"

# 门禁退出码：仅门禁用例（非 informational）FAIL 计入；informational 记录不影响
exit $((FAIL > 0 ? 1 : 0))
