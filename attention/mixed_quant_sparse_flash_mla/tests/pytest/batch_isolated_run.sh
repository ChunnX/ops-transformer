#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------
# 批量隔离执行脚本:
#   1. 获取指定路径下的所有 .pt 用例
#   2. 对每条用例单独拉起一个 pytest 进程执行, 进程间完全隔离
#      - 某条用例 device 越界/崩溃不会影响后续用例
#      - msprof 挂载在单用例进程上, 性能数据采集互不干扰
#   3. 启用msprof时, 每条用例执行 ITER_COUNT 次以取平均性能数据, 提升准确性
#      - 第1次: 精度校验 + 性能采集
#      - 第2~N次: 仅性能采集 (QSMLA_PERF_ONLY=1, 跳过精度校验与结果保存)
# 用法:
#   bash batch_isolated_run.sh [用例目录] [是否msprof采集: 0|1] [性能采集次数]
# 示例:
#   bash batch_isolated_run.sh                              # 默认 qsmla_testcase 目录, 不采集性能
#   bash batch_isolated_run.sh qsmla_testcase 1             # 启用 msprof, 默认采集5次
#   bash batch_isolated_run.sh qsmla_testcase 1 3           # 启用 msprof, 采集3次
# -----------------------------------------------------------------------------------------------------------

set -o pipefail

TEST_SCRIPT="test_mixed_quant_sparse_flash_mla_batch.py"
TESTCASE_DIR="${1:-qsmla_testcase}"
USE_MSPROF="${2:-0}"
ITER_COUNT="${3:-5}"
RESULT_XLSX="result.xlsx"
SUMMARY_LOG="batch_summary.log"
FAIL_LOG="batch_fail_list.log"

# 清理上一次的结果, 避免历史数据污染
[ -f "$RESULT_XLSX" ] && rm -f "$RESULT_XLSX"
[ -f "${RESULT_XLSX%.xlsx}_perf.xlsx" ] && rm -f "${RESULT_XLSX%.xlsx}_perf.xlsx"
: > "$SUMMARY_LOG"
: > "$FAIL_LOG"

# 清理旧的 PROF 文件夹, 避免与本次运行的数据混淆
for _prof_dir in PROF_*/; do
    [ -d "$_prof_dir" ] && rm -rf "$_prof_dir"
done

# 1. 获取指定路径下的所有用例路径
if [ ! -d "$TESTCASE_DIR" ]; then
    echo "错误: 用例目录不存在: $TESTCASE_DIR"
    exit 1
fi

mapfile -t CASE_FILES < <(find "$TESTCASE_DIR" -maxdepth 1 -name "*.pt" | sort)
TOTAL=${#CASE_FILES[@]}
if [ "$TOTAL" -eq 0 ]; then
    echo "错误: 目录 $TESTCASE_DIR 下未找到 .pt 用例"
    exit 1
fi

echo "共发现 $TOTAL 条用例, 目录: $TESTCASE_DIR , msprof采集: $USE_MSPROF, 性能采集次数: $ITER_COUNT"
echo "开始隔离批量执行..." | tee -a "$SUMMARY_LOG"

PASS=0
FAIL=0
FAIL_LIST=()

# 2. 对每条用例单独调用测试脚本, 独立进程; 启用msprof时每条用例执行ITER_COUNT次
i=0
for case_file in "${CASE_FILES[@]}"; do
    i=$((i+1))
    case_name=$(basename "$case_file")
    echo -e "\n===== [$i/$TOTAL] 执行用例: $case_name =====" | tee -a "$SUMMARY_LOG"

    case_pass=false

    # 确定实际执行次数: 不启用msprof时仅执行1次
    if [ "$USE_MSPROF" = "1" ]; then
        cur_iter_count="$ITER_COUNT"
    else
        cur_iter_count=1
    fi

    for iter in $(seq 1 "$cur_iter_count"); do
        if [ "$iter" -eq 1 ]; then
            iter_desc="精度+性能"
        else
            iter_desc="仅性能"
        fi
        echo "  --- 迭代 $iter/$cur_iter_count ($iter_desc) ---" | tee -a "$SUMMARY_LOG"

        PYTEST_CMD="python3 -m pytest -rA -s ${TEST_SCRIPT} -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning"
        if [ "$USE_MSPROF" = "1" ]; then
            if [ "$iter" -eq 1 ]; then
                RUN_CMD="QSMLA_TESTCASE_PATH=\"${case_file}\" msprof ${PYTEST_CMD}"
            else
                RUN_CMD="QSMLA_TESTCASE_PATH=\"${case_file}\" QSMLA_PERF_ONLY=1 msprof ${PYTEST_CMD}"
            fi
        else
            RUN_CMD="QSMLA_TESTCASE_PATH=\"${case_file}\" ${PYTEST_CMD}"
        fi

        eval "$RUN_CMD" 2>&1 | tee -a "$SUMMARY_LOG"
        status=${PIPESTATUS[0]}

        if [ "$status" -ne 0 ]; then
            if [ "$iter" -eq 1 ]; then
                echo "  [FAIL] 第1次迭代失败, 跳过后续迭代" | tee -a "$SUMMARY_LOG"
            else
                echo "  [WARN] 第${iter}次迭代失败, 继续下一次" | tee -a "$SUMMARY_LOG"
            fi
            break
        fi

        if [ "$iter" -eq 1 ]; then
            case_pass=true
        fi
    done

    if [ "$case_pass" = true ]; then
        PASS=$((PASS+1))
        echo "[PASS] $case_name" | tee -a "$SUMMARY_LOG"
    else
        FAIL=$((FAIL+1))
        FAIL_LIST+=("$case_name")
        echo "[FAIL] $case_name" | tee -a "$SUMMARY_LOG"
        echo "$case_name" >> "$FAIL_LOG"
    fi
done

# 3. 收集性能数据（仅当启用 msprof 采集时）
if [ "$USE_MSPROF" = "1" ]; then
    echo -e "\n========== 收集性能数据 ==========" | tee -a "$SUMMARY_LOG"
    python3 collect_perf_data.py --test_result_path "$RESULT_XLSX" --iter_count "$ITER_COUNT" 2>&1 | tee -a "$SUMMARY_LOG"
fi

# 汇总
echo -e "\n========== 批量执行汇总 ==========" | tee -a "$SUMMARY_LOG"
echo "总计: $TOTAL  通过: $PASS  失败: $FAIL" | tee -a "$SUMMARY_LOG"
if [ "$FAIL" -gt 0 ]; then
    echo "失败用例:" | tee -a "$SUMMARY_LOG"
    for f in "${FAIL_LIST[@]}"; do
        echo "  - $f" | tee -a "$SUMMARY_LOG"
    done
fi
echo "详细日志:  $SUMMARY_LOG"
echo "失败清单:  $FAIL_LOG"
echo "结果表格:  $RESULT_XLSX"
if [ "$USE_MSPROF" = "1" ]; then
    echo "性能表格:  ${RESULT_XLSX%.xlsx}_perf.xlsx"
fi

exit 0
