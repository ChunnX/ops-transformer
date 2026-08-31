#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 脚本路径
TEST_RECURRENT_GATED_DELTA_RULE_SINGLE_SCRIPT="test_recurrent_gated_delta_rule_single.py"

# ====================== 结果输出目录 ======================
RESULT_DIR="output"
mkdir -p "${RESULT_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${RESULT_DIR}/run_${TIMESTAMP}.log"
CSV_FILE="${RESULT_DIR}/result_${TIMESTAMP}.csv"

# ====================== 执行区======================

# 算子调测
run_single() {
    echo "===== 执行单算子用例调测 ====="
    TEST_MODE=single CSV_FILE="${CSV_FILE}" python3 -m pytest -rA -s $TEST_RECURRENT_GATED_DELTA_RULE_SINGLE_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning 2>&1 | tee "${LOG_FILE}"
    local exit_code=${PIPESTATUS[0]}
    echo "执行日志: ${LOG_FILE}"
    echo "CSV: ${CSV_FILE}"
    return ${exit_code}
}

# RDV测试
run_rdv() {
    echo "===== 执行RDV参数集测试 ====="
    TEST_MODE=rdv CSV_FILE="${CSV_FILE}" python3 -m pytest -rA -s $TEST_RECURRENT_GATED_DELTA_RULE_SINGLE_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning 2>&1 | tee "${LOG_FILE}"
    local exit_code=${PIPESTATUS[0]}
    echo "执行日志: ${LOG_FILE}"
    echo "CSV: ${CSV_FILE}"
    return ${exit_code}
}

# 随机用例测试
run_random() {
    local count="${1:-100}"
    echo "===== 执行随机用例调测 ($count 条) ====="
    TEST_MODE=random RANDOM_CASE_COUNT=$count CSV_FILE="${CSV_FILE}" python3 -m pytest -rA -s $TEST_RECURRENT_GATED_DELTA_RULE_SINGLE_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning 2>&1 | tee "${LOG_FILE}"
    local exit_code=${PIPESTATUS[0]}
    echo "执行日志: ${LOG_FILE}"
    echo "CSV: ${CSV_FILE}"
    return ${exit_code}
}

# 随机用例测试（不跑golden，仅NPU执行，加快速度）
run_random_npu() {
    local count="${1:-100}"
    echo "===== 执行随机用例调测-仅NPU ($count 条) ====="
    TEST_MODE=random SKIP_GOLDEN=1 RANDOM_CASE_COUNT=$count CSV_FILE="${CSV_FILE}" python3 -m pytest -rA -s $TEST_RECURRENT_GATED_DELTA_RULE_SINGLE_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning 2>&1 | tee "${LOG_FILE}"
    local exit_code=${PIPESTATUS[0]}
    echo "执行日志: ${LOG_FILE}"
    echo "CSV: ${CSV_FILE}"
    return ${exit_code}
}

# mssanitizer 检测结论回填 CSV（mss_check 列；mssanitizer 报错时 pytest 仍 PASSED，必须由日志解析回填）
_backfill_mss_check() {
    local csv_file="$1"
    local log_file="$2"
    local mss_status="$3"
    python3 -c "
import csv

csv_file = '${csv_file}'
mss_status = '${mss_status}'
err_count = 0
with open('${log_file}', errors='ignore') as f:
    err_count = sum(1 for line in f if '====== ERROR' in line)

rows = []
with open(csv_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    for row in reader:
        row['mss_check'] = mss_status if err_count == 0 else f'{mss_status}(errors={err_count})'
        rows.append(row)

with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f'mss_check 回填完成: {mss_status}, ERROR块数: {err_count}')
"
}

# mssanitizer 检测（随机用例，仅NPU不跑golden；tool 可选 memcheck/racecheck/initcheck/synccheck）
run_mss() {
    local count="${1:-100}"
    local mss_tool="${MSS_TOOL:-memcheck}"
    local mss_bin="${MSSANITIZER_BIN:-mssanitizer}"
    # 默认只检本算子 kernel，跳过 ZerosLike/ViewCopy/TensorMove/rand 等旁路 kernel；置空 MSS_KERNEL 关闭过滤
    local mss_kernel="${MSS_KERNEL:-RecurrentGatedDeltaRule}"
    local kernel_opts=""
    if [ -n "$mss_kernel" ]; then
        kernel_opts="--kernel-name=$mss_kernel"
    fi
    local log_file="${RESULT_DIR}/mss_${TIMESTAMP}.log"
    echo "===== 执行 mssanitizer($mss_tool) 随机用例检测 ($count 条), kernel过滤: ${mss_kernel:-无}, 日志: $log_file ====="
    TEST_MODE=random SKIP_GOLDEN=1 RANDOM_CASE_COUNT=$count CSV_FILE="${CSV_FILE}" MSS_TOOL=${mss_tool} \
        "$mss_bin" --tool=$mss_tool $kernel_opts $MSS_EXTRA_OPTS -- \
        python3 -m pytest -rA -s $TEST_RECURRENT_GATED_DELTA_RULE_SINGLE_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning \
        2>&1 | tee "$log_file"
    local pytest_status=${PIPESTATUS[0]}
    # mssanitizer 检出错误时进程退出码仍为0，必须解析屏显中的 ERROR 块
    local mss_status="FAIL"
    if grep -q "====== ERROR" "$log_file"; then
        echo "===== FAIL: mssanitizer 检测到问题 ====="
        grep "====== ERROR" "$log_file"
    elif [ "$pytest_status" -ne 0 ]; then
        mss_status="CRASH"
        echo "===== CRASH: pytest 退出码 $pytest_status ====="
    else
        mss_status="PASS"
        echo "===== PASS: mssanitizer 未检测到问题 ====="
    fi
    if [ -f "${CSV_FILE}" ]; then
        _backfill_mss_check "${CSV_FILE}" "$log_file" "$mss_status"
    fi
    echo "执行日志: ${log_file}"
    echo "CSV: ${CSV_FILE}"
    [ "$mss_status" != "PASS" ] && exit 1
    return 0
}

# 显示帮助信息
show_help() {
    echo "用法: $0 [参数]"
    echo "参数说明："
    echo "  single       执行单算子用例调测"
    echo "  rdv          执行RDV参数集测试"
    echo "  random [N]   随机生成并执行N条用例（默认100）"
    echo "  random_npu [N]  随机生成并执行N条用例，不跑golden仅NPU执行（默认100）"
    echo "  mss [N]   随机生成并执行N条用例，仅NPU执行并用mssanitizer检测（默认100）"
    echo "  help         显示本帮助信息"
    echo "示例："
    echo "  $0 single       # 执行single模式"
    echo "  $0 rdv          # 执行rdv模式"
    echo "  $0 random 100   # 随机执行100条用例"
    echo "  $0 random_npu 100  # 随机执行100条用例（仅NPU）"
    echo "  $0 mss 10          # 随机执行10条用例并用mssanitizer memcheck检测"
    echo "mss 模式环境变量："
    echo "  MSS_TOOL=racecheck         # 检测工具，默认memcheck，可选racecheck/initcheck/synccheck"
    echo "  MSS_KERNEL=''              # kernel过滤，默认RecurrentGatedDeltaRule只检本算子；置空检测全部kernel"
    echo "  MSS_EXTRA_OPTS='--leak-check=yes --full-backtrace=yes'  # 追加mssanitizer参数"
    echo "  MSSANITIZER_BIN=<PATH>     # mssanitizer路径，默认取PATH"
}

# ====================== 主逻辑 ======================
# 检查传入的参数数量
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "错误：参数数量错误，用法 $0 {single|rdv|random [N]|random_npu [N]|mss [N]|help}"
    show_help
    exit 1
fi

# 根据参数执行对应函数
case "$1" in
    single)
        run_single
        ;;
    rdv)
        run_rdv
        ;;
    random)
        run_random "$2"
        ;;
    random_npu)
        run_random_npu "$2"
        ;;
    mss)
        run_mss "$2"
        ;;
    help)
        show_help
        ;;
    *)
        echo "错误：未知参数 '$1'，仅支持 single/rdv/random/random_npu/mss/help"
        show_help
        exit 1
        ;;
esac

exit 0
