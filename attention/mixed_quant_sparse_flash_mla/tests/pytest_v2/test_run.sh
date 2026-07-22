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

set -euo pipefail

QSMLA_PT_SAVE_SCRIPT="./batch/test_mixed_quant_sparse_flash_mla_pt_save.py"
TEST_QSMLA_PT_BATCH_SCRIPT="test_mixed_quant_sparse_flash_mla_batch.py"
TEST_QSMLA_PT_BATCH_GRAPH_SCRIPT="test_mixed_quant_sparse_flash_mla_batch_graph.py"
TEST_QSMLA_SINGLE_SCRIPT="test_mixed_quant_sparse_flash_mla_single.py"
TEST_QSMLA_CONSISTENCY_SCRIPT="test_mixed_quant_sparse_flash_mla_batch_consistency.py"
QSMLA_CONSISTENCY_PREPARE_SCRIPT="./batch/mixed_quant_sparse_flash_mla_consistency_prepare.py"

PT_SAVE_DIR="mqsmla_testcase"
EXCEL_FILE="./excel/example.xlsx"
SHEET_NAME="decode"
KEEP_PT=false
BATCH_TEST_MODE=1
RESULT_PATH="./mqsmla_result.xlsx"
DEVICE_ID=0
CONSISTENCY_MODES=""
CONSISTENCY_PROFILE="blackbox-basic"
CONSISTENCY_SEED="20260716"
CONSISTENCY_RANDOMIZE=0
CONSISTENCY_SPLIT_GROUPS=""
CONSISTENCY_TOKEN_SPLIT_SIZES=""
CONSISTENCY_MODE3_BATCH=0
CONSISTENCY_COMMON_TOKENS=""
CONSISTENCY_DERIVED_EXTRA=""
PREPARE_REPORT_DIR=""
CONSISTENCY_CONFIG_FILE=""
CONSISTENCY_REPORT_ROOT="./result/mqsmla_consistency"
CONSISTENCY_DETAIL_SET=0
CONSISTENCY_SELECTOR_SET=0
CONSISTENCY_PROBE_ATTEMPTS=1

# ====================== 执行区 ======================

run_single() {
    echo "===== 执行单用例算子调测 ====="
    python3 -m pytest -rA -s $TEST_QSMLA_SINGLE_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
}

run_batch_save() {
    echo "===== 执行batch_save：从excel读取用例并保存pt文件 ====="
    echo "  excel: $EXCEL_FILE"
    echo "  sheet: $SHEET_NAME"
    echo "  pt目录: $PT_SAVE_DIR"
    export MQSMLA_EXCEL="$EXCEL_FILE"
    export MQSMLA_SHEET="$SHEET_NAME"
    export MQSMLA_PT_DIR="$PT_SAVE_DIR"
    python3 -m pytest -rA -s $QSMLA_PT_SAVE_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
    if [ $? -ne 0 ]; then
        echo "batch_save 执行失败，退出"
        exit 1
    fi
    echo -e "\n===== batch_save 完成！pt文件保存在 $PT_SAVE_DIR 目录 ====="
}

run_batch_exec() {
    echo "===== 执行batch_exec：读取pt文件并执行NPU测试 ====="

    if [ ! -d "$PT_SAVE_DIR" ]; then
        echo "错误: pt文件目录不存在: $PT_SAVE_DIR，请先执行 batch_save 生成pt文件"
        exit 1
    fi

    pt_count=$(ls -1 $PT_SAVE_DIR/*.pt 2>/dev/null | wc -l)
    if [ $pt_count -eq 0 ]; then
        echo "错误: 目录中没有找到.pt文件: $PT_SAVE_DIR，请先执行 batch_save 生成pt文件"
        exit 1
    fi

    echo "找到 $pt_count 个pt文件，开始执行NPU测试"
    export MQSMLA_PT_DIR="$PT_SAVE_DIR"
    export MQSMLA_BATCH_TEST_MODE="$BATCH_TEST_MODE"
    export MQSMLA_EXCEL_PATH="$EXCEL_FILE"
    export MQSMLA_RESULT_SAVE_PATH="$RESULT_PATH"
    export MQSMLA_DEVICE_ID="$DEVICE_ID"
    python3 -m pytest -rA -s $TEST_QSMLA_PT_BATCH_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
    if [ $? -ne 0 ]; then
        echo "batch_exec 执行失败"
        exit 1
    fi

    echo -e "\n===== batch_exec 完成！====="
}

run_batch_exec_graph() {
    echo "===== 执行batch_exec_graph：读取pt文件并执行Graph模式NPU测试 ====="

    if [ ! -d "$PT_SAVE_DIR" ]; then
        echo "错误: pt文件目录不存在: $PT_SAVE_DIR，请先执行 batch_save 生成pt文件"
        exit 1
    fi

    pt_count=$(ls -1 $PT_SAVE_DIR/*.pt 2>/dev/null | wc -l)
    if [ $pt_count -eq 0 ]; then
        echo "错误: 目录中没有找到.pt文件: $PT_SAVE_DIR，请先执行 batch_save 生成pt文件"
        exit 1
    fi

    echo "找到 $pt_count 个pt文件，开始执行Graph模式NPU测试"
    export MQSMLA_PT_DIR="$PT_SAVE_DIR"
    export MQSMLA_BATCH_TEST_MODE="$BATCH_TEST_MODE"
    export MQSMLA_EXCEL_PATH="$EXCEL_FILE"
    export MQSMLA_RESULT_SAVE_PATH="$RESULT_PATH"
    export MQSMLA_DEVICE_ID="$DEVICE_ID"
    python3 -m pytest -rA -s $TEST_QSMLA_PT_BATCH_GRAPH_SCRIPT -v -m graph -W ignore::UserWarning -W ignore::DeprecationWarning
    if [ $? -ne 0 ]; then
        echo "batch_exec_graph 执行失败"
        exit 1
    fi

    echo -e "\n===== batch_exec_graph 完成！====="
}

run_batch() {
    echo "===== 执行batch：从excel批量生成pt并执行NPU测试 ====="

    echo -e "\n===== 第一步：batch_save - 从excel生成pt文件 ====="
    run_batch_save
    if [ $? -ne 0 ]; then
        exit 1
    fi

    echo -e "\n===== 第二步：batch_exec - 读取pt文件执行NPU测试 ====="
    run_batch_exec
    if [ $? -ne 0 ]; then
        exit 1
    fi

    if [ "$KEEP_PT" = false ]; then
        echo -e "\n===== 清理pt文件（KEEP_PT=false） ====="
        [ -n "$PT_SAVE_DIR" ] && rm -rf $PT_SAVE_DIR
        echo "pt文件已清理"
    else
        echo -e "\n===== 保留pt文件（KEEP_PT=true）保存在 $PT_SAVE_DIR ====="
    fi

    echo -e "\n===== batch 全流程执行完成！====="
}

run_consistency() {
    echo "===== 执行 MQSMLA batch consistency（metadata + 主算子）====="
    if [ ! -d "$PT_SAVE_DIR" ]; then
        echo "错误: pt文件目录不存在: $PT_SAVE_DIR"
        return 1
    fi
    if [ -n "$CONSISTENCY_CONFIG_FILE" ] && [ "$CONSISTENCY_DETAIL_SET" -eq 1 ]; then
        echo "错误: --config-file 不能与模式或详细参数混用"
        return 1
    fi
    if [ -n "$CONSISTENCY_CONFIG_FILE" ]; then
        if [ ! -f "$CONSISTENCY_CONFIG_FILE" ]; then
            echo "错误: 配置文件不存在: $CONSISTENCY_CONFIG_FILE"
            return 1
        fi
    else
        if ! [[ "$CONSISTENCY_SEED" =~ ^[0-9]+$ ]]; then
            echo "错误: --seed 必须是非负整数，当前值: $CONSISTENCY_SEED"
            return 1
        fi
        if ! python3 - "$CONSISTENCY_MODES" "$CONSISTENCY_PROFILE" <<'PY'
import sys
import warnings
from batch.consistency.config import ModeRegistry
warnings.simplefilter("default", DeprecationWarning)
ModeRegistry.resolve(modes=sys.argv[1] or None, profile=sys.argv[2] or None, warn_legacy=True)
PY
        then
            echo "错误: 一致性模式或profile配置无效"
            return 1
        fi
    fi
    if ! [[ "$CONSISTENCY_PROBE_ATTEMPTS" =~ ^[0-9]+$ ]] || \
        [ "$CONSISTENCY_PROBE_ATTEMPTS" -lt 1 ] || [ "$CONSISTENCY_PROBE_ATTEMPTS" -gt 100 ]; then
        echo "错误: --probe-attempts 必须是1到100的整数，当前值: $CONSISTENCY_PROBE_ATTEMPTS"
        return 1
    fi

    local case_files=()
    local pt_file
    for pt_file in "$PT_SAVE_DIR"/*.pt; do
        [ -f "$pt_file" ] || continue
        case_files+=("$pt_file")
    done
    local total=${#case_files[@]}
    if [ "$total" -eq 0 ]; then
        echo "错误: 目录 $PT_SAVE_DIR 下未找到 .pt 用例"
        return 1
    fi

    local run_id
    run_id="$(date +%Y%m%d_%H%M%S)_$$"
    local run_dir="$CONSISTENCY_REPORT_ROOT/$run_id"
    mkdir -p "$run_dir"
    local pytest_log="$run_dir/pytest.log"

    export MQSMLA_PT_DIR="$PT_SAVE_DIR"
    export MQSMLA_DEVICE_ID="$DEVICE_ID"
    export MQSMLA_CONSISTENCY_TEST=1
    export MQSMLA_CONSISTENCY_REPORT_DIR="$run_dir"
    if [ -n "$CONSISTENCY_CONFIG_FILE" ]; then
        export MQSMLA_CONSISTENCY_CONFIG_FILE="$CONSISTENCY_CONFIG_FILE"
        unset MQSMLA_CONSISTENCY_MODES MQSMLA_CONSISTENCY_PROFILE
        unset MQSMLA_CONSISTENCY_SEED MQSMLA_CONSISTENCY_RANDOMIZE
        unset MQSMLA_CONSISTENCY_SPLIT_GROUPS MQSMLA_CONSISTENCY_TOKEN_SPLIT_SIZES
        unset MQSMLA_CONSISTENCY_MODE3_BATCH MQSMLA_CONSISTENCY_COMMON_TOKENS
        unset MQSMLA_CONSISTENCY_DERIVED_EXTRA_TOKENS
    else
        unset MQSMLA_CONSISTENCY_CONFIG_FILE
        export MQSMLA_CONSISTENCY_SEED="$CONSISTENCY_SEED"
        export MQSMLA_CONSISTENCY_RANDOMIZE="$CONSISTENCY_RANDOMIZE"
        export MQSMLA_CONSISTENCY_SPLIT_GROUPS="$CONSISTENCY_SPLIT_GROUPS"
        export MQSMLA_CONSISTENCY_TOKEN_SPLIT_SIZES="$CONSISTENCY_TOKEN_SPLIT_SIZES"
        export MQSMLA_CONSISTENCY_MODE3_BATCH="$CONSISTENCY_MODE3_BATCH"
        export MQSMLA_CONSISTENCY_COMMON_TOKENS="$CONSISTENCY_COMMON_TOKENS"
        export MQSMLA_CONSISTENCY_DERIVED_EXTRA_TOKENS="$CONSISTENCY_DERIVED_EXTRA"
        if [ -n "$CONSISTENCY_MODES" ]; then
            export MQSMLA_CONSISTENCY_MODES="$CONSISTENCY_MODES"
            unset MQSMLA_CONSISTENCY_PROFILE
        else
            unset MQSMLA_CONSISTENCY_MODES
            export MQSMLA_CONSISTENCY_PROFILE="$CONSISTENCY_PROFILE"
        fi
    fi

    local process_failures=0
    local capability_report="$run_dir/runtime_capability.json"
    export MQSMLA_CONSISTENCY_CAPABILITY_REPORT="$capability_report"
    echo "共 $total 条用例；结构化结果目录: $run_dir"
    echo "先执行level 3真实NPU调度探针（$CONSISTENCY_PROBE_ATTEMPTS 次）" | tee -a "$pytest_log"
    local probe_status
    if python3 -m batch.consistency.runtime_capability \
        --output "$capability_report" \
        --device-id "$DEVICE_ID" \
        --attempts "$CONSISTENCY_PROBE_ATTEMPTS" 2>&1 | tee -a "$pytest_log"
    then
        probe_status=0
    else
        probe_status=${PIPESTATUS[0]}
    fi
    if [ "$probe_status" -ne 0 ] && [ "$probe_status" -ne 2 ]; then
        process_failures=$((process_failures + 1))
    fi

    local index=0
    local case_file
    for case_file in "${case_files[@]}"; do
        index=$((index + 1))
        echo "===== [$index/$total] $(basename "$case_file") =====" | tee -a "$pytest_log"
        if QSAS_TESTCASE_PATH="$case_file" \
            python3 -m pytest -rA -s "$TEST_QSMLA_CONSISTENCY_SCRIPT" \
            -v -m consistency 2>&1 | tee -a "$pytest_log"
        then
            :
        else
            process_failures=$((process_failures + 1))
        fi
    done

    local aggregate_status
    if python3 -m batch.consistency.result_artifact \
        --report-dir "$run_dir" \
        --operator mqsmla \
        --expected-case-count "$total" \
        --execution-failure-count "$process_failures" | tee -a "$pytest_log"
    then
        aggregate_status=0
    else
        aggregate_status=${PIPESTATUS[0]}
    fi
    echo "权威汇总: $run_dir/consistency_result_summary.json"
    echo "逐模式结果: $run_dir/results"
    echo "运行时能力证据: $capability_report"
    echo "pytest文本日志（仅诊断）: $pytest_log"
    return "$aggregate_status"
}

run_consistency_prepare() {
    if [ -z "$PREPARE_REPORT_DIR" ]; then
        echo "错误: consistency_prepare 必须指定 --report-dir"
        exit 1
    fi
    echo "===== 执行 MQSMLA 无NPU一致性离线预检 ====="
    python3 "$QSMLA_CONSISTENCY_PREPARE_SCRIPT" \
        --excel "$EXCEL_FILE" --sheet "$SHEET_NAME" --pt-dir "$PT_SAVE_DIR" \
        --report-dir "$PREPARE_REPORT_DIR" --seed "$CONSISTENCY_SEED"
}

show_help() {
    echo "用法: $0 <命令> [选项]"
    echo ""
    echo "命令说明："
    echo "  single             执行单算子用例调测"
    echo "  batch_save         从excel读取用例，golden计算并保存pt文件"
    echo "  batch_exec         批量读取pt文件并执行NPU测试(CI模式)"
    echo "  batch_exec_graph   批量读取pt文件并执行Graph模式NPU测试"
    echo "  batch              全流程：从excel生成pt + NPU测试"
    echo "  consistency        执行level 3 batch一致性模式1-4"
    echo "  consistency_prepare 无NPU验收：Excel -> CPU golden -> PT -> 模式1-4派生PT"
    echo "  help               显示本帮助信息"
    echo ""
    echo "选项（batch_save/batch_exec/batch_exec_graph/batch 命令支持）："
    echo "  --excel <路径>      指定excel文件路径（默认: ./excel/example.xlsx）"
    echo "  --sheet <名称>      指定excel sheet名（默认: decode）"
    echo "  --pt-dir <目录>     指定pt文件保存/读取目录（默认: mqsmla_testcase）"
    echo "  --result <路径>     指定结果保存路径（默认: ./mqsmla_result.xlsx）"
    echo "  --report-dir <目录> consistency结构化结果根目录；prepare为JSON目录"
    echo "  --probe-attempts <n> level 3真实NPU调度前置探针次数（1-100，默认1）"
    echo "  --device-id <id>    指定NPU设备ID（默认: 0）"
    echo "  --mode <0|1>        批跑模式: 0=全量批跑(默认), 1=按表格中case批跑"
    echo "  --keep-pt           执行完成后保留pt文件（默认清理，仅batch命令）"
    echo "  --modes <名称>      命名模式，例如 reorder,split,token-split,shape-change"
    echo "  -M <数字>           数字模式迁移兼容（已废弃）"
    echo "  --profile <名称>    命名模式组合（默认 blackbox-basic）"
    echo "  --config-file <JSON> versioned正式配置；不能与模式/详细参数混用"
    echo "  --split-groups <组> 模式2显式分组，例如 '0,1,2;3,4,5,6,7'"
    echo "  --token-split-sizes 模式3分段，例如 '4096,12288'"
    echo "  --mode3-batch <id>  模式3/4源batch id"
    echo "  --common-tokens <n> 模式4共同前缀（需求示例300）"
    echo "  --derived-extra <n> 模式4派生额外token（需求示例200）"
    echo "  --seed <n>          fuzz随机种子"
    echo "  --randomize         仅在fuzz时随机模式2分组"
    echo ""
    echo "环境变量（.py文件读取）："
    echo "  MQSMLA_PT_DIR              pt文件目录（batch_save/batch_exec/batch_exec_graph）"
    echo "  MQSMLA_EXCEL               excel文件路径（batch_save）"
    echo "  MQSMLA_SHEET               excel sheet名（batch_save）"
    echo "  MQSMLA_EXCEL_PATH          excel文件路径（batch_exec/batch_exec_graph，mode=1时使用）"
    echo "  MQSMLA_BATCH_TEST_MODE     批跑模式: 0=全量, 1=按表格（batch_exec/batch_exec_graph）"
    echo "  MQSMLA_RESULT_SAVE_PATH    结果保存路径（batch_exec/batch_exec_graph）"
    echo "  MQSMLA_DEVICE_ID           NPU设备ID（batch_exec/batch_exec_graph）"
    echo ""
    echo "示例："
    echo "  $0 single"
    echo "  $0 batch_save"
    echo "  $0 batch_save --excel my.xlsx --sheet decode --pt-dir my_pt"
    echo "  $0 batch_exec                                          # 全量批跑"
    echo "  $0 batch_exec --mode 1 --excel ./excel/testcase.xlsx   # 按表格批跑"
    echo "  $0 batch_exec --pt-dir my_pt --result ./my_result.xlsx"
    echo "  $0 batch_exec_graph --mode 1                           # Graph模式按表格批跑"
    echo "  $0 batch --keep-pt"
    echo "  $0 consistency --pt-dir my_pt --config-file suite.json --report-dir ./result"
    echo "  $0 consistency_prepare --excel cases.xlsx --sheet MQSMLA --pt-dir /tmp/mq_pt --report-dir /tmp/mq_report"
    echo "  $0 batch --excel my.xlsx --sheet decode --pt-dir my_pt --mode 1 --keep-pt"
}

# ====================== 主逻辑 ======================

if [ $# -lt 1 ]; then
    echo "错误：必须传入至少一个命令参数"
    show_help
    exit 1
fi

COMMAND="$1"
shift

if [ "$COMMAND" = "batch_save" ] || [ "$COMMAND" = "batch_exec" ] || [ "$COMMAND" = "batch_exec_graph" ] || [ "$COMMAND" = "batch" ] || [ "$COMMAND" = "consistency" ] || [ "$COMMAND" = "consistency_prepare" ]; then
    while [ $# -gt 0 ]; do
        case "$1" in
            --excel|--sheet|--pt-dir|--result|--report-dir|--device-id|--mode|--modes|-M|--profile|--config-file|--split-groups|--token-split-sizes|--mode3-batch|--common-tokens|--derived-extra|--seed|--probe-attempts)
                if [ $# -lt 2 ] || [ -z "$2" ]; then
                    echo "错误：$1 需要参数值"
                    exit 1
                fi
                ;;
        esac
        case "$1" in
            --excel)
                if [ $# -lt 2 ]; then
                    echo "错误：--excel 需要参数值"
                    exit 1
                fi
                EXCEL_FILE="$2"
                shift 2
                ;;
            --sheet)
                if [ $# -lt 2 ]; then
                    echo "错误：--sheet 需要参数值"
                    exit 1
                fi
                SHEET_NAME="$2"
                shift 2
                ;;
            --pt-dir)
                if [ $# -lt 2 ]; then
                    echo "错误：--pt-dir 需要参数值"
                    exit 1
                fi
                PT_SAVE_DIR="$2"
                shift 2
                ;;
            --result)
                if [ $# -lt 2 ]; then
                    echo "错误：--result 需要参数值"
                    exit 1
                fi
                RESULT_PATH="$2"
                shift 2
                ;;
            --report-dir)
                if [ "$COMMAND" = "consistency" ]; then
                    CONSISTENCY_REPORT_ROOT="$2"
                else
                    PREPARE_REPORT_DIR="$2"
                fi
                shift 2
                ;;
            --device-id)
                if [ $# -lt 2 ]; then
                    echo "错误：--device-id 需要参数值"
                    exit 1
                fi
                DEVICE_ID="$2"
                shift 2
                ;;
            --mode)
                if [ $# -lt 2 ]; then
                    echo "错误：--mode 需要参数值"
                    exit 1
                fi
                BATCH_TEST_MODE="$2"
                shift 2
                ;;
            --keep-pt)
                if [ "$COMMAND" != "batch" ]; then
                    echo "错误：--keep-pt 仅适用于 batch 命令"
                    exit 1
                fi
                KEEP_PT=true
                shift
                ;;
            --modes|-M)
                if [ "$CONSISTENCY_SELECTOR_SET" -eq 1 ]; then
                    echo "错误：--modes/-M 与 --profile 只能指定一个"
                    exit 1
                fi
                CONSISTENCY_MODES="$2"
                CONSISTENCY_PROFILE=""
                CONSISTENCY_SELECTOR_SET=1
                CONSISTENCY_DETAIL_SET=1
                shift 2
                ;;
            --profile)
                if [ "$CONSISTENCY_SELECTOR_SET" -eq 1 ]; then
                    echo "错误：--modes/-M 与 --profile 只能指定一个"
                    exit 1
                fi
                CONSISTENCY_PROFILE="$2"
                CONSISTENCY_MODES=""
                CONSISTENCY_SELECTOR_SET=1
                CONSISTENCY_DETAIL_SET=1
                shift 2
                ;;
            --config-file)
                CONSISTENCY_CONFIG_FILE="$2"
                shift 2
                ;;
            --split-groups)
                CONSISTENCY_SPLIT_GROUPS="$2"
                CONSISTENCY_DETAIL_SET=1
                shift 2
                ;;
            --token-split-sizes)
                CONSISTENCY_TOKEN_SPLIT_SIZES="$2"
                CONSISTENCY_DETAIL_SET=1
                shift 2
                ;;
            --mode3-batch)
                CONSISTENCY_MODE3_BATCH="$2"
                CONSISTENCY_DETAIL_SET=1
                shift 2
                ;;
            --common-tokens)
                CONSISTENCY_COMMON_TOKENS="$2"
                CONSISTENCY_DETAIL_SET=1
                shift 2
                ;;
            --derived-extra)
                CONSISTENCY_DERIVED_EXTRA="$2"
                CONSISTENCY_DETAIL_SET=1
                shift 2
                ;;
            --seed)
                CONSISTENCY_SEED="$2"
                if [ "$COMMAND" = "consistency" ]; then
                    CONSISTENCY_DETAIL_SET=1
                fi
                shift 2
                ;;
            --probe-attempts)
                if [ "$COMMAND" != "consistency" ]; then
                    echo "错误：--probe-attempts 仅适用于 consistency 命令"
                    exit 1
                fi
                CONSISTENCY_PROBE_ATTEMPTS="$2"
                shift 2
                ;;
            --randomize)
                CONSISTENCY_RANDOMIZE=1
                CONSISTENCY_DETAIL_SET=1
                shift
                ;;
            *)
                echo "错误：未知选项 '$1'"
                show_help
                exit 1
                ;;
        esac
    done
fi

case "$COMMAND" in
    single)
        run_single
        ;;
    batch_save)
        run_batch_save
        ;;
    batch_exec)
        run_batch_exec
        ;;
    batch_exec_graph)
        run_batch_exec_graph
        ;;
    batch)
        run_batch
        ;;
    consistency)
        run_consistency
        ;;
    consistency_prepare)
        run_consistency_prepare
        ;;
    help)
        show_help
        ;;
    *)
        echo "错误：未知命令 '$COMMAND'"
        show_help
        exit 1
        ;;
esac
