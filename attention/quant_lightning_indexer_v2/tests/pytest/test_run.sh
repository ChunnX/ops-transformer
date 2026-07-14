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

# ====================== 配置区（默认值，可通过命令行 -E/-S/-P 覆盖）======================
# 需要读取的用例excel表格路径：
DEFAULT_EXCEL="./excel/test_cases.xlsx"
# 用例pt的文件存放路径：
DEFAULT_PT_PATH="./pt_path"

# 脚本路径
QLIV2_PT_SAVE_SCRIPT="./batch/quant_lightning_indexer_v2_pt_save.py"
LIST_PT_SCRIPT="./batch/list_pt_from_excel.py"
TEST_QLIV2_BATCH_SCRIPT="test_quant_lightning_indexer_v2_batch.py"
REPLACE_PATH_SCRIPT="./batch/replace_path.py"
TEST_QLIV2_SINGLE_SCRIPT="test_quant_lightning_indexer_v2_single.py"

# ====================== 执行区======================

# 单用例算子调测
run_single() {
    echo "===== 执行单用例算子调测 ====="
    python3 -m pytest -rA -s $TEST_QLIV2_SINGLE_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
}

# 用例批量生成调试
# 用法: run_batch excel_path sheet pt_path run_mode
# 从 Excel 生成 pt 文件 → 替换路径 → 执行 NPU 测试
run_batch() {
    local excel_path="$1"
    local excel_sheet="$2"
    local pt_path="$3"
    local run_mode="${4:-eager}"
    echo "===== 执行用例批量生成测试 (模式: ${run_mode}) ====="
    echo "Excel 文件: $excel_path"
    echo "Sheet 页: $excel_sheet"
    echo "pt  目录: $pt_path"

    echo -e "\n===== 第一步：执行quant_lightning_indexer_v2_pt_save.py ====="
    python3 $QLIV2_PT_SAVE_SCRIPT "$excel_path" "$pt_path" --sheet "$excel_sheet"
    if [ $? -ne 0 ]; then
        echo "quant_lightning_indexer_v2_pt_save.py 执行失败，退出"
        exit 1
    fi

    echo -e "\n===== 第二步：替换test_quant_lightning_indexer_v2_batch.py中的路径 ====="
    python3 $REPLACE_PATH_SCRIPT $TEST_QLIV2_BATCH_SCRIPT "$pt_path"
    if [ $? -ne 0 ]; then
        echo "替换路径失败，退出"
        exit 1
    fi

    echo -e "\n===== 第三步：执行pytest命令 (QLIV2_RUN_MODE=${run_mode}) ====="
    QLIV2_RUN_MODE="${run_mode}" python3 -m pytest -rA -s $TEST_QLIV2_BATCH_SCRIPT -v -m ci
    if [ $? -ne 0 ]; then
        echo "pytest执行失败"
        exit 1
    fi

    cp test_quant_lightning_indexer_v2_batch.py.bak test_quant_lightning_indexer_v2_batch.py

    echo -e "\n=====执行完成！====="
}

# 根据 Excel 表格筛选 pt 文件并批量执行 NPU 测试
# 用法: run_batch_exec [excel_path] [sheet] [pt_path] [run_mode]
run_batch_exec() {
    local excel_path="$1"
    local excel_sheet="$2"
    local pt_path="$3"
    local run_mode="${4:-eager}"
    echo "===== 根据 Excel 表格批量执行 NPU 测试 (模式: ${run_mode}) ====="

    if [ ! -f "$excel_path" ]; then
        echo "错误: Excel 文件不存在: $excel_path"
        exit 1
    fi
    if [ ! -d "$pt_path" ]; then
        echo "错误: pt 目录不存在: $pt_path"
        exit 1
    fi
    echo "Excel 文件: $excel_path"
    echo "Sheet 页: $excel_sheet"
    echo "pt  目录: $pt_path"

    echo -e "\n===== 第一步：从 Excel 提取用例列表并匹配 pt 文件 ====="
    PT_FILE_LIST=$(python3 "$LIST_PT_SCRIPT" "$excel_path" "$pt_path" --sheet "$excel_sheet")
    if [ $? -ne 0 ]; then
        echo "提取 pt 文件列表失败"
        exit 1
    fi
    pt_count=$(echo "$PT_FILE_LIST" | tr ',' '\n' | wc -l)
    echo "匹配到 ${pt_count} 个 pt 文件"

    echo -e "\n===== 第二步：执行 pytest (QLIV2_RUN_MODE=${run_mode}) ====="
    QLIV2_RUN_MODE="${run_mode}" QLIV2_PT_FILE_LIST="$PT_FILE_LIST" \
        python3 -m pytest -rA -s "$TEST_QLIV2_BATCH_SCRIPT" -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
    if [ $? -ne 0 ]; then
        echo "pytest 执行失败"
        exit 1
    fi

    echo -e "\n=====执行完成！====="
}

# 显示帮助信息
show_help() {
    echo "用法: $0 <command> [选项]"
    echo ""
    echo "命令说明："
    echo "  single              执行单算子用例调测"
    echo ""
    echo "  batch              从 Excel 生成 pt 文件并批量执行 NPU 测试"
    echo "                        -E excel_path  指定 Excel 文件路径（默认: ./excel/test_cases.xlsx）"
    echo "                        -S sheet_name  指定 Sheet 页名（默认: Sheet1）"
    echo "                        -P pt_path     指定 pt 文件存放目录（默认: ./pt_path）"
    echo "                        -M run_mode    eager|graph（默认: eager）"
    echo ""
    echo "  batch_exec          根据 Excel 表格筛选已有 pt 文件并执行 NPU 测试"
    echo "                        -E excel_path  指定 Excel 文件路径（默认: ./excel/test_cases.xlsx）"
    echo "                        -S sheet_name  指定 Sheet 页名（默认: Sheet1）"
    echo "                        -P pt_path     指定 pt 文件目录（默认: ./pt_path）"
    echo "                        -M run_mode    eager|graph（默认: eager）"
    echo ""
    echo "  help                显示本帮助信息"
    echo ""
    echo "示例："
    echo "  $0 single"
    echo "  $0 batch              # 从 excel 生成 pt 后批量执行（默认值）"
    echo "  $0 batch -E ./excel/test_cases.xlsx -S Sheet1 -P ./pt_path"
    echo "  $0 batch -M graph     # graph 模式"
    echo "  $0 batch_exec"
    echo "  $0 batch_exec -E ./excel/test_cases.xlsx -S Sheet1 -P ./pt_path"
    echo "  $0 batch_exec -E ./excel/test_cases.xlsx -M graph"
}

# ====================== 主逻辑 ======================
if [ $# -lt 1 ]; then
    echo "错误：必须传入至少一个参数"
    show_help
    exit 1
fi

COMMAND="$1"
shift

# 解析可选参数
EXCEL_PATH=""
EXCEL_SHEET="Sheet1"
PT_PATH="$DEFAULT_PT_PATH"
RUN_MODE="eager"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -E)
            EXCEL_PATH="$2"
            shift 2
            ;;
        -S)
            EXCEL_SHEET="$2"
            shift 2
            ;;
        -P)
            PT_PATH="$2"
            shift 2
            ;;
        -M)
            RUN_MODE="$2"
            shift 2
            ;;
        *)
            # 兼容旧用法：batch 后直接跟 run_mode
            if [[ "$COMMAND" == "batch" ]]; then
                RUN_MODE="$1"
            fi
            shift
            ;;
    esac
done

# 根据参数执行对应函数
case "$COMMAND" in
    single)
        run_single
        ;;
    batch)
        if [[ "$RUN_MODE" != "eager" && "$RUN_MODE" != "graph" ]]; then
            echo "错误：batch 模式仅支持 eager/graph，当前值: $RUN_MODE"
            show_help
            exit 1
        fi
        if [ -z "$EXCEL_PATH" ]; then
            EXCEL_PATH="$DEFAULT_EXCEL"
        fi
        run_batch "$EXCEL_PATH" "$EXCEL_SHEET" "$PT_PATH" "$RUN_MODE"
        ;;
    batch_exec)
        if [ -z "$EXCEL_PATH" ]; then
            EXCEL_PATH="$DEFAULT_EXCEL"
        fi
        if [[ "$RUN_MODE" != "eager" && "$RUN_MODE" != "graph" ]]; then
            echo "错误：batch_exec 模式仅支持 eager/graph，当前值: $RUN_MODE"
            show_help
            exit 1
        fi
        run_batch_exec "$EXCEL_PATH" "$EXCEL_SHEET" "$PT_PATH" "$RUN_MODE"
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

exit 0