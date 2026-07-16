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

# 显示帮助信息
show_help() {
    cat << EOF
使用方法: $0 {single|save|load|load_graph|consistency} [参数]

脚本选项:
  single         执行单跑功能
    -R           结果保存路径
    示例1: bash $0 single
    示例2: bash $0 single -R "./result/smla_result.xlsx"

  save           执行将excel文件中用例批量转化为PT文件的功能
    -E           excel表地址
    -S           sheet名
    -P           pt文件保存地址
    示例1: bash $0 save
    示例2: bash $0 save -E "./excel/example.xlsx" -S "Sheet1" -P "./data"

  load           执行批量执行PT形式保存的用例的功能
    -P           pt文件读取地址
    -R           结果保存路径
    -E           excel表地址（指定后仅跑表格中涉及的用例）
    -S           sheet名（配合-E使用）
    示例1: bash $0 load
    示例2: bash $0 load -P "./data" -R "./result/smla_result.xlsx"
    示例3: bash $0 load -P "./data" -E "./excel/example.xlsx" -S "CSA"

  load_graph     执行批量执行PT形式保存的用例的功能(Graph aclgraph模式)
    -P           pt文件读取地址
    -R           结果保存路径
    -E           excel表地址（指定后仅跑表格中涉及的用例）
    -S           sheet名（配合-E使用）
    示例1: bash $0 load_graph
    示例2: bash $0 load_graph -P "./data" -R "./result/smla_result.xlsx"
    示例3: bash $0 load_graph -P "./data" -E "./excel/example.xlsx" -S "CSA"

  consistency    执行batch一致性测试（默认关闭，需显式指定）
    -P           pt文件读取地址（默认 ./data）
    -M           一致性模式列表，逗号分隔（默认 1,2；模式3/4暂不支持）
    --seed       随机种子（默认 20260716）
    示例1: bash $0 consistency -P ./data -M 1,2 --seed 20260716
    示例2: bash $0 consistency -P ./data -M 1
    示例3: bash $0 consistency -P ./data -M 3
      （模式3会报错退出并输出unsupported原因）

通用选项:
  -h, --help   显示此帮助信息
EOF
}

# 检查参数数量
if [ $# -lt 1 ]; then
    echo "错误: 需要至少一个参数"
    show_help
    exit 1
fi

# 获取脚本类型
SCRIPT_TYPE=$1
shift  # 移除第一个参数，剩下的都是选项

# 处理 help 请求
if [[ "$SCRIPT_TYPE" == "-h" ]] || [[ "$SCRIPT_TYPE" == "--help" ]]; then
    show_help
    exit 0
fi

# 运行 单跑 脚本的函数
run_script_single() {
    echo "准备运行 单跑 脚本..."

    # 解析 test_sparse_flash_mla_single.py 的参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -R)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: -R 参数需要值"
                    exit 1
                fi
                R_VALUE="$2"
                shift 2
                ;;
            *)
                echo "错误: 未知参数 '$1'"
                echo "test_sparse_flash_mla_single.py 脚本支持的参数: -R"
                exit 1
                ;;
        esac
    done

    # 打印参数信息
    echo "==============================="
    echo "参数配置:"

    if [ -n "$R_VALUE" ]; then
        echo "  结果存储至文件 $R_VALUE"
        export RESULT_SAVE_PATH="$R_VALUE"
    else
        echo "  结果存储至文件 ./result/smla_result.xlsx"
    fi

    echo "==============================="

    # 运行 Python 脚本
    if [ "$VERBOSE" = true ]; then
        echo "正在运行 test_sparse_flash_mla_single.py..."
    fi

    # 检查脚本是否存在
    if [ ! -f "test_sparse_flash_mla_single.py" ]; then
        echo "错误: 找不到 test_sparse_flash_mla_single.py 脚本"
        exit 1
    fi
    # 运行脚本，传递环境变量
    python3 -m pytest -rA -s test_sparse_flash_mla_single.py  -v -m ci
}

# 运行 批跑生成pt文件 脚本的函数
run_script_save() {
    echo "准备运行 sparse_flash_mla_pt_save.py 脚本..."

    # 解析 sparse_flash_mla_pt_save.py 的参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -E)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: -E 参数需要值"
                    exit 1
                fi
                E_VALUE="$2"
                shift 2
                ;;
            -S)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: -S 参数需要值"
                    exit 1
                fi
                S_VALUE="$2"
                shift 2
                ;;
            -P)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: -P 参数需要值"
                    exit 1
                fi
                P_VALUE="$2"
                shift 2
                ;;
            *)
                echo "错误: 未知参数 '$1'"
                echo "sparse_flash_mla_pt_save.py 脚本支持的参数: -E, -S, -P"
                exit 1
                ;;
        esac
    done

    # 打印参数信息
    echo "==============================="
    echo "脚本: sparse_flash_mla_pt_save.py"
    echo "参数配置:"

    if [ -n "$E_VALUE" ]; then
        echo "  输入excel文件路径 $E_VALUE"
        export SMLA_EXCEL_PATH="$E_VALUE"
    else
        echo "  默认输入excel文件路径 excel/example.xlsx"
    fi

    if [ -n "$S_VALUE" ]; then
        echo "  使用sheet名 $S_VALUE"
        export SMLA_EXCEL_SHEET="$S_VALUE"
    else
        echo "  默认使用 Sheet1"
    fi

    if [ -n "$P_VALUE" ]; then
        echo "  PT文件保存地址 $P_VALUE"
        export SMLA_PT_SAVE_PATH="$P_VALUE"
    else
        echo "  默认PT文件保存地址 ./data"
    fi

    echo "==============================="
    # 检查脚本是否存在
    if [ ! -f "batch/sparse_flash_mla_pt_save.py" ]; then
        echo "错误: 找不到 sparse_flash_mla_pt_save.py 脚本"
        exit 1
    fi

    # 运行脚本，传递环境变量
    python3 -m pytest -rA -s batch/sparse_flash_mla_pt_save.py  -v -m ci
}

# 运行 批跑执行pt文件 脚本的函数
run_script_load() {
    echo "准备运行 test_sparse_flash_mla_batch.py 脚本..."

    # 解析 test_sparse_flash_mla_batch.py 的参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -P)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: -P 参数需要值"
                    exit 1
                fi
                P_VALUE="$2"
                shift 2
                ;;
            -R)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: -R 参数需要值"
                    exit 1
                fi
                R_VALUE="$2"
                shift 2
                ;;
            -E)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: -E 参数需要值"
                    exit 1
                fi
                E_VALUE="$2"
                shift 2
                ;;
            -S)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: -S 参数需要值"
                    exit 1
                fi
                S_VALUE="$2"
                shift 2
                ;;
            *)
                echo "错误: 未知参数 '$1'"
                echo "test_sparse_flash_mla_batch.py 脚本支持的参数: -P, -R, -E, -S"
                exit 1
                ;;
        esac
    done

    # 打印参数信息
    echo "==============================="
    echo "脚本: test_sparse_flash_mla_batch.py"
    echo "参数配置:"

    if [ -n "$P_VALUE" ]; then
        echo "  PT文件读取地址 $P_VALUE"
        export SMLA_PT_LOAD_PATH="$P_VALUE"
    else
        echo "  默认PT文件读取地址 ./data"
    fi

    if [ -n "$R_VALUE" ]; then
        echo "  结果存储至文件 $R_VALUE"
        export SMLA_RESULT_SAVE_PATH="$R_VALUE"
    else
        echo "  结果存储至文件 ./result/smla_result.xlsx"
    fi

    if [ -n "$E_VALUE" ]; then
        echo "  Excel文件路径 $E_VALUE"
        export SMLA_EXCEL_PATH="$E_VALUE"
        export SMLA_BATCH_TEST_MODE=1
        if [ -n "$S_VALUE" ]; then
            echo "  使用sheet名 $S_VALUE"
            export SMLA_EXCEL_SHEET="$S_VALUE"
        fi
    else
        echo "  未指定Excel，全量批跑目录下所有.pt文件"
    fi

    echo "==============================="

    # 检查脚本是否存在
    if [ ! -f "test_sparse_flash_mla_batch.py" ]; then
        echo "错误: 找不到 test_sparse_flash_mla_batch.py 脚本"
        exit 1
    fi

    # 获取用例目录
    LOAD_DIR="${SMLA_PT_LOAD_PATH:-./data}"
    if [ ! -d "$LOAD_DIR" ]; then
        echo "错误: 用例目录不存在: $LOAD_DIR"
        exit 1
    fi

    if [ "${SMLA_BATCH_TEST_MODE}" = "1" ] && [ -n "$E_VALUE" ]; then
        SHEET_NAME="${S_VALUE:-CSA}"
        echo "按表格筛选模式: Excel=$E_VALUE, Sheet=$SHEET_NAME"
        TARGET_NAMES=$(python3 -c "
import pandas as pd
df = pd.read_excel('$E_VALUE', sheet_name='$SHEET_NAME')
names = [str(n) for n in df['testcase_name'].dropna().tolist() if str(n) != 'None']
print('\n'.join(names))
")
        if [ -z "$TARGET_NAMES" ]; then
            echo "错误: 表格中没有有效的testcase_name"
            exit 1
        fi
        CASE_FILES=()
        while IFS= read -r target_name; do
            while IFS= read -r matched_file; do
                if [ -n "$matched_file" ] && [[ ! " ${CASE_FILES[*]} " =~ " ${matched_file} " ]]; then
                    CASE_FILES+=("$matched_file")
                fi
            done < <(find "$LOAD_DIR" -maxdepth 1 -name "*.pt" | grep "$target_name" | sort)
        done <<< "$TARGET_NAMES"
        echo "从表格中读取到目标用例名, 筛选后共 ${#CASE_FILES[@]} 个.pt文件"
    else
        mapfile -t CASE_FILES < <(find "$LOAD_DIR" -maxdepth 1 -name "*.pt" | sort)
    fi

    TOTAL=${#CASE_FILES[@]}
    if [ "$TOTAL" -eq 0 ]; then
        echo "错误: 目录 $LOAD_DIR 下未找到匹配的 .pt 用例"
        exit 1
    fi

    echo "共 $TOTAL 条用例待执行, 目录: $LOAD_DIR"
    echo "开始隔离批量执行（每条用例独立 pytest 进程）..."

    PASS=0
    FAIL=0
    FAIL_LIST=()
    SUMMARY_LOG="batch_summary.log"
    FAIL_LOG="batch_fail_list.log"
    : > "$SUMMARY_LOG"
    : > "$FAIL_LOG"

    i=0
    for case_file in "${CASE_FILES[@]}"; do
        i=$((i+1))
        case_name=$(basename "$case_file")
        echo -e "\n===== [$i/$TOTAL] 执行用例: $case_name =====" | tee -a "$SUMMARY_LOG"

        QSAS_TESTCASE_PATH="$case_file" python3 -m pytest -rA -s test_sparse_flash_mla_batch.py -v -m ci 2>&1 | tee -a "$SUMMARY_LOG"
        status=${PIPESTATUS[0]}

        if [ "$status" -eq 0 ]; then
            PASS=$((PASS+1))
            echo "[PASS] $case_name" | tee -a "$SUMMARY_LOG"
        else
            FAIL=$((FAIL+1))
            FAIL_LIST+=("$case_name")
            echo "[FAIL] $case_name" | tee -a "$SUMMARY_LOG"
            echo "$case_name" >> "$FAIL_LOG"
        fi
    done

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
    echo "结果表格:  ${SMLA_RESULT_SAVE_PATH:-./result/smla_result.xlsx}"
}

# 运行 批跑执行pt文件(Graph模式) 脚本的函数
run_script_load_graph() {
    echo "准备运行 test_sparse_flash_mla_batch_graph.py 脚本(Graph模式)..."

    # 解析 test_sparse_flash_mla_batch_graph.py 的参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -P)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: -P 参数需要值"
                    exit 1
                fi
                P_VALUE="$2"
                shift 2
                ;;
            -R)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: -R 参数需要值"
                    exit 1
                fi
                R_VALUE="$2"
                shift 2
                ;;
            -E)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: -E 参数需要值"
                    exit 1
                fi
                E_VALUE="$2"
                shift 2
                ;;
            -S)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: -S 参数需要值"
                    exit 1
                fi
                S_VALUE="$2"
                shift 2
                ;;
            *)
                echo "错误: 未知参数 '$1'"
                echo "test_sparse_flash_mla_batch_graph.py 脚本支持的参数: -P, -R, -E, -S"
                exit 1
                ;;
        esac
    done

    # 打印参数信息
    echo "==============================="
    echo "脚本: test_sparse_flash_mla_batch_graph.py (Graph模式)"
    echo "参数配置:"

    if [ -n "$P_VALUE" ]; then
        echo "  PT文件读取地址 $P_VALUE"
        export SMLA_PT_LOAD_PATH="$P_VALUE"
    else
        echo "  默认PT文件读取地址 ./data"
    fi

    if [ -n "$R_VALUE" ]; then
        echo "  结果存储至文件 $R_VALUE"
        export SMLA_RESULT_SAVE_PATH="$R_VALUE"
    else
        echo "  结果存储至文件 ./result/smla_result.xlsx"
    fi

    if [ -n "$E_VALUE" ]; then
        echo "  Excel文件路径 $E_VALUE"
        export SMLA_EXCEL_PATH="$E_VALUE"
        export SMLA_BATCH_TEST_MODE=1
        if [ -n "$S_VALUE" ]; then
            echo "  使用sheet名 $S_VALUE"
            export SMLA_EXCEL_SHEET="$S_VALUE"
        fi
    else
        echo "  未指定Excel，全量批跑目录下所有.pt文件"
    fi

    echo "==============================="

    # 检查脚本是否存在
    if [ ! -f "test_sparse_flash_mla_batch_graph.py" ]; then
        echo "错误: 找不到 test_sparse_flash_mla_batch_graph.py 脚本"
        exit 1
    fi

    # 获取用例目录
    LOAD_DIR="${SMLA_PT_LOAD_PATH:-./data}"
    if [ ! -d "$LOAD_DIR" ]; then
        echo "错误: 用例目录不存在: $LOAD_DIR"
        exit 1
    fi

    if [ "${SMLA_BATCH_TEST_MODE}" = "1" ] && [ -n "$E_VALUE" ]; then
        SHEET_NAME="${S_VALUE:-CSA}"
        echo "按表格筛选模式: Excel=$E_VALUE, Sheet=$SHEET_NAME"
        TARGET_NAMES=$(python3 -c "
import pandas as pd
df = pd.read_excel('$E_VALUE', sheet_name='$SHEET_NAME')
names = [str(n) for n in df['testcase_name'].dropna().tolist() if str(n) != 'None']
print('\n'.join(names))
")
        if [ -z "$TARGET_NAMES" ]; then
            echo "错误: 表格中没有有效的testcase_name"
            exit 1
        fi
        CASE_FILES=()
        while IFS= read -r target_name; do
            while IFS= read -r matched_file; do
                if [ -n "$matched_file" ] && [[ ! " ${CASE_FILES[*]} " =~ " ${matched_file} " ]]; then
                    CASE_FILES+=("$matched_file")
                fi
            done < <(find "$LOAD_DIR" -maxdepth 1 -name "*.pt" | grep "$target_name" | sort)
        done <<< "$TARGET_NAMES"
        echo "从表格中读取到目标用例名, 筛选后共 ${#CASE_FILES[@]} 个.pt文件"
    else
        mapfile -t CASE_FILES < <(find "$LOAD_DIR" -maxdepth 1 -name "*.pt" | sort)
    fi

    TOTAL=${#CASE_FILES[@]}
    if [ "$TOTAL" -eq 0 ]; then
        echo "错误: 目录 $LOAD_DIR 下未找到匹配的 .pt 用例"
        exit 1
    fi

    echo "共 $TOTAL 条用例待执行, 目录: $LOAD_DIR"
    echo "开始隔离批量执行(Graph模式, 每条用例独立 pytest 进程)..."

    PASS=0
    FAIL=0
    FAIL_LIST=()
    SUMMARY_LOG="batch_graph_summary.log"
    FAIL_LOG="batch_graph_fail_list.log"
    : > "$SUMMARY_LOG"
    : > "$FAIL_LOG"

    i=0
    for case_file in "${CASE_FILES[@]}"; do
        i=$((i+1))
        case_name=$(basename "$case_file")
        echo -e "\n===== [$i/$TOTAL] 执行用例(Graph模式): $case_name =====" | tee -a "$SUMMARY_LOG"

        QSAS_TESTCASE_PATH="$case_file" python3 -m pytest -rA -s test_sparse_flash_mla_batch_graph.py -v -m graph 2>&1 | tee -a "$SUMMARY_LOG"
        status=${PIPESTATUS[0]}

        if [ "$status" -eq 0 ]; then
            PASS=$((PASS+1))
            echo "[PASS] $case_name" | tee -a "$SUMMARY_LOG"
        else
            FAIL=$((FAIL+1))
            FAIL_LIST+=("$case_name")
            echo "[FAIL] $case_name" | tee -a "$SUMMARY_LOG"
            echo "$case_name" >> "$FAIL_LOG"
        fi
    done

    echo -e "\n========== 批量执行汇总(Graph模式) ==========" | tee -a "$SUMMARY_LOG"
    echo "总计: $TOTAL  通过: $PASS  失败: $FAIL" | tee -a "$SUMMARY_LOG"
    if [ "$FAIL" -gt 0 ]; then
        echo "失败用例:" | tee -a "$SUMMARY_LOG"
        for f in "${FAIL_LIST[@]}"; do
            echo "  - $f" | tee -a "$SUMMARY_LOG"
        done
    fi
    echo "详细日志:  $SUMMARY_LOG"
    echo "失败清单:  $FAIL_LOG"
    echo "结果表格:  ${SMLA_RESULT_SAVE_PATH:-./result/smla_result.xlsx}"
}

# 运行 batch一致性测试 脚本的函数
run_script_consistency() {
    echo "准备运行 test_sparse_flash_mla_batch_consistency.py (batch一致性测试)..."

    # 默认值
    CONSISTENCY_PT_DIR="./data"
    CONSISTENCY_MODES="1,2"
    CONSISTENCY_SEED="20260716"

    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -P)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: -P 参数需要值"
                    exit 1
                fi
                CONSISTENCY_PT_DIR="$2"
                shift 2
                ;;
            -M)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: -M 参数需要值"
                    exit 1
                fi
                CONSISTENCY_MODES="$2"
                shift 2
                ;;
            --seed)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: --seed 参数需要值"
                    exit 1
                fi
                CONSISTENCY_SEED="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                return 0
                ;;
            *)
                echo "错误: 未知参数 '$1'"
                echo "consistency 脚本支持的参数: -P, -M, --seed"
                exit 1
                ;;
        esac
    done

    # 打印参数信息
    echo "==============================="
    echo "脚本: test_sparse_flash_mla_batch_consistency.py"
    echo "参数配置:"
    echo "  PT文件读取地址 $CONSISTENCY_PT_DIR"
    echo "  一致性模式 $CONSISTENCY_MODES"
    echo "  随机种子 $CONSISTENCY_SEED"
    echo "==============================="

    # 检查脚本是否存在
    if [ ! -f "test_sparse_flash_mla_batch_consistency.py" ]; then
        echo "错误: 找不到 test_sparse_flash_mla_batch_consistency.py 脚本"
        exit 1
    fi

    if ! [[ "$CONSISTENCY_SEED" =~ ^[0-9]+$ ]]; then
        echo "错误: --seed 必须是非负整数，当前值: $CONSISTENCY_SEED"
        exit 1
    fi

    # 只接受本阶段实际实现的模式，避免 pytest 全部 skip 后被误报为通过。
    IFS=',' read -r -a requested_modes <<< "$CONSISTENCY_MODES"
    if [ "${#requested_modes[@]}" -eq 0 ]; then
        echo "错误: -M 至少需要指定模式1或2"
        exit 1
    fi
    for m in "${requested_modes[@]}"; do
        case "$m" in
            1|2)
                ;;
            3)
                echo "错误: 模式3 (batch内token拆分) 暂不支持。"
                echo "原因: 需要CPU golden等价性预检查和query-position/mask语义验证。参见开发计划§10。"
                exit 1
                ;;
            4)
                echo "错误: 模式4 (前插/删除token) 暂不支持。"
                echo "原因: 改变query position、causal/window mask、TND prefix sums和稀疏索引。"
                echo "需要mask语义预检查和stateful prefill/decode测试设计。参见开发计划§11。"
                exit 1
                ;;
            *)
                echo "错误: 不支持的一致性模式 '$m'，当前仅支持1和2"
                exit 1
                ;;
        esac
    done

    # 检查用例目录
    if [ ! -d "$CONSISTENCY_PT_DIR" ]; then
        echo "错误: 用例目录不存在: $CONSISTENCY_PT_DIR"
        exit 1
    fi

    # 设置环境变量
    export SMLA_CONSISTENCY_TEST=1
    export SMLA_CONSISTENCY_MODES="$CONSISTENCY_MODES"
    export SMLA_CONSISTENCY_SEED="$CONSISTENCY_SEED"
    export SMLA_PT_LOAD_PATH="$CONSISTENCY_PT_DIR"

    # 获取B>=2的.pt文件列表
    CASE_FILES=()
    for pt_file in "$CONSISTENCY_PT_DIR"/*.pt; do
        [ -f "$pt_file" ] || continue
        B_VAL=$(python3 - "$pt_file" 2>/dev/null <<'PY'
import torch
import sys
try:
    d = torch.load(sys.argv[1], map_location='cpu')
    print(int(d.get('B', 0)))
except Exception:
    print(0)
PY
)
        if [ -n "$B_VAL" ] && [ "$B_VAL" -ge 2 ]; then
            CASE_FILES+=("$pt_file")
        fi
    done

    TOTAL=${#CASE_FILES[@]}
    if [ "$TOTAL" -eq 0 ]; then
        echo "错误: 目录 $CONSISTENCY_PT_DIR 下未找到 B>=2 的 .pt 用例"
        echo "提示: 先运行 'bash test_run.sh save' 生成多batch的.pt文件"
        exit 1
    fi

    echo "共 $TOTAL 条B>=2用例待执行, 目录: $CONSISTENCY_PT_DIR"
    echo "开始隔离批量执行（每条用例独立 pytest 进程）..."

    PASS=0
    FAIL=0
    UNSUPPORTED=0
    FAIL_LIST=()
    SUMMARY_LOG="consistency_summary.log"
    FAIL_LOG="consistency_fail_list.log"
    : > "$SUMMARY_LOG"
    : > "$FAIL_LOG"

    i=0
    for case_file in "${CASE_FILES[@]}"; do
        i=$((i+1))
        case_name=$(basename "$case_file")
        echo -e "\n===== [$i/$TOTAL] 执行一致性测试: $case_name =====" | tee -a "$SUMMARY_LOG"

        CASE_LOG=$(mktemp "${TMPDIR:-/tmp}/smla-consistency-case.XXXXXX")
        QSAS_TESTCASE_PATH="$case_file" \
            python3 -m pytest -rA -s test_sparse_flash_mla_batch_consistency.py \
            -v -m consistency 2>&1 | tee -a "$SUMMARY_LOG" | tee "$CASE_LOG"
        status=${PIPESTATUS[0]}

        if [ "$status" -eq 0 ] && grep -q "Batch-consistency NPU validation is unavailable" "$CASE_LOG"; then
            UNSUPPORTED=$((UNSUPPORTED+1))
            echo "[UNSUPPORTED] $case_name" | tee -a "$SUMMARY_LOG"
        elif [ "$status" -eq 0 ]; then
            PASS=$((PASS+1))
            echo "[PASS] $case_name" | tee -a "$SUMMARY_LOG"
        else
            FAIL=$((FAIL+1))
            FAIL_LIST+=("$case_name")
            echo "[FAIL] $case_name" | tee -a "$SUMMARY_LOG"
            echo "$case_name" >> "$FAIL_LOG"
        fi
        rm -f "$CASE_LOG"
    done

    echo -e "\n========== batch一致性测试汇总 ==========" | tee -a "$SUMMARY_LOG"
    echo "总计: $TOTAL  通过: $PASS  环境不支持: $UNSUPPORTED  失败: $FAIL" | tee -a "$SUMMARY_LOG"
    if [ "$FAIL" -gt 0 ]; then
        echo "失败用例:" | tee -a "$SUMMARY_LOG"
        for f in "${FAIL_LIST[@]}"; do
            echo "  - $f" | tee -a "$SUMMARY_LOG"
        done
    fi
    echo "详细日志:  $SUMMARY_LOG"
    echo "失败清单:  $FAIL_LOG"
    if [ "$FAIL" -gt 0 ]; then
        return 1
    fi
}

# 根据脚本类型调用相应的函数
case "$SCRIPT_TYPE" in
    single)
        run_script_single "$@"
        ;;
    save)
        run_script_save "$@"
        ;;
    load)
        run_script_load "$@"
        ;;
    load_graph)
        run_script_load_graph "$@"
        ;;
    consistency)
        run_script_consistency "$@" || exit $?
        ;;
    *)
        echo "错误: 未知的脚本类型 '$SCRIPT_TYPE'"
        echo "可用类型: single, save, load, load_graph, consistency"
        show_help
        exit 1
        ;;
esac

echo "==============================="
echo "所有任务完成"
