#!/bin/bash

# 脚本路径
BSA_PT_SAVE_SCRIPT="./batch/test_quant_block_sparse_attn_pt_save.py"
TEST_BSA_PT_BATCH_SCRIPT="test_quant_block_sparse_attn_batch.py"
TEST_BSA_SINGLE_SCRIPT="test_quant_block_sparse_attn_single.py"

# ====================== 执行区======================

# 单用例算子调测
run_single() {
    echo "===== 执行单用例算子调测 ====="
    python3 -m pytest -rA -s $TEST_BSA_SINGLE_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
}

# 用例批量生成调试
run_batch() {
    echo "===== 执行用例批量生成测试 ====="

    echo -e "\n===== 第一步：执行test_quant_block_sparse_attn_pt_save.py ====="
    python3 -m pytest -rA -s $BSA_PT_SAVE_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
    if [ $? -ne 0 ]; then
        echo "test_quant_block_sparse_attn_pt_save.py 执行失败，退出"
        exit 1
    fi

    echo -e "\n===== 第二步：执行pytest命令 ====="
    python3 -m pytest -rA -s $TEST_BSA_PT_BATCH_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
    if [ $? -ne 0 ]; then
        echo "pytest执行失败"
        exit 1
    fi

    echo -e "\n=====执行完成！====="
}

# 指定CSV的单条用例批量流程，i为CSV数据行号（从1开始，不含表头）
run_batch_single() {
    if [ $# -lt 2 ]; then
        echo "错误：batch_single 至少需要传入一个CSV文件和用例行号"
        show_help
        exit 1
    fi

    local case_index="${@: -1}"
    local csv_count=$(($# - 1))
    local csv_files=("${@:1:${csv_count}}")

    if ! [[ "$case_index" =~ ^[1-9][0-9]*$ ]]; then
        echo "错误：用例行号必须是从1开始的正整数，当前为 '$case_index'"
        exit 1
    fi

    local joined_csv
    joined_csv=$(IFS=,; echo "${csv_files[*]}")
    local csv_key="${joined_csv//\//_}"
    csv_key="${csv_key//,/_}"
    csv_key="${csv_key//\*/all}"
    local testcase_dir="bsa_testcase/batch_single_${csv_key}_${case_index}_$$"

    export QBSA_BATCH_CSV="$joined_csv"
    export QBSA_BATCH_CASE_INDEX="$case_index"
    export BSA_TESTCASE_PATH="$testcase_dir"
    export BSA_RESULT_PATH="result_batch_single_${case_index}.xlsx"
    export BSA_SKIP_CPU_GOLDEN=1
    export BSA_SKIP_RESULT_COMPARE=1
    export BSA_PRINT_COMPUTE_LOG=1

    echo "===== 执行指定CSV单条用例 ====="
    echo "CSV文件: $joined_csv"
    echo "用例行号: $case_index"
    echo "用例目录: $BSA_TESTCASE_PATH"
    echo "性能模式: 跳过CPU golden生成和结果比对"

    echo -e "\n===== 第一步：生成指定用例PT ====="
    python3 -m pytest -rA -s $BSA_PT_SAVE_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
    if [ $? -ne 0 ]; then
        echo "test_quant_block_sparse_attn_pt_save.py 执行失败，退出"
        exit 1
    fi

    echo -e "\n===== 第二步：执行指定用例 ====="
    python3 -m pytest -rA -s $TEST_BSA_PT_BATCH_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
    if [ $? -ne 0 ]; then
        echo "pytest执行失败"
        exit 1
    fi

    echo -e "\n=====执行完成！====="
}

# 显示帮助信息
show_help() {
    echo "用法: $0 [参数]"
    echo "参数说明："
    echo "  single    执行单算子用例调测"
    echo "  batch     执行用例批量生成调试"
    echo "  batch_single <csv...> <i>  执行指定CSV中第i条用例（i从1开始，不含表头）"
    echo "  help      显示本帮助信息"
    echo "示例："
    echo "  $0 single  # 执行single模式"
    echo "  $0 batch   # 执行batch模式"
    echo "  $0 batch_single cases_fia_fullquant_mxfp8_perf_rdv.csv 1"
    echo "  $0 batch_single cases_fia_fullquant_mxfp8_perf_rdv.csv cases_fia_fullquant_gqa_perf_rdv.csv 1"
}

# ====================== 主逻辑 ======================
# 检查传入的参数数量
if [ $# -lt 1 ]; then
    echo "错误：必须传入参数（single/batch/batch_single/help）"
    show_help
    exit 1
fi

# 根据参数执行对应函数
case "$1" in
    single)
        run_single
        ;;
    batch)
        if [ $# -ne 1 ]; then
            echo "错误：batch 不支持额外参数"
            show_help
            exit 1
        fi
        run_batch
        ;;
    batch_single)
        shift
        run_batch_single "$@"
        ;;
    help)
        show_help
        ;;
    *)
        echo "错误：未知参数 '$1'，仅支持 single/batch/batch_single/help"
        show_help
        exit 1
        ;;
esac

exit 0
