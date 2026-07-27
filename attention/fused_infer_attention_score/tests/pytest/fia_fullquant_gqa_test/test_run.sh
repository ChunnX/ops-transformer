#!/bin/bash

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
QBSA_BATCH_DIR="/home/cann/Desktop/FA/ops-transformer-FIA-rdvcase/experimental/attention/quant_block_sparse_attn/tests/pytest/batch"
GQA_ONLY_NPU_SCRIPT="fia_fullquant_gqa_only_npu.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"

show_help() {
    echo "用法: $0 batch_single <csv文件> <i>"
    echo
    echo "参数说明："
    echo "  csv文件  GQA性能CSV文件名、相对路径或绝对路径"
    echo "  i        CSV数据行号，从1开始，不包含表头"
    echo
    echo "示例："
    echo "  $0 batch_single cases_fia_fullquant_gqa_perf_rdv.csv 1"
}

resolve_csv_path() {
    local csv_arg="$1"

    if [[ "$csv_arg" = /* && -f "$csv_arg" ]]; then
        realpath "$csv_arg"
        return 0
    fi
    if [[ -f "$SCRIPT_DIR/$csv_arg" ]]; then
        realpath "$SCRIPT_DIR/$csv_arg"
        return 0
    fi
    if [[ -f "$QBSA_BATCH_DIR/$csv_arg" ]]; then
        realpath "$QBSA_BATCH_DIR/$csv_arg"
        return 0
    fi

    echo "错误：找不到CSV文件 '$csv_arg'" >&2
    return 1
}

run_batch_single() {
    if [[ $# -ne 2 ]]; then
        show_help
        exit 1
    fi

    local csv_path case_index
    csv_path=$(resolve_csv_path "$1") || exit 1
    case_index="$2"
    if ! [[ "$case_index" =~ ^[1-9][0-9]*$ ]]; then
        echo "错误：i必须是正整数，当前为 '$case_index'" >&2
        exit 1
    fi

    echo "===== FIA FullQuant GQA batch_single ====="
    echo "CSV文件: $csv_path"
    echo "用例行号: $case_index"
    echo "执行脚本: $SCRIPT_DIR/$GQA_ONLY_NPU_SCRIPT"
    echo "性能模式: only NPU，不生成CPU golden，不做result_compare"

    (
        cd "$SCRIPT_DIR" || exit 1
        "$PYTHON_BIN" "$GQA_ONLY_NPU_SCRIPT" \
            --csv "$csv_path" \
            --case-index "$case_index" \
            --run-times "${FIA_GQA_RUN_TIMES:-1}"
    )
}

if [[ $# -lt 1 ]]; then
    show_help
    exit 1
fi

case "$1" in
    batch_single)
        shift
        run_batch_single "$@"
        ;;
    help|-h|--help)
        show_help
        ;;
    *)
        echo "错误：未知参数 '$1'" >&2
        show_help
        exit 1
        ;;
esac
