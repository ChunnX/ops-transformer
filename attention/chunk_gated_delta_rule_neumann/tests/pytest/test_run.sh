#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# 脚本路径
TEST_CHUNK_GATED_DELTA_RULE_NEUMANN_SINGLE_SCRIPT="test_chunk_gated_delta_rule_neumann_single.py"
TEST_CHUNK_GATED_DELTA_RULE_THREE_WAY_SCRIPT="compare_chunk_gdr_three_way.py"
TEST_CHUNK_GATED_DELTA_RULE_PERFORMANCE_SCRIPT="run_chunk_gdr_performance_msprof.py"

# ====================== 执行区======================

# 算子调测
run_single() {
    echo "===== 执行用例算子调测 ====="
    python3 -m pytest -rA -s $TEST_CHUNK_GATED_DELTA_RULE_NEUMANN_SINGLE_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
}

run_compare() {
    echo "===== 执行三方精度对比 ====="
    python3 -m pytest -rA -s $TEST_CHUNK_GATED_DELTA_RULE_THREE_WAY_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
}

run_performance() {
    echo "===== 执行双算子 msprof 性能对比 ====="
    python3 $TEST_CHUNK_GATED_DELTA_RULE_PERFORMANCE_SCRIPT
}

# 显示帮助信息
show_help() {
    echo "用法: $0 [参数]"
    echo "参数说明："
    echo "  single       执行单算子用例调测"
    echo "  compare      执行CPU golden、ChunkGDR、ChunkGDRNeumann三方精度对比"
    echo "  performance  执行ChunkGDR、ChunkGDRNeumann双算子msprof性能对比"
    echo "  help         显示本帮助信息"
    echo "示例："
    echo "  $0 single       # 执行single模式"
    echo "  $0 compare      # 执行三方精度对比"
    echo "  $0 performance  # 执行msprof性能对比"
}

# ====================== 主逻辑 ======================
# 检查传入的参数数量
if [ $# -ne 1 ]; then
    echo "错误：必须传入且仅传入一个参数（single/compare/performance/help）"
    show_help
    exit 1
fi

# 根据参数执行对应函数
case "$1" in
    single)
        run_single
        exit $?
        ;;
    compare)
        run_compare
        exit $?
        ;;
    performance)
        run_performance
        exit $?
        ;;
    help)
        show_help
        exit 0
        ;;
    *)
        echo "错误：未知参数 '$1'，仅支持 single/compare/performance/help"
        show_help
        exit 1
        ;;
esac
