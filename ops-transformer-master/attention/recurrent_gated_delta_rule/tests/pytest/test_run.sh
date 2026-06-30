#!/bin/bash

# 脚本路径
TEST_RECURRENT_GATED_DELTA_RULE_SINGLE_SCRIPT="test_recurrent_gated_delta_rule_single.py"

# ====================== 执行区======================

# 算子调测
run_single() {
    echo "===== 执行单算子用例调测 ====="
    TEST_MODE=single python3 -m pytest -rA -s $TEST_RECURRENT_GATED_DELTA_RULE_SINGLE_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
}

# RDV测试
run_rdv() {
    echo "===== 执行RDV参数集测试 ====="
    TEST_MODE=rdv python3 -m pytest -rA -s $TEST_RECURRENT_GATED_DELTA_RULE_SINGLE_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
}

# 显示帮助信息
show_help() {
    echo "用法: $0 [参数]"
    echo "参数说明："
    echo "  single    执行单算子用例调测"
    echo "  rdv       执行RDV参数集测试"
    echo "  help      显示本帮助信息"
    echo "示例："
    echo "  $0 single  # 执行single模式"
    echo "  $0 rdv     # 执行rdv模式"
}

# ====================== 主逻辑 ======================
# 检查传入的参数数量
if [ $# -ne 1 ]; then
    echo "错误：必须传入且仅传入一个参数（single/batch/help）"
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
    help)
        show_help
        ;;
    *)
        echo "错误：未知参数 '$1'，仅支持 single/rdv/help"
        show_help
        exit 1
        ;;
esac

exit 0