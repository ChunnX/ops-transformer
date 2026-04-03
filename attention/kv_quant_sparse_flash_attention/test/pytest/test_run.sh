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

# 脚本路径
PT_SAVE_SCRIPT="./batch/test_kv_quant_sparse_flash_attention_pt_save.py"
TEST_BATCH_SCRIPT="test_kv_quant_sparse_flash_attention_batch.py"
TEST_SINGLE_SCRIPT="test_kv_quant_sparse_flash_attention_single.py"
PT_SAVE_PATH="./pt_files/"

mkdir -p $PT_SAVE_PATH

# ====================== 执行区======================

# 单用例算子调测
run_single() {
    echo "===== 执行单用例算子调测 ====="
    python3 -m pytest -rA -s "$TEST_SINGLE_SCRIPT" -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
}

# 用例批量生成调试
run_batch() {
    echo "===== 执行用例批量生成调试 ====="

    echo -e "\n===== 第一步：执行test_kv_quant_sparse_flash_attention_pt_save.py ====="
    python3 -m pytest -rA -s "$PT_SAVE_SCRIPT" -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
    if [ $? -ne 0 ]; then
        echo "test_kv_quant_sparse_flash_attention_pt_save.py 执行失败，退出"
        exit 1
    fi

    echo -e "\n===== 第二步：执行pytest命令 ====="
    python3 -m pytest -rA -s "$TEST_BATCH_SCRIPT" -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
    if [ $? -ne 0 ]; then
        echo "pytest执行失败"
        exit 1
    fi

    echo -e "\n=====执行完成！====="
}

show_help() {
    echo "用法: $0 [参数]"
    echo "参数说明："
    echo "  single      执行单算子用例调测（含 CPU golden + NPU + 精度对比）"
    echo "  batch       执行用例批量生成调试"
    echo "  help        显示本帮助信息"
    echo "示例："
    echo "  $0 single    # 执行single模式"
    echo "  $0 batch     # 执行batch模式"
}

if [ $# -ne 1 ]; then
    show_help
    exit 1
fi

case "$1" in
    single)
        run_single
        ;;
    batch)
        run_batch
        ;;
    help)
        show_help
        ;;
    *)
        show_help
        exit 1
        ;;
esac

exit 0