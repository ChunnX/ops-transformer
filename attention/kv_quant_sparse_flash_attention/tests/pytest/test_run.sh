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

# 从 Excel 批量生成 pt 文件
run_batch_save() {
    echo "===== 批量生成 pt 文件 ====="
    python3 -m pytest -rA -s "$PT_SAVE_SCRIPT" -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
    if [ $? -ne 0 ]; then
        echo "生成 pt 文件失败，退出"
        exit 1
    fi
    echo "===== pt 文件生成完成 ====="
}

# 从 pt 文件批量执行 NPU 测试
run_batch_exec() {
    echo "===== 从 pt 文件批量执行 NPU 测试 ====="
    python3 -m pytest -rA -s "$TEST_BATCH_SCRIPT" -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
    if [ $? -ne 0 ]; then
        echo "批量执行失败"
        exit 1
    fi
    echo "===== 批量执行完成 ====="
}

show_help() {
    echo "用法: $0 [参数]"
    echo "参数说明："
    echo "  single        执行单算子用例调测（含 CPU golden + NPU + 精度对比）"
    echo "  batch_save    从 Excel 批量生成 pt 文件"
    echo "  batch_exec    从 pt 文件批量执行 NPU 测试"
    echo "  help          显示本帮助信息"
    echo "示例："
    echo "  $0 single       # 执行 single 模式"
    echo "  $0 batch_save   # 生成 pt 文件"
    echo "  $0 batch_exec   # 执行 NPU 测试"
}

if [ $# -ne 1 ]; then
    show_help
    exit 1
fi

case "$1" in
    single)
        run_single
        ;;
    batch_save)
        run_batch_save
        ;;
    batch_exec)
        run_batch_exec
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