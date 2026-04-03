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

PT_SAVE_SCRIPT="./batch/test_kv_quant_sparse_flash_attention_pt_save.py"
TEST_BATCH_SCRIPT="test_kv_quant_sparse_flash_attention_batch.py"
TEST_SINGLE_SCRIPT="test_kv_quant_sparse_flash_attention_single.py"
TEST_GRAPH_SCRIPT="test_kv_quant_sparse_flash_attention_batch_graph.py"

run_single() {
    python3 -m pytest -rA -s "$TEST_SINGLE_SCRIPT" -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
}

run_batch() {
    python3 -m pytest -rA -s "$PT_SAVE_SCRIPT" -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning || exit 1
    python3 -m pytest -rA -s "$TEST_BATCH_SCRIPT" -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning || exit 1
}

run_graph() {
    python3 -m pytest -rA -s "$TEST_GRAPH_SCRIPT" -v -m graph -W ignore::UserWarning -W ignore::DeprecationWarning
}

show_help() {
    echo "Usage: $0 [single|batch|graph|help]"
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
    graph)
        run_graph
        ;;
    help)
        show_help
        ;;
    *)
        show_help
        exit 1
        ;;
esac
