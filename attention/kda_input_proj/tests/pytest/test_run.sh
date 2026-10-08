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

TEST_ACCURACY_SCRIPT="test_kda_input_proj_accuracy.py"

run_single() {
    echo "===== 执行 KdaInputProj 精度 ST ====="
    echo "TEST_DEVICE_ID=${TEST_DEVICE_ID:-auto}"
    if [[ -z "${ASCEND_CUSTOM_OPP_PATH}" && -d "${HOME}/cann_custom_ops/vendors/custom_transformer" ]]; then
        export ASCEND_CUSTOM_OPP_PATH="${HOME}/cann_custom_ops/vendors/custom_transformer"
        export LD_LIBRARY_PATH="${ASCEND_CUSTOM_OPP_PATH}/op_api/lib:${LD_LIBRARY_PATH}"
    fi
    python3 -m pytest -rA -s $TEST_ACCURACY_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
}

show_help() {
    echo "用法: $0 [参数]"
    echo "参数说明："
    echo "  single    执行精度用例"
    echo "  help      显示本帮助信息"
}

if [ $# -ne 1 ]; then
    echo "错误：必须传入且仅传入一个参数（single/help）"
    show_help
    exit 1
fi

case "$1" in
    single)
        run_single
        ;;
    help)
        show_help
        ;;
    *)
        echo "错误：未知参数 '$1'，仅支持 single/help"
        show_help
        exit 1
        ;;
esac

exit 0
