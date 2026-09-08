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

# 使用单用例脚本生成测试数据并采集性能。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

export MQSMLA_RESULT_SAVE_PATH="${MQSMLA_RESULT_SAVE_PATH:-result.xlsx}"
# 性能数据处理要求每个用例只执行一组 metadata / attention 算子。
export MQSMLA_BATCH_CONSISTENCY=off
# 生成、保存输入后执行NPU，不生成CPU golden或进行精度对比。
export MQSMLA_INPUTS_ONLY=0

rm -f -- "$MQSMLA_RESULT_SAVE_PATH"
test_status=0
msprof python3 -m pytest -rA -s test_mixed_quant_sparse_flash_mla_single.py \
    -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning \
    2>&1 | tee res.log || test_status=$?

# NPU执行失败时仍尝试生成性能报告，同时保留测试失败退出码。
report_args=(--test_result_path="$MQSMLA_RESULT_SAVE_PATH" --delete_perf_file True)
if [ -f perf_golden.xlsx ]; then
    report_args+=(--is_compare True)
else
    echo "perf_golden.xlsx不存在，仅生成性能报告，不进行基线对比。"
fi
python3 process_perf_data.py "${report_args[@]}"
exit "$test_status"
