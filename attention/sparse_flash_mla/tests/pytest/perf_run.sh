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

set -o pipefail

TEST_SCRIPT="${SMLA_PERF_TEST_SCRIPT:-test_sparse_flash_mla_batch.py}"
RESULT_PATH="${SMLA_RESULT_SAVE_PATH:-./result/smla_result.xlsx}"
export SMLA_RESULT_SAVE_PATH="$RESULT_PATH"

# Do not remove historical result or PROF files: process_perf_data.py merges every
# retained profile run, so repeated executions remain in the same performance table.
msprof python3 -m pytest -rA -s "$TEST_SCRIPT" -v -m ci \
    -W ignore::UserWarning -W ignore::DeprecationWarning "$@" 2>&1 | tee res.log
pytest_status=${PIPESTATUS[0]}

python3 process_perf_data.py --test_result_path "$RESULT_PATH"
perf_status=$?

if [ "$pytest_status" -ne 0 ]; then
    exit "$pytest_status"
fi
exit "$perf_status"
