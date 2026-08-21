#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Opt-in MQSMLA deterministic-level-3 batch-consistency tests."""

import os
import sys
from pathlib import Path

import pytest
import torch_npu

COMMON_PYTEST_PATH = Path(__file__).resolve().parents[3] / "common/tests/pytest"
if str(COMMON_PYTEST_PATH) not in sys.path:
    sys.path.insert(0, str(COMMON_PYTEST_PATH))

import result_compare_method  # noqa: E402
from batch import mixed_quant_sparse_flash_mla_process  # noqa: E402
from batch_consistency.model import RunResult  # noqa: E402
from batch_consistency.pytest_support import collect_case_matrix, run_pytest_case  # noqa: E402


class MixedQuantSparseFlashMlaExecutor:
    """Normalize the ordinary MQSMLA eager pytest tuple for the shared runner."""

    def __call__(self, data):
        device_id = int(os.environ.get("MQSMLA_DEVICE_ID", "0"))
        output, _cpu_output, _cpu_lse, softmax_lse = (
            mixed_quant_sparse_flash_mla_process.test_mqsmla_quant_process_ci(
                data, device_id=device_id
            )
        )
        return RunResult(output, softmax_lse)


TEST_MATRIX = collect_case_matrix("MQSMLA", "mqsmla_testcase")

pytestmark = pytest.mark.consistency


@pytest.mark.parametrize("case_path,mode", TEST_MATRIX)
def test_mixed_quant_sparse_flash_mla_batch_consistency(case_path, mode):
    device_id = int(os.environ.get("MQSMLA_DEVICE_ID", "0"))
    run_pytest_case(
        case_path,
        mode,
        "MQSMLA",
        "mqsmla",
        MixedQuantSparseFlashMlaExecutor(),
        result_compare_method.check_result,
        lambda: torch_npu.npu.set_device(device_id),
    )
