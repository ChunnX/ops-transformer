#!/usr/bin/python3
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

import gc

import pytest
import torch
import torch_npu  # noqa: F401  确保 npu 后端注册

from common import quant_flash_attn_golden as golden


def execute_test(params, golden_mode, cache_dir=None):
    gc.collect()
    torch.npu.synchronize()
    torch.npu.empty_cache()

    atten_percent, atten_result, _, _ = golden.execute_full(params)
    return atten_percent, atten_result


def check_results(atten_percent, atten_result):
    if atten_result is None:
        return
    if atten_result != "Pass":
        pct = atten_percent if atten_percent is not None else "N/A"
        pytest.fail(
            f"Attention output compare failed: result={atten_result}, PctRlt={pct}"
        )
