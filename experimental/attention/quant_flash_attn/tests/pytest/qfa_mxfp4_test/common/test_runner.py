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
"""共享测试执行逻辑"""

import pytest

from common import quant_flash_attn_golden as golden


def execute_test(params, golden_mode, cache_dir=None):
    """执行单个测试 case。

    mxfp4 的 run_fia_eager 为一体化函数，golden_mode 在此语义为：
    - all: 全流程执行并校验精度（默认）
    - 其他模式（gen/cpu/npu/compare）：由于 golden 内部强耦合无法纯拆分，
      统一退化为全流程执行，保证结果一致性。cache_dir 暂不强制使用，
      保留接口供未来 golden 拆分后启用。
    """
    atten_percent, atten_result, _, _ = golden.execute_full(params)
    return atten_percent, atten_result


def check_results(atten_percent, atten_result):
    """校验结果，失败则 pytest.fail"""
    if atten_result is None:
        return
    if atten_result != "Pass":
        pct = atten_percent if atten_percent is not None else "N/A"
        pytest.fail(
            f"Attention output compare failed: result={atten_result}, PctRlt={pct}"
        )
