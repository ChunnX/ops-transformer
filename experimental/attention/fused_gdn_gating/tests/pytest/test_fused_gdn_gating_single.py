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

import pytest
import torch_npu

import custom_ops  # noqa: E402, F401
from fused_gdn_gating_paramset import ENABLED_PARAMS  # noqa: E402
import fused_gdn_gating_golden  # noqa: E402

_OP_REGISTERED = custom_ops.has_npu_fused_gdn_gating_op()
_OP_SKIP_REASON = (
    "torch.ops.custom.npu_fused_gdn_gating is not registered. "
    "Install the CANN custom operator package "
    "(`bash build.sh --pkg --experimental --ops=fused_gdn_gating --soc=<soc> "
    "&& bash output/custom_opp_*.run`), or set FUSED_GDN_GATING_CUSTOM_OPS_PATH "
    "to a directory containing custom_ops_lib*.so."
)


def case_id(case):
    return f"{case['case_id']}:{case['testcase_name']}"


def _is_310p():
    try:
        soc_name = torch_npu.npu.get_device_name(0)
        return "310P" in soc_name or "310p" in soc_name
    except Exception:
        return False


@pytest.mark.ci
@pytest.mark.parametrize("case", ENABLED_PARAMS, ids=case_id)
def test_fused_gdn_gating(case):
    if not _OP_REGISTERED:
        pytest.skip(_OP_SKIP_REASON)
    is_310p = _is_310p()
    if case["soc"] == "910b" and is_310p:
        pytest.skip("BF16 not supported on 310P")
    if case["soc"] == "310p" and not is_310p:
        pytest.skip("310P specific test")

    fused_gdn_gating_golden.run_fused_gdn_gating_eager(case)
