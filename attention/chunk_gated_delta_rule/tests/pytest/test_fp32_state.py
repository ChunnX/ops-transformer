# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import torch_npu
import pytest

from chunk_gated_delta_rule_golden import run_chunk_gated_delta_rule_eager

DEVICE_ID = 0

FP32_STATE_CASES = [
    {"B": 1, "seqlen": 64,    "nk": 4, "nv": 4, "dk": 128, "dv": 128, "chunk_size": 64,
     "data_type": torch.bfloat16, "state_dtype": torch.float32, "desc": "short_seq"},
    {"B": 1, "seqlen": 16384, "nk": 4, "nv": 4, "dk": 128, "dv": 128, "chunk_size": 64,
     "data_type": torch.bfloat16, "state_dtype": torch.float32, "desc": "long_seq"},
    {"B": 2, "seqlen": 128,   "nk": 4, "nv": 4, "dk": 128, "dv": 128, "chunk_size": 64,
     "data_type": torch.bfloat16, "state_dtype": torch.float32, "desc": "multibatch"},
    {"B": 1, "seqlen": 100,   "nk": 4, "nv": 4, "dk": 128, "dv": 128, "chunk_size": 64,
     "data_type": torch.bfloat16, "state_dtype": torch.float32, "desc": "tail_chunk"},
]


@pytest.mark.parametrize("case", FP32_STATE_CASES, ids=[c["desc"] for c in FP32_STATE_CASES])
def test_fp32_state(case):
    torch_npu.npu.set_device(DEVICE_ID)
    run_chunk_gated_delta_rule_eager(
        case["B"], case["seqlen"], case["nk"], case["nv"],
        case["dk"], case["dv"], chunk_size=case["chunk_size"],
        data_type=case["data_type"], state_dtype=case["state_dtype"],
    )
