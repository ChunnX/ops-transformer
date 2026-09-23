# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pytest
import torch

from test_compressor_arch22 import _run_ring_sequence


@pytest.mark.ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [128, 512])
@pytest.mark.parametrize(
    "layout_th,length",
    [
        (False, 1),
        (False, 1003),
        (False, 8192),
        (False, 10240),
        (True, 1),
        (True, 1003),
        (True, 8192),
        (True, 10240),
    ],
)
def test_arch22_h5120(dtype, layout_th, head_dim, length):
    capacity = 1024
    _run_ring_sequence(
        dtype,
        layout_th,
        [length, 1, 1, 1, 1],
        head_dim=head_dim,
        hidden=5120,
        capacity=capacity,
        starts=[capacity - 3, 2 * capacity - 2],
    )
