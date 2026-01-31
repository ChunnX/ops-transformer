# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
import npu_ops_transformer

SHAPES = [
    (1,),
    (3,),
    (1024,),
    (10, 10),
]

DTYPES = [
    torch.float32,
    torch.float16,
    torch.bfloat16
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_abs(shape, dtype):
    x = torch.randn(*shape, dtype=dtype)
    cpu_result = torch.ops.aten.abs(x)
    npu_result = npu_ops_transformer.ops.abs(x.npu()).cpu()
    assert torch.allclose(cpu_result, npu_result, rtol=1e-6)
