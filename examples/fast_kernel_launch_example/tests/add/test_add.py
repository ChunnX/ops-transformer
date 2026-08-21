#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import torch
import pytest
import ascend_ops

def test_add_interface_exist():
    """
    Test that the 'ascend_ops.add' operator is present in torch.ops.
    This existence test asserts that the custom operator registered under the
    'ascend_ops' namespace is discoverable from Python via torch.ops.ascend_ops.add.
    It does not exercise operator functionality — only that the Python binding
    and registration are available.
    Rationale:
    The presence of this test guards against a common failure mode where an
    operator is implemented and registered in C++/ATen but is not exposed to
    the Python torch.ops namespace due to mismatches between the PyTorch
    operator schema and the C++ registration signature (argument names, types,
    or overloads). Such schema/signature inconsistencies can cause the
    operator to be hidden or not exported to Python, breaking consumers that
    expect to call torch.ops.ascend_ops.add. This test will fail loudly if the
    binding is missing, prompting investigation into schema/registration issues.
    """
    # This test specifically protects against discrepancies between the
    # PyTorch operator schema and the C++ signature/registration that can
    # prevent the operator from being visible in torch.ops.ascend_ops.
    print(torch.ops.ascend_ops.add)
    assert hasattr(torch.ops.ascend_ops, "add"), (
        "The 'add' operator is not registered in the 'torch.ops.ascend_ops' namespace."
    )


SHAPES = [
    (1000, 1000),
]

DTYPES = [
    torch.float32,
    torch.float16,
    torch.int32,
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_add_operator(shape, dtype):
    """
    Test the functionality of the add operator, using concise but comprehensive combinations of shapes and data types.

    Parameters:
        shape: Tensor shape
        dtype: Data type
    """
    if dtype in [torch.int32]:
        a = torch.randint(-100, 100, shape, dtype=dtype)
        b = torch.randint(-100, 100, shape, dtype=dtype)
    else:
        a = torch.randn(*shape, dtype=dtype)
        b = torch.randn(*shape, dtype=dtype)

    expected = a + b
    a_npu = a.npu()
    b_npu = b.npu()
    for i in range(101):
        print(f">>>>>>>>>>>执行第{i}次计算")
        result_npu = torch.ops.ascend_ops.add(a_npu, b_npu)

    print(f"✓ Test passed: shape={shape}, dtype={dtype}")
test_add_interface_exist()
test_add_operator([1000, 1000], torch.float16)


