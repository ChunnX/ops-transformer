#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program is free software, you can redistribute it and/or modify.
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This file is a part of the CANN Open Software.
Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
"""

import torch
import torch_npu
import numpy as np


def test_npu_ai_infra_mhc_post_grad():
    """
    Test npu_ai_infra_mhc_post_grad operator
    """
    # Test parameters
    B = 1      # batch size
    S = 4096      # sequence length
    n = 4      # number of heads
    D = 2560    # dimension

    # Create input tensors
    grad_output = torch.randn(B, S, n, D, dtype=torch.bfloat16).npu()
    x = torch.randn(B, S, n, D, dtype=torch.bfloat16).npu()
    h_res = torch.randn(B, S, n, n, dtype=torch.float32).npu()
    h_out = torch.randn(B, S, D, dtype=torch.bfloat16).npu()
    h_post = torch.randn(B, S, n, dtype=torch.float32).npu()

    # Call the operator
    grad_x, grad_h_res, grad_h_out, grad_h_post = torch.ops.custom.npu_ai_infra_mhc_post_grad(
        grad_output, x, h_res, h_out, h_post
    )

    # Verify output shapes
    assert grad_x.shape == (B, S, n, D), f"grad_x shape mismatch: {grad_x.shape}"
    assert grad_h_res.shape == (B, S, n, n), f"grad_h_res shape mismatch: {grad_h_res.shape}"
    assert grad_h_out.shape == (B, S, D), f"grad_h_out shape mismatch: {grad_h_out.shape}"
    assert grad_h_post.shape == (B, S, n), f"grad_h_post shape mismatch: {grad_h_post.shape}"

    # Verify output dtypes
    assert grad_x.dtype == torch.bfloat16, f"grad_x dtype mismatch: {grad_x.dtype}"
    assert grad_h_res.dtype == torch.float32, f"grad_h_res dtype mismatch: {grad_h_res.dtype}"
    assert grad_h_out.dtype == torch.bfloat16, f"grad_h_out dtype mismatch: {grad_h_out.dtype}"
    assert grad_h_post.dtype == torch.float32, f"grad_h_post dtype mismatch: {grad_h_post.dtype}"

    print("Test passed!")


if __name__ == "__main__":
    import omni_training_custom_ops
    test_npu_ai_infra_mhc_post_grad()
