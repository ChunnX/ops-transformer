# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch


CHUNK_SIZE = 64
INVERSE_SHAPE = 32


def make_cc_input(tile_num, seed=0, dtype=torch.bfloat16, device="cpu"):
    torch.manual_seed(seed)
    data = torch.randn(tile_num, CHUNK_SIZE, CHUNK_SIZE, dtype=torch.float32) * 0.05
    data = torch.tril(data, diagonal=-1)
    return data.to(dtype=dtype, device=device)


def inverse_32_block(block):
    lower = torch.tril(block.float(), diagonal=-1)
    inv = torch.zeros_like(lower)
    eye_idx = torch.arange(INVERSE_SHAPE, device=block.device)
    inv[..., eye_idx, eye_idx] = 1.0
    for i in range(1, INVERSE_SHAPE):
        inv[..., i, :i] = -torch.matmul(lower[..., i : i + 1, :i], inv[..., :i, :i]).squeeze(-2)
    return inv


def tri_inverse_golden(cc_in):
    a = torch.tril(cc_in.float(), diagonal=-1)

    top_left = inverse_32_block(a[..., :INVERSE_SHAPE, :INVERSE_SHAPE]).to(torch.bfloat16)
    right_down = inverse_32_block(a[..., INVERSE_SHAPE:, INVERSE_SHAPE:]).to(torch.bfloat16)

    left_down = (-a[..., INVERSE_SHAPE:, :INVERSE_SHAPE]).to(torch.bfloat16)
    left_down = (left_down.float() @ top_left.float()).to(torch.bfloat16)
    left_down = (right_down.float() @ left_down.float()).to(torch.bfloat16)

    out = torch.zeros_like(a)
    out[..., :INVERSE_SHAPE, :INVERSE_SHAPE] = top_left.float()
    out[..., INVERSE_SHAPE:, INVERSE_SHAPE:] = right_down.float()
    out[..., INVERSE_SHAPE:, :INVERSE_SHAPE] = left_down.float()
    return out.to(cc_in.dtype)


def compare_tri_inverse(actual, expected, rtol=7.8125e-3, atol=1e-3):
    actual_f = actual.float().cpu()
    expected_f = expected.float().cpu()
    diff = (actual_f - expected_f).abs()
    rel = diff / (expected_f.abs() + 1e-3)
    max_abs = diff.max().item()
    max_re = rel.max().item()
    avg_re = rel.mean().item()
    rmse = torch.sqrt(torch.mean((actual_f - expected_f) ** 2)).item()
    passed = torch.allclose(actual_f, expected_f, rtol=rtol, atol=atol)
    return {
        "passed": passed,
        "max_abs": max_abs,
        "max_re": max_re,
        "avg_re": avg_re,
        "rmse": rmse,
    }
