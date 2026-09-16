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

"""Minimal functional test for FusedScatterCopyQuantSparseFlashAttention."""

from __future__ import annotations

import os

import pytest
import torch
import torch_npu

from fused_scatter_copy_quant_sparse_flash_attention_golden import make_test_data


def swapped_from_cpu(cpu_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if not hasattr(torch_npu, "empty_with_swapped_memory"):
        raise RuntimeError("torch_npu.empty_with_swapped_memory is required for the DRAM input")
    swapped = torch_npu.empty_with_swapped_memory(
        cpu_tensor.shape, dtype=cpu_tensor.dtype, device=device
    )
    swapped.zero_()
    swapped.add_(cpu_tensor.to(device))
    torch.npu.synchronize()
    return swapped


def check_attention(actual: torch.Tensor, golden: torch.Tensor) -> None:
    actual = actual.detach().cpu().float()
    golden = golden.float()
    if actual.shape != golden.shape or not bool(torch.isfinite(actual).all()):
        raise AssertionError("attention output has an invalid shape or non-finite value")
    relative_l2 = (
        (actual - golden).flatten(1).norm(dim=1)
        / golden.flatten(1).norm(dim=1).clamp_min(1e-7)
    )
    if not torch.allclose(actual, golden, atol=0.008, rtol=0.05) or float(relative_l2.max()) > 0.08:
        raise AssertionError(
            f"attention output differs from CPU golden, max_rel_l2={float(relative_l2.max()):.6f}"
        )


@pytest.mark.ci
@pytest.mark.parametrize("miss_count", [8, 513], ids=["fused_miss", "union_copy_fallback"])
def test_fused_scatter_copy_quant_sparse_flash_attention(miss_count: int) -> None:
    device_id = int(os.getenv("DEVICE_ID", "0"))
    torch.npu.set_device(device_id)
    device = torch.device(f"npu:{device_id}")
    op = getattr(torch_npu, "npu_fused_scatter_copy_quant_sparse_flash_attention", None)
    if op is None:
        raise RuntimeError(
            "torch_npu.npu_fused_scatter_copy_quant_sparse_flash_attention is unavailable; "
            "install and activate the custom operator package first"
        )

    data = make_test_data(miss_count)
    hbm_kv_bytes = data.initial_hbm.to(device)
    dram_kv_bytes = swapped_from_cpu(data.dram, device)
    inputs = [
        data.query,
        data.hbm_block_table,
        data.dram_block_table,
        data.topk_dst_slots,
        data.topk_src_ids,
        data.topk_miss_counts,
        data.actual_seq_lengths_query,
        data.resident_seq_lengths,
        data.copy_src_ids,
        data.copy_dst_slots,
        data.copy_counts,
    ]
    device_inputs = [tensor.to(device) for tensor in inputs]

    result = op(
        device_inputs[0],
        hbm_kv_bytes,
        dram_kv_bytes,
        *device_inputs[1:],
        float(data.scale_value),
    )
    torch.npu.synchronize()
    if isinstance(result, torch.Tensor):
        hbm_result, attention_out = hbm_kv_bytes, result
    elif isinstance(result, (tuple, list)) and len(result) == 1:
        hbm_result, attention_out = hbm_kv_bytes, result[0]
    elif isinstance(result, (tuple, list)) and len(result) == 2:
        hbm_result, attention_out = result
    else:
        raise AssertionError("operator must return attention_out and may also return the in-place HBM alias")
    if not torch.equal(hbm_result.cpu(), data.expected_hbm):
        raise AssertionError("scatter copy output differs from byte-exact golden")
    if not torch.equal(hbm_kv_bytes.cpu(), data.expected_hbm):
        raise AssertionError("hbm_kv_bytes was not updated in place")
    check_attention(attention_out, data.attention_golden)
