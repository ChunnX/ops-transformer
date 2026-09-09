# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU reference decodes the actual packed bytes, independent of input generation."""

import hashlib
import math

import torch


CENTERS = (
    -0.12091285,
    -0.09111122,
    -0.07112455,
    -0.05513602,
    -0.04132067,
    -0.02874970,
    -0.01700489,
    -0.00568677,
    0.00547294,
    0.01680406,
    0.02857605,
    0.04108622,
    0.05492980,
    0.07101817,
    0.09115373,
    0.12037795,
)


def golden(
    query,
    key,
    value,
    sparse_indices,
    *,
    block_table,
    actual_seq_lengths_query,
    actual_seq_lengths_kv,
    scale_value=1.0,
    layout_kv="PA_BSND",
    sparse_block_size=1,
    sparse_mode=3,
    return_softmax_lse=False,
    **kwargs,
):
    """Return float64 attention, max and sum; no NPU or test-case cache is used."""
    del value, kwargs
    q = query.detach().cpu().double()
    cache = key.detach().cpu().contiguous().view(torch.uint8)
    if layout_kv == "PA_BNBD":
        cache = cache.transpose(1, 2).contiguous()
    elif layout_kv != "PA_BSND":
        raise ValueError("unsupported cache layout")
    block_size = cache.shape[1]
    packed = cache.reshape(-1, 386)
    codes = torch.empty((packed.shape[0], 512), dtype=torch.int64)
    codes[:, 0::2] = packed[:, :256] & 15
    codes[:, 1::2] = packed[:, :256] >> 4
    codebook = torch.tensor(CENTERS, dtype=query.dtype).double()
    latent = codebook[codes]
    rope = packed[:, 256:384].contiguous().view(query.dtype).double()
    scale = packed[:, 384:386].contiguous().view(torch.float16).double().flatten()
    indices = sparse_indices.detach().cpu().long()
    table = block_table.detach().cpu().long()
    q_ends = actual_seq_lengths_query.detach().cpu().tolist()
    kv_lengths = actual_seq_lengths_kv.detach().cpu().tolist()
    output = torch.zeros((*q.shape[:2], 512), dtype=torch.float64)
    maximum = torch.full((1, *q.shape[:2]), -2e38, dtype=torch.float32).double()
    total = torch.zeros_like(maximum)
    start = 0
    for batch, end in enumerate(q_ends):
        for row in range(start, end):
            visible = kv_lengths[batch]
            if sparse_mode == 3:
                visible = min(
                    visible, kv_lengths[batch] - (end - start) + row - start + 1
                )
            selected = []
            for block in indices[row, 0].tolist():
                if block == -1:
                    break
                selected.extend(
                    token
                    for token in range(
                        block * sparse_block_size, (block + 1) * sparse_block_size
                    )
                    if 0 <= token < visible
                )
            if not selected:
                if kv_lengths[batch] != 0:
                    # Nonempty KV with all-masked selection has implementation-defined finite-mask sum.
                    total[0, row] = float("nan")
                continue
            local = torch.tensor(selected, dtype=torch.int64)
            physical = (
                table[batch, local // block_size] * block_size + local % block_size
            )
            scores = (
                (
                    q[row, :, :512] @ latent[physical].T
                    + q[row, :, 512:] @ rope[physical].T
                )
                * scale[physical]
                * scale_value
            )
            maxima = scores.max(dim=-1).values
            weights = torch.exp(scores - maxima[:, None])
            sums = weights.sum(dim=-1)
            output[row] = (weights / sums[:, None]) @ (
                latent[physical] * scale[physical, None]
            )
            maximum[0, row] = maxima
            total[0, row] = sums
        start = end
    if not return_softmax_lse:
        maximum = total = torch.empty(0, dtype=torch.float64)
    return output, maximum, total


def generate_inputs(
    query,
    key,
    value,
    sparse_indices,
    *,
    block_table,
    actual_seq_lengths_query,
    actual_seq_lengths_kv,
    key_dequant_scale=None,
    value_dequant_scale=None,
    layout_kv="PA_BSND",
    sparse_block_size=1,
    sparse_mode=3,
    testcase_name=None,
    **kwargs,
):
    """Fill TTK-allocated tensors in place, retaining cache view strides and offsets."""
    del kwargs
    seed = int.from_bytes(
        hashlib.sha256(str(testcase_name).encode()).digest()[:4], "little"
    )
    generator = torch.Generator().manual_seed(seed)
    pages = key.shape[0]
    block_size = key.shape[2 if layout_kv == "PA_BNBD" else 1]
    slots = pages * block_size
    data = torch.zeros((slots, 386), dtype=torch.uint8)
    codes = torch.randint(0, 16, (slots, 512), generator=generator, dtype=torch.uint8)
    data[:, :256] = codes[:, 0::2] | (codes[:, 1::2] << 4)
    scale = (torch.rand(slots, generator=generator) * 2.875 + 0.125).half()
    rope = (
        torch.randn((slots, 64), generator=generator) * 0.02 / scale.float()[:, None]
    ).to(query.dtype)
    data[:, 256:384] = rope.view(torch.uint8)
    data[:, 384:386] = scale.view(torch.uint8).reshape(-1, 2)
    key.copy_(data.view(torch.int8).reshape(key.shape).to(key.device))
    value.copy_(key)
    query.copy_(
        (torch.randn(query.shape, generator=generator) * 0.1).to(
            device=query.device, dtype=query.dtype
        )
    )
    batch_size, columns = block_table.shape
    q_lengths = [
        query.shape[0] // batch_size + int(batch < query.shape[0] % batch_size)
        for batch in range(batch_size)
    ]
    kv_lengths = [
        columns * block_size - batch % block_size for batch in range(batch_size)
    ]
    for batch in range(1, batch_size, 3):
        kv_lengths[batch] = 0
    order = torch.randperm(pages, generator=generator)
    table = order[torch.arange(batch_size * columns) % pages].reshape(
        batch_size, columns
    )
    block_table.copy_(table.to(device=block_table.device, dtype=torch.int32))
    actual_seq_lengths_query.copy_(
        torch.tensor(q_lengths).cumsum(0).to(actual_seq_lengths_query)
    )
    actual_seq_lengths_kv.copy_(torch.tensor(kv_lengths).to(actual_seq_lengths_kv))
    selected = torch.full(sparse_indices.shape, -1, dtype=torch.int32)
    start = 0
    for batch, length in enumerate(q_lengths):
        for offset in range(length):
            visible = kv_lengths[batch]
            if sparse_mode == 3:
                visible = min(visible, kv_lengths[batch] - length + offset + 1)
            count = min(
                selected.shape[-1], math.ceil(max(0, visible) / sparse_block_size)
            )
            selected[start + offset, 0, :count] = torch.arange(count, dtype=torch.int32)
        start += length
    sparse_indices.copy_(selected.to(sparse_indices.device))
    for scale_tensor in (key_dequant_scale, value_dequant_scale):
        if scale_tensor is not None:
            scale_tensor.fill_(1)


def compare(*outputs):
    """TTK NPU outputs followed by golden outputs, with explicit per-output metrics."""
    if len(outputs) != 6:
        raise ValueError(
            "expected three actual outputs followed by three golden outputs"
        )
    results = []
    for index, (actual, expected) in enumerate(zip(outputs[:3], outputs[3:])):
        actual = torch.as_tensor(actual).detach().cpu().double()
        expected = torch.as_tensor(expected).detach().cpu().double()
        if actual.shape != expected.shape:
            results.append(
                {"pass": False, "precision": 0.0, "error_info": "output shape mismatch"}
            )
            continue
        # Only the finite-mask sum of a fully masked nonempty row is unspecified.
        mask = (
            ~torch.isnan(expected)
            if index == 2
            else torch.ones_like(expected, dtype=torch.bool)
        )
        actual, expected = actual[mask], expected[mask]
        error = (actual - expected).abs()
        matched = (
            float((error <= 2**-9 * (1 + expected.abs())).double().mean())
            if error.numel()
            else 1.0
        )
        maximum = float(error.max()) if error.numel() else 0.0
        passed = (
            bool(torch.isfinite(actual).all()) and matched >= 0.99 and maximum <= 0.1
        )
        results.append(
            {
                "pass": passed,
                "precision": matched,
                "error_info": None
                if passed
                else f"output[{index}] matched={matched}, max_abs={maximum}",
            }
        )
    return results
