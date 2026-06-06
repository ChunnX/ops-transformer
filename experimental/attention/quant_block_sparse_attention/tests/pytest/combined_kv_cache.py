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

import torch


ALIGN_BYTES = 32
FP32_BYTES = 4


def _fp8_dtype():
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("torch.float8_e4m3fn is required for combined KV cache views")
    return torch.float8_e4m3fn


def _align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def make_combined_kv_meta(block_num, block_size, n2, d_size, dv_size, block_padding_bytes, layout_kv):
    if layout_kv != "PA_BNSD":
        raise ValueError(f"unsupported combined KV layout: {layout_kv}")

    combine_bytes = _align_up(d_size + dv_size + FP32_BYTES, ALIGN_BYTES)
    pa_block_stride = block_size * n2 * combine_bytes + block_padding_bytes
    if pa_block_stride % FP32_BYTES != 0:
        raise ValueError(f"pa_block_stride should be divisible by {FP32_BYTES}, got {pa_block_stride}")

    return {
        "layout_kv": layout_kv,
        "block_num": block_num,
        "block_size": block_size,
        "n2": n2,
        "d_size": d_size,
        "dv_size": dv_size,
        "combine_bytes": combine_bytes,
        "pa_block_stride": pa_block_stride,
        "key_offset_bytes": 0,
        "value_offset_bytes": d_size,
        "k_scale_offset_bytes": d_size + dv_size,
    }


def make_combined_kv_views(storage, meta):
    if storage.dtype != torch.uint8 or storage.dim() != 1 or not storage.is_contiguous():
        raise ValueError("combined KV storage should be a contiguous rank-1 uint8 tensor")

    block_num = meta["block_num"]
    block_size = meta["block_size"]
    n2 = meta["n2"]
    d_size = meta["d_size"]
    dv_size = meta["dv_size"]
    combine_bytes = meta["combine_bytes"]
    pa_block_stride = meta["pa_block_stride"]
    required_bytes = block_num * pa_block_stride
    if storage.numel() < required_bytes:
        raise ValueError(f"combined KV storage has {storage.numel()} bytes, expected at least {required_bytes}")

    if meta["layout_kv"] != "PA_BNSD":
        raise ValueError(f"unsupported combined KV layout: {meta['layout_kv']}")

    fp8_storage = storage.view(_fp8_dtype())
    fp8_stride = (pa_block_stride, block_size * combine_bytes, combine_bytes, 1)
    key = torch.as_strided(
        fp8_storage, (block_num, n2, block_size, d_size), fp8_stride, meta["key_offset_bytes"])
    value = torch.as_strided(
        fp8_storage, (block_num, n2, block_size, dv_size), fp8_stride, meta["value_offset_bytes"])

    fp32_storage = storage.view(torch.float32)
    k_scale_stride = tuple(stride // FP32_BYTES for stride in fp8_stride[:3])
    k_scale = torch.as_strided(
        fp32_storage, (block_num, n2, block_size), k_scale_stride,
        meta["k_scale_offset_bytes"] // FP32_BYTES)
    return key, value, k_scale


def pack_combined_kv_cache(dense_key, dense_value, dense_k_scale, block_table, block_size,
                           block_padding_bytes, layout_kv):
    if dense_key.dim() != 4 or dense_value.dim() != 4 or dense_k_scale.dim() != 3:
        raise ValueError("dense key/value/k_scale should have ranks 4/4/3")
    if dense_key.shape[:3] != dense_value.shape[:3] or dense_key.shape[:3] != dense_k_scale.shape:
        raise ValueError("dense key/value/k_scale B/S/N dimensions should match")

    batch, seq_len, n2, d_size = dense_key.shape
    dv_size = dense_value.shape[-1]
    block_num = int(block_table.max().item()) + 1
    meta = make_combined_kv_meta(
        block_num, block_size, n2, d_size, dv_size, block_padding_bytes, layout_kv)
    storage = torch.zeros((block_num * meta["pa_block_stride"],), dtype=torch.uint8)
    key, value, k_scale = make_combined_kv_views(storage, meta)

    for batch_idx in range(batch):
        for logical_block in range(block_table.shape[1]):
            physical_block = int(block_table[batch_idx, logical_block].item())
            start = logical_block * block_size
            end = min(start + block_size, seq_len)
            token_count = end - start
            if token_count <= 0:
                continue
            key[physical_block, :, :token_count].copy_(
                dense_key[batch_idx, start:end].permute(1, 0, 2))
            value[physical_block, :, :token_count].copy_(
                dense_value[batch_idx, start:end].permute(1, 0, 2))
            k_scale[physical_block, :, :token_count].copy_(
                dense_k_scale[batch_idx, start:end].permute(1, 0))

    return storage, meta


def assert_combined_kv_views(storage, meta):
    key, value, k_scale = make_combined_kv_views(storage, meta)
    storage_ptr = storage.untyped_storage().data_ptr()
    if not (key.untyped_storage().data_ptr() == value.untyped_storage().data_ptr() == storage_ptr):
        raise AssertionError("key and value should share the combined KV storage")
    if k_scale.untyped_storage().data_ptr() != storage_ptr:
        raise AssertionError("k_scale should share the combined KV storage")

    offsets = (
        key.data_ptr() - storage.data_ptr(),
        value.data_ptr() - storage.data_ptr(),
        k_scale.data_ptr() - storage.data_ptr(),
    )
    expected_offsets = (
        meta["key_offset_bytes"],
        meta["value_offset_bytes"],
        meta["k_scale_offset_bytes"],
    )
    if offsets != expected_offsets:
        raise AssertionError(f"unexpected combined KV view offsets: {offsets}, expected {expected_offsets}")

    if key.stride(0) != meta["pa_block_stride"] or value.stride(0) != meta["pa_block_stride"]:
        raise AssertionError("FP8 key/value stride(0) should equal pa_block_stride")
    if k_scale.stride(0) * FP32_BYTES != meta["pa_block_stride"]:
        raise AssertionError("FP32 k_scale byte stride(0) should equal pa_block_stride")
    return key, value, k_scale
