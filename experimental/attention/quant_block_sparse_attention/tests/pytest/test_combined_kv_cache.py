#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch

import combined_kv_cache


def _sample_dense_inputs():
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch.float8_e4m3fn is required for combined KV cache tests")
    fp8_dtype = torch.float8_e4m3fn
    dense_key = torch.arange(2 * 5 * 2 * 8, dtype=torch.float32).reshape(2, 5, 2, 8).to(fp8_dtype)
    dense_value = (-torch.arange(2 * 5 * 2 * 8, dtype=torch.float32)).reshape(2, 5, 2, 8).to(fp8_dtype)
    dense_k_scale = torch.arange(2 * 5 * 2, dtype=torch.float32).reshape(2, 5, 2)
    block_table = torch.tensor([[1, 0], [3, 2]], dtype=torch.int32)
    return dense_key, dense_value, dense_k_scale, block_table


def test_combined_kv_cache_views_and_padding():
    dense_key, dense_value, dense_k_scale, block_table = _sample_dense_inputs()
    storage, meta = combined_kv_cache.pack_combined_kv_cache(
        dense_key, dense_value, dense_k_scale, block_table, block_size=4,
        block_padding_bytes=32, layout_kv="PA_BNSD")
    key, value, k_scale = combined_kv_cache.assert_combined_kv_views(storage, meta)

    assert meta["combine_bytes"] == 32
    assert meta["pa_block_stride"] == 4 * 2 * 32 + 32
    assert key.stride() == (meta["pa_block_stride"], 4 * 32, 32, 1)
    assert value.stride() == key.stride()
    assert tuple(stride * 4 for stride in k_scale.stride()) == key.stride()[:3]

    for batch_idx in range(2):
        for logical_block in range(2):
            physical_block = int(block_table[batch_idx, logical_block])
            start = logical_block * 4
            end = min(start + 4, 5)
            count = end - start
            torch.testing.assert_close(
                key[physical_block, :, :count].to(torch.float32),
                dense_key[batch_idx, start:end].permute(1, 0, 2).to(torch.float32))
            torch.testing.assert_close(
                value[physical_block, :, :count].to(torch.float32),
                dense_value[batch_idx, start:end].permute(1, 0, 2).to(torch.float32))
            torch.testing.assert_close(
                k_scale[physical_block, :, :count],
                dense_k_scale[batch_idx, start:end].permute(1, 0))


def test_combined_kv_cache_storage_serialization(tmp_path):
    dense_key, dense_value, dense_k_scale, block_table = _sample_dense_inputs()
    storage, meta = combined_kv_cache.pack_combined_kv_cache(
        dense_key, dense_value, dense_k_scale, block_table, block_size=4,
        block_padding_bytes=64, layout_kv="PA_BNSD")

    path = tmp_path / "combined_kv.pt"
    torch.save({"storage": storage, "meta": meta}, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    key, value, k_scale = combined_kv_cache.assert_combined_kv_views(loaded["storage"], loaded["meta"])

    assert key.shape == (4, 2, 4, 8)
    assert value.shape == key.shape
    assert k_scale.shape == (4, 2, 4)
    assert loaded["meta"]["pa_block_stride"] == meta["pa_block_stride"]
