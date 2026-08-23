from __future__ import annotations

import inspect

import pytest
import torch

from key_pool_public_loader import load_key_pool_public_api


def test_key_pool_public_schema_and_meta():
    torch_npu = pytest.importorskip("torch_npu")
    load_key_pool_public_api()

    assert callable(torch_npu.key_pool)
    assert not hasattr(torch_npu, "key_pool_npu")
    parameters = inspect.signature(torch_npu.key_pool).parameters
    assert list(parameters)[:7] == [
        "hidden_states",
        "wk",
        "gate_weight",
        "ape",
        "state_cache",
        "cache_block_table",
        "start_pos",
    ]
    assert "cu_seqlens" in parameters
    assert "seqused" in parameters

    hidden_states = torch.empty((2, 4, 4096), dtype=torch.bfloat16, device="meta")
    wk = torch.empty((128, 4096), dtype=torch.bfloat16, device="meta")
    gate_weight = torch.empty_like(wk)
    ape = torch.empty((4, 128), dtype=torch.float32, device="meta")
    state_cache = torch.empty((8, 16, 256), dtype=torch.float32, device="meta")
    cache_block_table = torch.empty((2, 4), dtype=torch.int32, device="meta")
    start_pos = torch.empty((2,), dtype=torch.int32, device="meta")

    output = torch.ops.cann_ops_transformer.key_pool(
        hidden_states,
        wk,
        gate_weight,
        ape,
        state_cache,
        cache_block_table,
        start_pos,
        cmp_ratio=4,
    )
    assert output.shape == (2, 16, 128)
    assert output.dtype == torch.bfloat16


def test_key_pool_public_accepts_layer_norm_and_rejects_rope():
    torch_npu = pytest.importorskip("torch_npu")
    load_key_pool_public_api()

    hidden_states = torch.empty((1, 1, 4096), dtype=torch.bfloat16, device="meta")
    wk = torch.empty((128, 4096), dtype=torch.bfloat16, device="meta")
    gate_weight = torch.empty_like(wk)
    ape = torch.empty((4, 128), dtype=torch.float32, device="meta")
    state_cache = torch.empty((2, 4, 256), dtype=torch.float32, device="meta")
    cache_block_table = torch.empty((1, 2), dtype=torch.int32, device="meta")
    start_pos = torch.empty((1,), dtype=torch.int32, device="meta")

    output = torch_npu.key_pool(
        hidden_states,
        wk,
        gate_weight,
        ape,
        state_cache,
        cache_block_table,
        start_pos,
        norm_weight=torch.empty((128,), dtype=torch.float32, device="meta"),
        norm_bias=torch.empty((128,), dtype=torch.float32, device="meta"),
    )
    assert output.shape == (1, 2, 128)

    with pytest.raises(ValueError, match="pair"):
        torch_npu.key_pool(
            hidden_states,
            wk,
            gate_weight,
            ape,
            state_cache,
            cache_block_table,
            start_pos,
            cos=torch.empty((1, 1, 64), dtype=torch.bfloat16, device="meta"),
        )

    with pytest.raises(NotImplementedError, match="RoPE"):
        torch_npu.key_pool(
            hidden_states,
            wk,
            gate_weight,
            ape,
            state_cache,
            cache_block_table,
            start_pos,
            cos=torch.empty((1, 1, 64), dtype=torch.bfloat16, device="meta"),
            sin=torch.empty((1, 1, 64), dtype=torch.bfloat16, device="meta"),
        )


def test_key_pool_public_th_cu_seqlens_and_seqused_contract():
    torch_npu = pytest.importorskip("torch_npu")
    load_key_pool_public_api()

    hidden_states = torch.empty((5, 4096), dtype=torch.bfloat16, device="meta")
    wk = torch.empty((128, 4096), dtype=torch.bfloat16, device="meta")
    gate_weight = torch.empty_like(wk)
    ape = torch.empty((4, 128), dtype=torch.float32, device="meta")
    state_cache = torch.empty((8, 16, 256), dtype=torch.float32, device="meta")
    cache_block_table = torch.empty((2, 4), dtype=torch.int32, device="meta")
    start_pos = torch.empty((2,), dtype=torch.int32, device="meta")
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32, device="meta")

    output = torch_npu.key_pool(
        hidden_states,
        wk,
        gate_weight,
        ape,
        state_cache,
        cache_block_table,
        start_pos,
        cu_seqlens=cu_seqlens,
    )
    assert output.shape == (2, 16, 128)

    with pytest.raises(ValueError, match="cu_seqlens is required"):
        torch_npu.key_pool(
            hidden_states, wk, gate_weight, ape, state_cache, cache_block_table, start_pos
        )

    with pytest.raises(ValueError, match="seqused"):
        torch_npu.key_pool(
            hidden_states,
            wk,
            gate_weight,
            ape,
            state_cache,
            cache_block_table,
            start_pos,
            cu_seqlens=cu_seqlens,
            seqused=torch.empty((2,), dtype=torch.int32, device="meta"),
        )
