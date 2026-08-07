from __future__ import annotations

import inspect

import pytest
import torch

from k_pool_public_loader import load_k_pool_public_api


def test_k_pool_public_schema_and_meta():
    torch_npu = pytest.importorskip("torch_npu")
    load_k_pool_public_api()

    assert callable(torch_npu.k_pool_npu)
    parameters = inspect.signature(torch_npu.k_pool_npu).parameters
    assert list(parameters)[:7] == [
        "hidden_states",
        "wk",
        "gate_weight",
        "ape",
        "state_cache",
        "cache_block_table",
        "start_pos",
    ]

    hidden_states = torch.empty((2, 4, 4096), dtype=torch.bfloat16, device="meta")
    wk = torch.empty((128, 4096), dtype=torch.bfloat16, device="meta")
    gate_weight = torch.empty_like(wk)
    ape = torch.empty((4, 128), dtype=torch.float32, device="meta")
    state_cache = torch.empty((8, 16, 256), dtype=torch.float32, device="meta")
    cache_block_table = torch.empty((2, 4), dtype=torch.int32, device="meta")
    start_pos = torch.empty((2,), dtype=torch.int32, device="meta")

    output = torch.ops.cann_ops_transformer.k_pool(
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


def test_k_pool_public_rejects_reserved_stages():
    torch_npu = pytest.importorskip("torch_npu")
    load_k_pool_public_api()

    hidden_states = torch.empty((1, 1, 4096), dtype=torch.bfloat16, device="meta")
    wk = torch.empty((128, 4096), dtype=torch.bfloat16, device="meta")
    gate_weight = torch.empty_like(wk)
    ape = torch.empty((4, 128), dtype=torch.float32, device="meta")
    state_cache = torch.empty((2, 4, 256), dtype=torch.float32, device="meta")
    cache_block_table = torch.empty((1, 2), dtype=torch.int32, device="meta")
    start_pos = torch.empty((1,), dtype=torch.int32, device="meta")

    with pytest.raises(NotImplementedError, match="LayerNorm"):
        torch_npu.k_pool_npu(
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
