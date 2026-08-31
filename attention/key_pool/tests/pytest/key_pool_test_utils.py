from __future__ import annotations

import torch

from key_pool_golden import (
    compare_golden_and_oracle,
    run_key_pool_golden,
    run_key_pool_oracle,
)


def make_inputs(
    layout: str,
    lengths: list[int],
    ratio: int,
    block_size: int,
    start_pos: list[int],
    norm: bool = False,
):
    seed = (
        20260806
        + ratio
        + block_size
        + sum(lengths)
        + sum(start_pos)
        + len(lengths) * 17
    )
    torch.manual_seed(seed)
    h = 16
    d = 8
    max_updated = max(
        (start + length for start, length in zip(start_pos, lengths)),
        default=0,
    )
    max_blocks = max(1, (max_updated + block_size - 1) // block_size)
    block_num = 1 + len(lengths) * max_blocks
    table = torch.zeros((len(lengths), max_blocks), dtype=torch.int32)
    next_block = 1
    for batch in range(len(lengths)):
        for logical in range(max_blocks):
            table[batch, logical] = next_block
            next_block += 1
    cache = torch.randn((block_num, block_size, 2 * d), dtype=torch.float32)
    if layout == "BSH":
        if len(set(lengths)) > 1:
            raise ValueError("BSH helper requires equal per-batch lengths")
        hidden = torch.randn((len(lengths), lengths[0], h), dtype=torch.bfloat16)
        cu_seqlens = None
    else:
        hidden = torch.randn((sum(lengths), h), dtype=torch.bfloat16)
        cu_seqlens = torch.tensor(
            [0, *torch.tensor(lengths, dtype=torch.int32).cumsum(0).tolist()],
            dtype=torch.int32,
        )
    wk = torch.randn((d, h), dtype=torch.bfloat16)
    gate_weight = torch.randn((d, h), dtype=torch.bfloat16)
    ape = torch.randn((ratio, d), dtype=torch.float32)
    if norm:
        norm_weight = torch.randn((d,), dtype=torch.float32)
        norm_bias = torch.randn((d,), dtype=torch.float32)
    else:
        norm_weight = norm_bias = None
    return (
        hidden,
        wk,
        gate_weight,
        ape,
        cache,
        table,
        torch.tensor(start_pos, dtype=torch.int32),
        norm_weight,
        norm_bias,
        None,
        None,
        cu_seqlens,
        None,
    )


def run_pair(*args, **kwargs):
    golden_args = [item.clone() if isinstance(item, torch.Tensor) else item for item in args]
    oracle_args = [item.clone() if isinstance(item, torch.Tensor) else item for item in args]
    golden, golden_intermediates = run_key_pool_golden(
        *golden_args, enforce_model_dims=False, **kwargs
    )
    oracle, oracle_cache = run_key_pool_oracle(
        *oracle_args, enforce_model_dims=False, **kwargs
    )
    compare_golden_and_oracle(
        golden, oracle, golden_intermediates["state_cache"], oracle_cache
    )
    return golden
