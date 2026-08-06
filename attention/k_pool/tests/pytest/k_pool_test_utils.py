from __future__ import annotations

from typing import Any

import torch

from k_pool_golden import (
    compare_golden_and_oracle,
    run_k_pool_golden,
    run_k_pool_oracle,
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
        seq_lens = None
    else:
        hidden = torch.randn((sum(lengths), h), dtype=torch.bfloat16)
        seq_lens = torch.tensor(lengths, dtype=torch.int32)
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
        seq_lens,
    )


def make_manifest_inputs(case: dict[str, Any]):
    batch = int(case["B"])
    if case["layout"] == "TH":
        lengths = [int(item) for item in str(case["seq_lens"]).split("|")]
    else:
        lengths = [int(case["length"])] * batch
    starts = [int(item) for item in str(case["start_pos"]).split("|")]
    if len(starts) != batch:
        raise ValueError(f"{case['case_id']}: start_pos count does not match B")
    if len(lengths) != batch:
        raise ValueError(f"{case['case_id']}: seq_lens count does not match B")
    return make_inputs(
        str(case["layout"]),
        lengths,
        int(case["cmp_ratio"]),
        int(case["block_size"]),
        starts,
        norm=case["norm"] == "on",
    )


def make_invalid_manifest_inputs(case: dict[str, Any]):
    reason = str(case["invalid_reason"])
    base_layout = "TH" if reason.startswith("TH_") or reason.startswith("sum_seq") or reason.startswith("seq_lens") else "BSH"
    lengths = [4, 4]
    args = list(make_inputs(base_layout, lengths, 4, 128, [0, 0], norm=False))
    kwargs: dict[str, Any] = {"cmp_ratio": 4, "enforce_model_dims": False}

    if reason == "cmp_ratio_not_supported":
        kwargs["cmp_ratio"] = 3
        args[3] = torch.randn((3, 8), dtype=torch.float32)
    elif reason == "hidden_states_dtype_not_bf16":
        args[0] = args[0].float()
    elif reason == "wk_shape_not_D_by_H":
        args[1] = torch.randn((7, 16), dtype=torch.bfloat16)
    elif reason == "gate_weight_shape_mismatch":
        args[2] = torch.randn((7, 16), dtype=torch.bfloat16)
    elif reason == "state_cache_dtype_not_fp32":
        args[4] = args[4].to(torch.bfloat16)
    elif reason == "ape_first_dim_not_cmp_ratio":
        args[3] = torch.randn((3, 8), dtype=torch.float32)
    elif reason == "cache_block_table_dtype_not_int32":
        args[5] = args[5].to(torch.int64)
    elif reason == "start_pos_negative":
        args[6][0] = -1
    elif reason == "updated_length_exceeds_cache_capacity":
        args[6][:] = 1024
    elif reason == "sum_seq_lens_not_T":
        args[11] = torch.tensor([3, 3], dtype=torch.int32)
    elif reason == "TH_seq_lens_missing":
        args[11] = None
    elif reason == "BSH_seq_lens_not_empty":
        args[11] = torch.tensor([4, 4], dtype=torch.int32)
    elif reason == "norm_weight_bias_not_paired":
        args[7] = torch.ones((8,), dtype=torch.float32)
    elif reason == "norm_eps_not_positive":
        kwargs["norm_eps"] = 0.0
    elif reason == "cos_sin_not_paired":
        args[9] = torch.ones((2, 4, 4), dtype=torch.bfloat16)
    elif reason == "cos_sin_shape_mismatch":
        args[9] = torch.ones((2, 4, 4), dtype=torch.bfloat16)
        args[10] = torch.ones((2, 4, 2), dtype=torch.bfloat16)
    elif reason == "rotary_mode_not_0_or_1":
        kwargs["rotary_mode"] = 2
    elif reason == "cache_block_id_out_of_range":
        args[5][0, 0] = args[4].size(0)
    elif reason == "state_cache_empty":
        args[4] = torch.empty((0, 128, 16), dtype=torch.float32)
    elif reason == "pooled_key_shape_mismatch":
        raise NotImplementedError("output tensor shape is validated by aclnn wrapper, not CPU Golden")
    elif reason == "history_cache_has_uninitialized_hole":
        raise NotImplementedError("history validity metadata is a caller contract in the mask-free interface")
    elif reason == "input_contains_padding":
        raise NotImplementedError("padding detection requires removed attention_mask metadata")
    elif reason == "cache_block_zero_is_invalid":
        args[5][0, 0] = 0
    elif reason == "hidden_states_rank_not_2_or_3":
        args[0] = args[0].unsqueeze(0)
    elif reason == "ape_dtype_not_fp32":
        args[3] = args[3].to(torch.bfloat16)
    elif reason == "norm_dtype_not_fp32":
        args[7] = torch.ones((8,), dtype=torch.bfloat16)
        args[8] = torch.ones((8,), dtype=torch.bfloat16)
    elif reason == "rope_dtype_not_bf16":
        args[9] = torch.ones((2, 4, 4), dtype=torch.float32)
        args[10] = torch.ones_like(args[9])
    elif reason == "seq_lens_dtype_not_int32":
        args[11] = args[11].to(torch.int64)
    elif reason == "start_pos_dtype_not_int32":
        args[6] = args[6].to(torch.int64)
    elif reason == "aclnn_cmp_ratio_must_be_explicit":
        raise NotImplementedError("C API explicit attribute requirement is not expressible in Python Golden")
    else:
        raise AssertionError(f"unhandled invalid reason: {reason}")
    return args, kwargs


def run_pair(*args, **kwargs):
    golden_args = [item.clone() if isinstance(item, torch.Tensor) else item for item in args]
    oracle_args = [item.clone() if isinstance(item, torch.Tensor) else item for item in args]
    golden, golden_intermediates = run_k_pool_golden(
        *golden_args, enforce_model_dims=False, **kwargs
    )
    oracle, oracle_cache = run_k_pool_oracle(
        *oracle_args, enforce_model_dims=False, **kwargs
    )
    compare_golden_and_oracle(
        golden, oracle, golden_intermediates["state_cache"], oracle_cache
    )
    return golden
