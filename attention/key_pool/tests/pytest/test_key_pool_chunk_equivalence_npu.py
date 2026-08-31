from __future__ import annotations

import json

import torch
import torch_npu

from key_pool_public_loader import load_key_pool_public_api
from test_key_pool_layout_equivalence_npu import _run_npu, _segments
from test_key_pool_public_npu import CASES, _make_case_inputs


def _split_hidden(hidden_states, layout, lengths, first_lengths):
    if layout == "BSH":
        first = hidden_states[:, : first_lengths[0]]
        second = hidden_states[:, first_lengths[0] :]
        return first, second
    segments = _segments(hidden_states, lengths)
    first = torch.cat(
        [segment[:first_len] for segment, first_len in zip(segments, first_lengths)],
        dim=0,
    )
    second = torch.cat(
        [segment[first_len:] for segment, first_len in zip(segments, first_lengths)],
        dim=0,
    )
    return first, second


def _new_pool_count(start_pos: int, length: int, cmp_ratio: int) -> int:
    return (start_pos + length) // cmp_ratio - start_pos // cmp_ratio


def _active_tail_mask(
    state_cache,
    cache_block_table,
    start_pos,
    lengths,
    cmp_ratio,
):
    """Select only the cache rows that remain valid after the final call.

    KeyPool persists the final incomplete group only. A chunked call can leave
    stale bytes from an intermediate tail after that group is later pooled;
    those bytes are intentionally outside the semantic cache state.
    """
    mask = torch.zeros(
        state_cache.shape[:2],
        dtype=torch.bool,
    )
    block_size = state_cache.size(1)
    for batch, (start, length) in enumerate(zip(start_pos.tolist(), lengths)):
        end_pos = start + length
        tail_len = end_pos % cmp_ratio
        tail_start = max(start, end_pos - tail_len)
        if tail_start == end_pos:
            continue
        logical = torch.arange(tail_start, end_pos, dtype=torch.long)
        physical = cache_block_table[batch, logical // block_size].long()
        mask[physical, logical % block_size] = True
    return mask


def _assemble_chunk_output(
    first_output,
    second_output,
    start_pos,
    first_lengths,
    second_lengths,
    cmp_ratio,
):
    """Restore the full-call pool order from two incremental outputs."""
    assembled = torch.zeros_like(first_output)
    for batch, (start, first_len, second_len) in enumerate(
        zip(start_pos.tolist(), first_lengths, second_lengths)
    ):
        first_count = _new_pool_count(start, first_len, cmp_ratio)
        second_start = start + first_len
        second_count = _new_pool_count(second_start, second_len, cmp_ratio)
        assembled[batch, :first_count] = first_output[batch, :first_count]
        assembled[batch, first_count : first_count + second_count] = second_output[
            batch, :second_count
        ]
    return assembled


def _run_case(case_name, first_lengths, norm_on):
    case = CASES[case_name]
    (
        hidden_states,
        wk,
        gate_weight,
        ape,
        state_cache,
        cache_block_table,
        start_pos,
        cu_seqlens,
    ) = _make_case_inputs(case)
    norm_weight = norm_bias = None
    if norm_on:
        norm_weight = torch.randn((wk.size(0),), dtype=torch.float32)
        norm_bias = torch.randn((wk.size(0),), dtype=torch.float32)

    full_output, full_cache = _run_npu(
        hidden_states,
        wk,
        gate_weight,
        ape,
        state_cache.clone(),
        cache_block_table,
        start_pos,
        cu_seqlens,
        case.cmp_ratio,
        norm_weight,
        norm_bias,
    )

    first_hidden, second_hidden = _split_hidden(
        hidden_states, case.layout, case.lengths, first_lengths
    )
    first_cu_seqlens = (
        None
        if case.layout == "BSH"
        else torch.tensor(
            [0, *torch.tensor(first_lengths, dtype=torch.int32).cumsum(0).tolist()],
            dtype=torch.int32,
        )
    )
    second_lengths = tuple(
        total - first
        for total, first in zip(case.lengths, first_lengths)
    )
    second_cu_seqlens = (
        None
        if case.layout == "BSH"
        else torch.tensor(
            [0, *torch.tensor(second_lengths, dtype=torch.int32).cumsum(0).tolist()],
            dtype=torch.int32,
        )
    )
    first_output, first_cache = _run_npu(
        first_hidden,
        wk,
        gate_weight,
        ape,
        state_cache.clone(),
        cache_block_table,
        start_pos,
        first_cu_seqlens,
        case.cmp_ratio,
        norm_weight,
        norm_bias,
    )
    second_start_pos = start_pos + torch.tensor(first_lengths, dtype=torch.int32)
    chunk_output, chunk_cache = _run_npu(
        second_hidden,
        wk,
        gate_weight,
        ape,
        first_cache,
        cache_block_table,
        second_start_pos,
        second_cu_seqlens,
        case.cmp_ratio,
        norm_weight,
        norm_bias,
    )

    assembled_chunk_output = _assemble_chunk_output(
        first_output,
        chunk_output,
        start_pos,
        first_lengths,
        second_lengths,
        case.cmp_ratio,
    )
    output_equal = torch.equal(full_output, assembled_chunk_output)
    active_tail_mask = _active_tail_mask(
        state_cache,
        cache_block_table,
        start_pos,
        case.lengths,
        case.cmp_ratio,
    )
    cache_bitwise = torch.equal(
        full_cache[active_tail_mask],
        chunk_cache[active_tail_mask],
    )
    cache_close = bool(
        torch.isclose(
            full_cache[active_tail_mask],
            chunk_cache[active_tail_mask],
            rtol=0.0078125,
            atol=0.001,
            equal_nan=True,
        ).all().item()
    )
    result = {
        "case": case_name,
        "norm_on": norm_on,
        "first_lengths": first_lengths,
        "second_lengths": second_lengths,
        "output_bitwise": output_equal,
        "active_tail_cache_bitwise": cache_bitwise,
        "active_tail_cache_precision": cache_close,
        "active_tail_rows": int(active_tail_mask.sum().item()),
        "pass": output_equal and cache_close,
    }
    if not cache_bitwise:
        diff = (full_cache[active_tail_mask] - chunk_cache[active_tail_mask]).abs()
        result["cache_max_abs"] = float(diff.max().item())
        result["active_tail_cache_bitwise_mismatch_count"] = int(
            torch.count_nonzero(
                full_cache[active_tail_mask] != chunk_cache[active_tail_mask]
            )
        )
    return result


def main():
    load_key_pool_public_api()
    torch_npu.npu.set_device(0)
    cases = (
        ("bsh_history_r4", (2,)),
        ("th_full_history_r4", (1, 1)),
        ("th_unequal_r4", (1, 2)),
    )
    results = [
        _run_case(case_name, first_lengths, norm_on)
        for case_name, first_lengths in cases
        for norm_on in (False, True)
    ]
    print(json.dumps(results, indent=2))
    if not all(item["pass"] for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
