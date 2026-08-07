from __future__ import annotations

import json

import torch
import torch_npu

from k_pool_public_loader import load_k_pool_public_api
from test_k_pool_layout_equivalence_npu import _run_npu, _segments
from test_k_pool_public_npu import CASES, _make_case_inputs


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
        seq_lens,
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
        seq_lens,
        case.cmp_ratio,
        norm_weight,
        norm_bias,
    )

    first_hidden, second_hidden = _split_hidden(
        hidden_states, case.layout, case.lengths, first_lengths
    )
    first_seq_lens = (
        None
        if case.layout == "BSH"
        else torch.tensor(first_lengths, dtype=torch.int32)
    )
    second_lengths = tuple(
        total - first
        for total, first in zip(case.lengths, first_lengths)
    )
    second_seq_lens = (
        None
        if case.layout == "BSH"
        else torch.tensor(second_lengths, dtype=torch.int32)
    )
    _, first_cache = _run_npu(
        first_hidden,
        wk,
        gate_weight,
        ape,
        state_cache.clone(),
        cache_block_table,
        start_pos,
        first_seq_lens,
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
        second_seq_lens,
        case.cmp_ratio,
        norm_weight,
        norm_bias,
    )

    output_equal = torch.equal(full_output, chunk_output)
    cache_bitwise = torch.equal(full_cache, chunk_cache)
    cache_close = bool(
        torch.isclose(
            full_cache,
            chunk_cache,
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
        "cache_bitwise": cache_bitwise,
        "cache_precision": cache_close,
        "pass": output_equal and cache_close,
    }
    if not cache_bitwise:
        diff = (full_cache - chunk_cache).abs()
        result["cache_max_abs"] = float(diff.max().item())
        result["cache_bitwise_mismatch_count"] = int(
            torch.count_nonzero(full_cache != chunk_cache)
        )
    return result


def main():
    load_k_pool_public_api()
    torch_npu.npu.set_device(0)
    cases = (
        ("bsh_history_r4", (2,)),
        ("th_full_history_r4", (1, 1)),
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
