from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import torch
import torch_npu

from k_pool_golden import run_k_pool_golden
from k_pool_public_loader import load_k_pool_public_api


@dataclass(frozen=True)
class Case:
    name: str
    layout: str
    lengths: tuple[int, ...]
    start_pos: tuple[int, ...]
    cmp_ratio: int
    block_size: int = 4
    scramble_blocks: bool = False


CASES = {
    case.name: case
    for case in (
        Case("bsh_minimal_r4", "BSH", (4,), (0,), 4),
        Case("bsh_minimal_r2", "BSH", (2,), (0,), 2),
        Case("bsh_history_r4", "BSH", (5,), (3,), 4),
        Case("th_unequal_r4", "TH", (3, 5), (0, 2), 4),
        Case("bsh_minimal_r8", "BSH", (8,), (0,), 8),
        Case("bsh_history_r16", "BSH", (16,), (16,), 16),
        Case("bsh_history_r32", "BSH", (32,), (32,), 32),
        Case("bsh_history_r64", "BSH", (64,), (64,), 64),
        Case("bsh_history_r128", "BSH", (128,), (128,), 128),
        Case("bsh_full_history_r4", "BSH", (4,), (8,), 4),
        Case("bsh_cross_block_r4", "BSH", (4,), (3,), 4, block_size=3),
        Case(
            "th_full_history_r4",
            "TH",
            (2, 3),
            (6, 9),
            4,
            block_size=3,
            scramble_blocks=True,
        ),
    )
}


def _make_case_inputs(case: Case):
    torch.manual_seed(20260807 + case.cmp_ratio + sum(case.lengths) + sum(case.start_pos))
    hidden_size = 4096
    head_dim = 128
    block_size = case.block_size
    max_updated = max(
        start + length for start, length in zip(case.start_pos, case.lengths)
    )
    max_blocks = max(1, (max_updated + block_size - 1) // block_size)
    block_num = 1 + len(case.lengths) * max_blocks

    cache_block_table = torch.zeros(
        (len(case.lengths), max_blocks), dtype=torch.int32
    )
    physical_block = 1
    for batch in range(len(case.lengths)):
        for logical_block in range(max_blocks):
            cache_block_table[batch, logical_block] = physical_block
            physical_block += 1
        if case.scramble_blocks:
            cache_block_table[batch] = cache_block_table[batch].flip(0)

    if case.layout == "BSH":
        if len(set(case.lengths)) != 1:
            raise ValueError("BSH cases require equal per-batch lengths")
        hidden_states = torch.randn(
            (len(case.lengths), case.lengths[0], hidden_size),
            dtype=torch.bfloat16,
        )
        seq_lens = None
    else:
        hidden_states = torch.randn(
            (sum(case.lengths), hidden_size), dtype=torch.bfloat16
        )
        seq_lens = torch.tensor(case.lengths, dtype=torch.int32)

    wk = (torch.randn(head_dim, hidden_size) * 0.02).to(torch.bfloat16)
    gate_weight = (torch.randn(head_dim, hidden_size) * 0.02).to(torch.bfloat16)
    ape = torch.randn(case.cmp_ratio, head_dim, dtype=torch.float32) * 0.1
    state_cache = (
        torch.randn(block_num, block_size, 2 * head_dim, dtype=torch.float32)
        * 0.1
    )
    start_pos = torch.tensor(case.start_pos, dtype=torch.int32)
    return (
        hidden_states,
        wk,
        gate_weight,
        ape,
        state_cache,
        cache_block_table,
        start_pos,
        seq_lens,
    )


def _updated_mask(state_cache, cache_block_table, start_pos, lengths):
    mask = torch.zeros(state_cache.shape[:2], dtype=torch.bool)
    block_size = state_cache.size(1)
    for batch, length in enumerate(lengths):
        for token in range(length):
            logical_pos = int(start_pos[batch]) + token
            logical_block, block_offset = divmod(logical_pos, block_size)
            physical_block = int(cache_block_table[batch, logical_block])
            mask[physical_block, block_offset] = True
    return mask


def _precision(name, actual, expected, rtol, atol, required_percent=99.5):
    actual_fp32 = actual.float()
    expected_fp32 = expected.float()
    close = torch.isclose(
        actual_fp32, expected_fp32, rtol=rtol, atol=atol, equal_nan=True
    )
    pass_percent = float(close.float().mean().item() * 100.0) if close.numel() else 100.0
    max_abs = (
        float((actual_fp32 - expected_fp32).abs().max().item())
        if actual_fp32.numel()
        else 0.0
    )
    return {
        "name": name,
        "pass": pass_percent >= required_percent,
        "pass_percent": pass_percent,
        "max_abs": max_abs,
        "rtol": rtol,
        "atol": atol,
    }


def run_case(case: Case):
    load_k_pool_public_api()
    torch_npu.npu.set_device(0)
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

    golden_cache = state_cache.clone()
    golden_output, _ = run_k_pool_golden(
        hidden_states,
        wk,
        gate_weight,
        ape,
        golden_cache,
        cache_block_table,
        start_pos,
        seq_lens=seq_lens,
        cmp_ratio=case.cmp_ratio,
    )

    npu_cache = state_cache.npu()
    npu_output = torch_npu.k_pool_npu(
        hidden_states.npu(),
        wk.npu(),
        gate_weight.npu(),
        ape.npu(),
        npu_cache,
        cache_block_table.npu(),
        start_pos.npu(),
        seq_lens=None if seq_lens is None else seq_lens.npu(),
        cmp_ratio=case.cmp_ratio,
    )
    torch_npu.npu.synchronize()
    actual_output = npu_output.cpu()
    actual_cache = npu_cache.cpu()

    mask = _updated_mask(
        state_cache, cache_block_table, start_pos, case.lengths
    )
    head_dim = wk.size(0)
    checks = [
        _precision(
            "pooled_key",
            actual_output,
            golden_output,
            rtol=0.0078125,
            atol=0.0001,
        ),
        _precision(
            "cache_k_updated",
            actual_cache[:, :, :head_dim][mask],
            golden_cache[:, :, :head_dim][mask],
            rtol=0.0078125,
            atol=0.0001,
        ),
        _precision(
            "cache_gate_updated",
            actual_cache[:, :, head_dim:][mask],
            golden_cache[:, :, head_dim:][mask],
            rtol=0.0078125,
            atol=0.0001,
        ),
        _precision(
            "cache_k_untouched",
            actual_cache[:, :, :head_dim][~mask],
            golden_cache[:, :, :head_dim][~mask],
            rtol=0.0,
            atol=0.0,
            required_percent=100.0,
        ),
        _precision(
            "cache_gate_untouched",
            actual_cache[:, :, head_dim:][~mask],
            golden_cache[:, :, head_dim:][~mask],
            rtol=0.0,
            atol=0.0,
            required_percent=100.0,
        ),
    ]

    tail_nonzero = 0
    for batch, length in enumerate(case.lengths):
        valid_pools = (case.start_pos[batch] + length) // case.cmp_ratio
        tail_nonzero += int(torch.count_nonzero(actual_output[batch, valid_pools:]))
    checks.append(
        {
            "name": "pooled_key_tail_zero",
            "pass": tail_nonzero == 0,
            "nonzero_count": tail_nonzero,
        }
    )
    return {
        "case": case.name,
        "layout": case.layout,
        "lengths": case.lengths,
        "start_pos": case.start_pos,
        "cmp_ratio": case.cmp_ratio,
        "block_size": case.block_size,
        "scramble_blocks": case.scramble_blocks,
        "pass": all(check["pass"] for check in checks),
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case", choices=(*CASES.keys(), "all"), default="bsh_minimal_r4"
    )
    args = parser.parse_args()
    selected = CASES.values() if args.case == "all" else (CASES[args.case],)
    results = [run_case(case) for case in selected]
    print(json.dumps(results, indent=2))
    if not all(result["pass"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
