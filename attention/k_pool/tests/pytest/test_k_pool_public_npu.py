from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

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
    head_dim: int = 128


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
        # KPOOL-037 isolation: vary only the fourth TH batch around a
        # zero-length batch with a nonzero historical start position.
        Case(
            "th_zero_history_b4_r4",
            "TH",
            (1, 1, 1, 0),
            (0, 4, 4, 4),
            4,
            block_size=1,
        ),
        Case(
            "th_zero_start0_b4_r4",
            "TH",
            (1, 1, 1, 0),
            (0, 4, 4, 0),
            4,
            block_size=1,
        ),
        Case(
            "th_nonzero_history_b4_r4",
            "TH",
            (1, 1, 1, 1),
            (0, 4, 4, 4),
            4,
            block_size=1,
        ),
        Case(
            "th_zero_history_b4_r16",
            "TH",
            (1, 1, 1, 0),
            (0, 16, 16, 16),
            16,
            block_size=128,
        ),
        Case("bsh_head_dim_512_r4", "BSH", (4,), (0,), 4, head_dim=512),
    )
}


def _parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split("|"))


def _load_manifest_cases(path: Path) -> list[Case]:
    cases = []
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["expected"] != "run" or row["norm"] != "off" or row["rope"] != "off":
                continue
            batch_size = int(row["B"])
            if row["layout"] == "BSH":
                lengths = (int(row["length"]),) * batch_size
            else:
                lengths = _parse_int_list(row["seq_lens"])
            start_pos = _parse_int_list(row["start_pos"])
            if len(lengths) != batch_size or len(start_pos) != batch_size:
                raise ValueError(f"invalid manifest batch shape: {row['case_id']}")
            cases.append(
                Case(
                    name=row["case_id"],
                    layout=row["layout"],
                    lengths=lengths,
                    start_pos=start_pos,
                    cmp_ratio=int(row["cmp_ratio"]),
                    block_size=int(row["block_size"]),
                    scramble_blocks=row["block_table"] != "ordered",
                )
            )
    if not cases:
        raise ValueError(f"no executable Norm-off/RoPE-off cases in manifest: {path}")
    return cases


def _limit_batch(case: Case, batch_size: int) -> Case:
    if batch_size <= 0 or batch_size > len(case.lengths):
        raise ValueError(f"invalid batch limit {batch_size} for {case.name}")
    return Case(
        name=f"{case.name}_b{batch_size}",
        layout=case.layout,
        lengths=case.lengths[:batch_size],
        start_pos=case.start_pos[:batch_size],
        cmp_ratio=case.cmp_ratio,
        block_size=case.block_size,
        scramble_blocks=case.scramble_blocks,
        head_dim=case.head_dim,
    )


def _make_case_inputs(case: Case):
    torch.manual_seed(20260807 + case.cmp_ratio + sum(case.lengths) + sum(case.start_pos))
    hidden_size = 4096
    head_dim = case.head_dim
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
    abs_diff = (actual_fp32 - expected_fp32).abs()
    close = torch.isclose(
        actual_fp32, expected_fp32, rtol=rtol, atol=atol, equal_nan=True
    )
    pass_percent = float(close.float().mean().item() * 100.0) if close.numel() else 100.0
    result = {
        "name": name,
        "pass": pass_percent >= required_percent,
        "pass_percent": pass_percent,
        "max_abs": float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
        "rtol": rtol,
        "atol": atol,
    }
    if not result["pass"] and abs_diff.numel():
        max_index = list(torch.unravel_index(abs_diff.argmax(), abs_diff.shape))
        result["max_index"] = [int(index.item()) for index in max_index]
        index_tuple = tuple(max_index)
        result["actual_at_max"] = float(actual_fp32[index_tuple].item())
        result["expected_at_max"] = float(expected_fp32[index_tuple].item())
    return result


def run_case(case: Case, per_batch: bool = False, norm_on: bool = False):
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
    norm_weight = None
    norm_bias = None
    if norm_on:
        norm_weight = torch.randn((wk.size(0),), dtype=torch.float32)
        norm_bias = torch.randn((wk.size(0),), dtype=torch.float32)

    golden_cache = state_cache.clone()
    golden_output, _ = run_k_pool_golden(
        hidden_states,
        wk,
        gate_weight,
        ape,
        golden_cache,
        cache_block_table,
        start_pos,
        norm_weight=norm_weight,
        norm_bias=norm_bias,
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
        norm_weight=None if norm_weight is None else norm_weight.npu(),
        norm_bias=None if norm_bias is None else norm_bias.npu(),
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
    result = {
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
    if per_batch:
        batch_checks = []
        for batch, length in enumerate(case.lengths):
            valid_pools = (case.start_pos[batch] + length) // case.cmp_ratio
            batch_check = _precision(
                f"pooled_key_batch_{batch}",
                actual_output[batch, :valid_pools],
                golden_output[batch, :valid_pools],
                rtol=0.0078125,
                atol=0.0001,
            )
            batch_check["valid_pools"] = valid_pools
            batch_checks.append(batch_check)
        result["pooled_key_by_batch"] = batch_checks
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case", choices=(*CASES.keys(), "all"), default="bsh_minimal_r4"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Run all Norm-off/RoPE-off executable cases from a CSV manifest.",
    )
    parser.add_argument(
        "--manifest-case",
        action="append",
        help="Select one or more case_id values from --manifest.",
    )
    parser.add_argument(
        "--per-batch",
        action="store_true",
        help="Include valid pooled_key precision results for each batch.",
    )
    parser.add_argument(
        "--limit-batch",
        type=int,
        help="Keep only the first N batches of each selected case for isolation.",
    )
    parser.add_argument(
        "--norm-on",
        action="store_true",
        help="Enable FP32 LayerNorm on the current K projection.",
    )
    args = parser.parse_args()
    if args.manifest is not None:
        selected = _load_manifest_cases(args.manifest)
        if args.manifest_case:
            selected_ids = set(args.manifest_case)
            selected = [case for case in selected if case.name in selected_ids]
            if not selected:
                raise ValueError("none of --manifest-case values are executable manifest cases")
    else:
        selected = CASES.values() if args.case == "all" else (CASES[args.case],)
    if args.limit_batch is not None:
        selected = [_limit_batch(case, args.limit_batch) for case in selected]
    results = [run_case(case, per_batch=args.per_batch, norm_on=args.norm_on) for case in selected]
    print(json.dumps(results, indent=2))
    if not all(result["pass"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
