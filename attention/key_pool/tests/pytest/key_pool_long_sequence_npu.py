from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import torch
import torch_npu

from key_pool_public_loader import load_key_pool_public_api
from test_key_pool_public_npu import (
    _load_manifest_cases,
    _make_case_inputs,
    _precision,
    _updated_mask,
)


def _token_indices(lengths: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    batch_ids = torch.repeat_interleave(
        torch.arange(len(lengths), dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
    )
    token_offsets = torch.cat(
        [torch.arange(length, dtype=torch.long) for length in lengths]
    )
    return batch_ids, token_offsets


def _cpu_golden(inputs, case, norm_on: bool, norm_eps: float):
    hidden_states, wk, gate_weight, ape, state_cache, block_table, start_pos, _ = inputs
    head_dim = wk.size(0)
    block_size = state_cache.size(1)
    hidden_2d = hidden_states.reshape(-1, hidden_states.size(-1))
    hidden_fp32 = hidden_2d.float()
    k = torch.matmul(hidden_fp32, wk.float().transpose(0, 1)).to(torch.bfloat16)
    gate = torch.matmul(
        hidden_fp32, gate_weight.float().transpose(0, 1)
    ).to(torch.bfloat16)
    norm_weight = norm_bias = None
    if norm_on:
        norm_weight = torch.randn((head_dim,), dtype=torch.float32)
        norm_bias = torch.randn((head_dim,), dtype=torch.float32)
        k_fp32 = k.float()
        mean = k_fp32.mean(dim=-1, keepdim=True)
        variance = (k_fp32 - mean).square().mean(dim=-1, keepdim=True)
        k = (
            (k_fp32 - mean)
            / torch.sqrt(variance + norm_eps)
            * norm_weight
            + norm_bias
        )

    pool_cache = state_cache.clone()
    golden_cache = state_cache.clone()
    batch_ids, token_offsets = _token_indices(case.lengths)
    logical_write = start_pos.long()[batch_ids] + token_offsets
    physical_write = block_table.long()[
        batch_ids, logical_write // block_size
    ]
    block_offsets = logical_write % block_size
    pool_cache[physical_write, block_offsets, :head_dim] = k.float()
    pool_cache[physical_write, block_offsets, head_dim:] = gate.float()
    tail_mask = torch.zeros_like(logical_write, dtype=torch.bool)
    token_base = 0
    for batch, length in enumerate(case.lengths):
        end_pos = int(start_pos[batch]) + length
        tail_len = end_pos % case.cmp_ratio
        tail_start = max(int(start_pos[batch]), end_pos - tail_len)
        tail_mask[token_base : token_base + length] = (
            logical_write[token_base : token_base + length] >= tail_start
        )
        token_base += length
    golden_cache[
        physical_write[tail_mask], block_offsets[tail_mask], :head_dim
    ] = k[tail_mask].float()
    golden_cache[
        physical_write[tail_mask], block_offsets[tail_mask], head_dim:
    ] = gate[tail_mask].float()

    logical_capacity = block_table.size(1) * block_size
    pool_capacity = (
        logical_capacity + case.cmp_ratio - 1
    ) // case.cmp_ratio
    golden_output = torch.zeros(
        (len(case.lengths), pool_capacity, head_dim), dtype=torch.bfloat16
    )
    for batch, length in enumerate(case.lengths):
        updated_len = int(start_pos[batch]) + length
        valid_pools = updated_len // case.cmp_ratio - int(start_pos[batch]) // case.cmp_ratio
        if valid_pools == 0:
            continue
        first_pool = int(start_pos[batch]) // case.cmp_ratio
        logical = torch.arange(
            first_pool * case.cmp_ratio,
            (first_pool + valid_pools) * case.cmp_ratio,
            dtype=torch.long,
        ).view(valid_pools, case.cmp_ratio)
        physical = block_table.long()[batch, logical // block_size]
        gathered = pool_cache[physical, logical % block_size]
        keys = gathered[..., :head_dim]
        gates = gathered[..., head_dim:]
        probabilities = torch.softmax(
            gates + ape.view(1, case.cmp_ratio, head_dim), dim=1
        ).to(torch.bfloat16).float()
        golden_output[batch, :valid_pools] = (
            probabilities * keys
        ).sum(dim=1).to(torch.bfloat16)

    return golden_output, golden_cache, norm_weight, norm_bias


def run_case(case, device: int, norm_on: bool, norm_eps: float):
    torch.manual_seed(
        20260807 + case.cmp_ratio + sum(case.lengths) + sum(case.start_pos)
    )
    inputs = _make_case_inputs(case)
    (
        hidden_states,
        wk,
        gate_weight,
        ape,
        state_cache,
        block_table,
        start_pos,
        cu_seqlens,
    ) = inputs
    golden_output, golden_cache, norm_weight, norm_bias = _cpu_golden(
        inputs, case, norm_on, norm_eps
    )

    torch_npu.npu.set_device(device)
    npu_cache = state_cache.npu()
    npu_output = torch_npu.key_pool(
        hidden_states.npu(),
        wk.npu(),
        gate_weight.npu(),
        ape.npu(),
        npu_cache,
        block_table.npu(),
        start_pos.npu(),
        norm_weight=None if norm_weight is None else norm_weight.npu(),
        norm_bias=None if norm_bias is None else norm_bias.npu(),
        cu_seqlens=None if cu_seqlens is None else cu_seqlens.npu(),
        cmp_ratio=case.cmp_ratio,
        norm_eps=norm_eps,
    )
    torch_npu.npu.synchronize()
    actual_output = npu_output.cpu()
    actual_cache = npu_cache.cpu()

    mask = _updated_mask(
        state_cache, block_table, start_pos, case.lengths, case.cmp_ratio
    )
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
            actual_cache[:, :, : case.head_dim][mask],
            golden_cache[:, :, : case.head_dim][mask],
            rtol=0.0078125,
            atol=0.0001,
        ),
        _precision(
            "cache_gate_updated",
            actual_cache[:, :, case.head_dim :][mask],
            golden_cache[:, :, case.head_dim :][mask],
            rtol=0.0078125,
            atol=0.0001,
        ),
        _precision(
            "cache_k_untouched",
            actual_cache[:, :, : case.head_dim][~mask],
            golden_cache[:, :, : case.head_dim][~mask],
            rtol=0.0,
            atol=0.0,
            required_percent=100.0,
        ),
        _precision(
            "cache_gate_untouched",
            actual_cache[:, :, case.head_dim :][~mask],
            golden_cache[:, :, case.head_dim :][~mask],
            rtol=0.0,
            atol=0.0,
            required_percent=100.0,
        ),
    ]

    tail_nonzero = 0
    batch_checks = []
    for batch, length in enumerate(case.lengths):
        valid_pools = (
            (case.start_pos[batch] + length) // case.cmp_ratio
            - case.start_pos[batch] // case.cmp_ratio
        )
        tail_nonzero += int(
            torch.count_nonzero(actual_output[batch, valid_pools:])
        )
        batch_check = _precision(
            f"pooled_key_batch_{batch}",
            actual_output[batch, :valid_pools],
            golden_output[batch, :valid_pools],
            rtol=0.0078125,
            atol=0.0001,
        )
        batch_check["valid_pools"] = valid_pools
        batch_checks.append(batch_check)
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
        "total_current_tokens": sum(case.lengths),
        "start_pos": case.start_pos,
        "cmp_ratio": case.cmp_ratio,
        "block_size": case.block_size,
        "scramble_blocks": case.scramble_blocks,
        "norm": "on" if norm_on else "off",
        "pass": all(check["pass"] for check in checks)
        and all(check["pass"] for check in batch_checks),
        "checks": checks,
        "pooled_key_by_batch": batch_checks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-case", action="append")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--norm-on", action="store_true")
    parser.add_argument("--norm-eps", type=float, default=1e-6)
    args = parser.parse_args()

    load_key_pool_public_api()
    device = (
        args.device
        if args.device is not None
        else int(os.environ.get("ASCEND_DEVICE_ID", "0"))
    )
    selected = _load_manifest_cases(args.manifest)
    if args.manifest_case:
        selected_ids = set(args.manifest_case)
        selected = [case for case in selected if case.name in selected_ids]
    if not selected:
        raise ValueError("no executable long-sequence cases selected")

    results = []
    for index, case in enumerate(selected, start=1):
        result = run_case(case, device, args.norm_on, args.norm_eps)
        results.append(result)
        print(
            json.dumps(
                {
                    "progress": f"{index}/{len(selected)}",
                    "case": result["case"],
                    "norm": result["norm"],
                    "pass": result["pass"],
                    "max_pooled_key_abs": result["checks"][0]["max_abs"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        gc.collect()
        torch_npu.npu.empty_cache()

    summary = {
        "device": device,
        "norm": "on" if args.norm_on else "off",
        "case_count": len(results),
        "passed": sum(result["pass"] for result in results),
        "all_passed": all(result["pass"] for result in results),
        "results": results,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
