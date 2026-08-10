from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu

from k_pool_public_loader import load_k_pool_public_api
from test_k_pool_public_npu import _load_manifest_cases, _make_case_inputs


def _to_npu_inputs(inputs):
    hidden_states, wk, gate_weight, ape, state_cache, block_table, start_pos, seq_lens = inputs
    return (
        hidden_states.npu(),
        wk.npu(),
        gate_weight.npu(),
        ape.npu(),
        state_cache.npu(),
        block_table.npu(),
        start_pos.npu(),
        None if seq_lens is None else seq_lens.npu(),
    )


def _measure_fused(inputs, case, warmup: int, repeat: int):
    hidden_states, wk, gate_weight, ape, state_cache, block_table, start_pos, seq_lens = _to_npu_inputs(inputs)

    print(f"[{case.name}] fused warmup", flush=True)
    for _ in range(warmup):
        cache = state_cache.clone()
        torch_npu.k_pool_npu(
            hidden_states,
            wk,
            gate_weight,
            ape,
            cache,
            block_table,
            start_pos,
            seq_lens=seq_lens,
            cmp_ratio=case.cmp_ratio,
        )
    torch.npu.synchronize()

    elapsed = []
    output = None
    print(f"[{case.name}] fused measure", flush=True)
    for _ in range(repeat):
        cache = state_cache.clone()
        start = time.perf_counter()
        output = torch_npu.k_pool_npu(
            hidden_states,
            wk,
            gate_weight,
            ape,
            cache,
            block_table,
            start_pos,
            seq_lens=seq_lens,
            cmp_ratio=case.cmp_ratio,
        )
        torch.npu.synchronize()
        elapsed.append((time.perf_counter() - start) * 1e6)
    return elapsed, output


def _make_token_indices(case, device):
    batch_ids = []
    token_offsets = []
    for batch, length in enumerate(case.lengths):
        batch_ids.extend([batch] * length)
        token_offsets.extend(range(length))
    return (
        torch.tensor(batch_ids, dtype=torch.long, device=device),
        torch.tensor(token_offsets, dtype=torch.long, device=device),
    )


def _run_torch_decomposition(npu_inputs, case, batch_ids, token_offsets):
    hidden_states, wk, gate_weight, ape, state_cache, block_table, start_pos, _ = npu_inputs
    head_dim = wk.size(0)
    block_size = state_cache.size(1)
    hidden_2d = hidden_states.reshape(-1, hidden_states.size(-1))
    hidden_fp32 = hidden_2d.float()
    k = torch.matmul(hidden_fp32, wk.float().transpose(0, 1)).to(torch.bfloat16)
    gate = torch.matmul(hidden_fp32, gate_weight.float().transpose(0, 1)).to(torch.bfloat16)

    logical_write = start_pos.long()[batch_ids] + token_offsets
    physical_write = block_table.long()[batch_ids, logical_write // block_size]
    state_cache[physical_write, logical_write % block_size] = torch.cat((k, gate), dim=-1).float()

    logical_capacity = block_table.size(1) * block_size
    pool_capacity = (logical_capacity + case.cmp_ratio - 1) // case.cmp_ratio
    logical = torch.arange(
        pool_capacity * case.cmp_ratio,
        dtype=torch.long,
        device=hidden_states.device,
    )
    safe_logical = logical.clamp(max=logical_capacity - 1)
    physical = block_table.long()[:, safe_logical // block_size]
    gathered = state_cache[physical, safe_logical % block_size]
    gathered = gathered.view(len(case.lengths), pool_capacity, case.cmp_ratio, -1)
    keys = gathered[..., :head_dim]
    gates = gathered[..., head_dim:]

    updated_lens = start_pos.long() + torch.tensor(
        case.lengths, dtype=torch.long, device=hidden_states.device
    )
    valid = logical.view(1, pool_capacity, case.cmp_ratio) < updated_lens[:, None, None]
    logits = gates + ape.view(1, 1, case.cmp_ratio, head_dim)
    logits = logits.masked_fill(~valid[..., None], float("-inf"))
    probabilities = torch.nan_to_num(torch.softmax(logits, dim=2)).to(torch.bfloat16).float()
    pooled = (probabilities * keys).sum(dim=2).to(torch.bfloat16)
    return pooled.masked_fill(~valid.all(dim=2)[..., None], 0)


def _measure_oracle(inputs, case, warmup: int, repeat: int):
    npu_inputs = _to_npu_inputs(inputs)
    batch_ids, token_offsets = _make_token_indices(case, npu_inputs[0].device)

    def run_once():
        run_inputs = (*npu_inputs[:4], npu_inputs[4].clone(), *npu_inputs[5:])
        return _run_torch_decomposition(run_inputs, case, batch_ids, token_offsets)

    print(f"[{case.name}] torch warmup", flush=True)
    for _ in range(warmup):
        run_once()
    torch.npu.synchronize()

    elapsed = []
    output = None
    print(f"[{case.name}] torch measure", flush=True)
    for _ in range(repeat):
        start = time.perf_counter()
        output = run_once()
        torch.npu.synchronize()
        elapsed.append((time.perf_counter() - start) * 1e6)
    return elapsed, output


def _result(case, fused, oracle):
    fused_median = statistics.median(fused[0])
    oracle_median = statistics.median(oracle[0])
    speedup = oracle_median / fused_median if fused_median else 0.0
    diff = (fused[1].float() - oracle[1].float()).abs()
    return {
        "case": case.name,
        "layout": case.layout,
        "lengths": case.lengths,
        "start_pos": case.start_pos,
        "cmp_ratio": case.cmp_ratio,
        "block_size": case.block_size,
        "fused_us": fused[0],
        "oracle_us": oracle[0],
        "fused_median_us": fused_median,
        "oracle_median_us": oracle_median,
        "speedup": speedup,
        "max_abs_fused_vs_oracle": float(diff.max().cpu().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case", action="append")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("warmup must be non-negative and repeat must be positive")

    load_k_pool_public_api()
    cases = _load_manifest_cases(args.manifest)
    if args.case:
        selected = set(args.case)
        cases = [case for case in cases if case.name in selected]
    if not cases:
        raise ValueError("no performance cases selected")

    results = []
    for case in cases:
        inputs = _make_case_inputs(case)
        fused = _measure_fused(inputs, case, args.warmup, args.repeat)
        oracle = _measure_oracle(inputs, case, args.warmup, args.repeat)
        result = _result(case, fused, oracle)
        results.append(result)
        print(json.dumps(result, sort_keys=True))

    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
