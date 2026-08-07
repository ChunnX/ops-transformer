from __future__ import annotations

import argparse
import json

import torch
import torch_npu

from k_pool_public_loader import load_k_pool_public_api
from test_k_pool_public_npu import CASES, Case, _make_case_inputs


LAYOUT_CASES = (
    "th_full_history_r4",
    "th_zero_history_b4_r4",
    "th_zero_history_b4_r16",
)


def _run_npu(
    hidden_states,
    wk,
    gate_weight,
    ape,
    state_cache,
    cache_block_table,
    start_pos,
    seq_lens,
    cmp_ratio,
    norm_weight,
    norm_bias,
):
    npu_cache = state_cache.npu()
    output = torch_npu.k_pool_npu(
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
        cmp_ratio=cmp_ratio,
    )
    torch_npu.npu.synchronize()
    return output.cpu(), npu_cache.cpu()


def _segments(hidden_states, lengths):
    result = []
    offset = 0
    for length in lengths:
        result.append(hidden_states[offset : offset + length])
        offset += length
    return result


def _compare(name, actual, expected, bitwise=True, atol=0.0001):
    if not bitwise:
        close = torch.isclose(
            actual.float(),
            expected.float(),
            rtol=0.0078125,
            atol=atol,
            equal_nan=True,
        )
        passed = bool(close.all().item())
    else:
        passed = torch.equal(actual, expected)
    result = {"name": name, "pass": passed, "bitwise": bitwise}
    if not passed:
        diff = (actual.float() - expected.float()).abs()
        result["max_abs"] = float(diff.max().item()) if diff.numel() else 0.0
        result["mismatch_count"] = int(torch.count_nonzero(~close)) if not bitwise else int(
            torch.count_nonzero(actual != expected)
        )
    return result


def _run_case(case: Case, norm_on: bool):
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

    checks = []
    segments = _segments(hidden_states, case.lengths)
    for batch, segment in enumerate(segments):
        if segment.size(0) == 0:
            continue
        batch_output, batch_cache = _run_npu(
            segment.unsqueeze(0),
            wk,
            gate_weight,
            ape,
            state_cache.clone(),
            cache_block_table[batch : batch + 1],
            start_pos[batch : batch + 1],
            None,
            case.cmp_ratio,
            norm_weight,
            norm_bias,
        )
        checks.append(
            _compare(
                f"th_vs_bsh_output_batch_{batch}",
                full_output[batch],
                batch_output[0],
            )
        )
        physical_blocks = torch.unique(cache_block_table[batch])
        physical_blocks = physical_blocks[physical_blocks != 0].to(torch.int64)
        checks.append(
            _compare(
                f"th_vs_bsh_cache_batch_{batch}",
                full_cache[physical_blocks],
                batch_cache[physical_blocks],
                bitwise=False,
                atol=0.001,
            )
        )

    order = tuple(reversed(range(len(case.lengths))))
    perm_hidden = torch.cat([segments[index] for index in order], dim=0)
    perm_lengths = torch.tensor(
        [case.lengths[index] for index in order], dtype=torch.int32
    )
    perm_output, perm_cache = _run_npu(
        perm_hidden,
        wk,
        gate_weight,
        ape,
        state_cache.clone(),
        cache_block_table[list(order)],
        start_pos[list(order)],
        perm_lengths,
        case.cmp_ratio,
        norm_weight,
        norm_bias,
    )
    for perm_batch, original_batch in enumerate(order):
        checks.append(
            _compare(
                f"batch_position_output_{original_batch}",
                full_output[original_batch],
                perm_output[perm_batch],
            )
        )
    checks.append(_compare("batch_position_cache", full_cache, perm_cache))

    return {
        "case": case.name,
        "norm_on": norm_on,
        "pass": all(check["pass"] for check in checks),
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=(*LAYOUT_CASES, "all"),
        default="all",
    )
    parser.add_argument(
        "--norm",
        choices=("off", "on", "both"),
        default="both",
    )
    args = parser.parse_args()

    load_k_pool_public_api()
    torch_npu.npu.set_device(0)
    selected_cases = (
        LAYOUT_CASES if args.case == "all" else (args.case,)
    )
    norm_modes = (False, True) if args.norm == "both" else (args.norm == "on",)
    results = [
        _run_case(CASES[case_name], norm_on)
        for case_name in selected_cases
        for norm_on in norm_modes
    ]
    print(json.dumps(results, indent=2))
    if not all(result["pass"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
