#!/usr/bin/env python3
"""Compare 0810, the frozen baseline, and the official operator on one case.

The fixed case is BSND: B=2, S1=2, S2=2048, K=2048, N1=128, Nidx1=64.
The input is created once and reused by every implementation.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from typing import Any, Callable, Dict, Sequence, Tuple

import torch
import torch_npu


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TILELANG_DIR = os.path.dirname(SCRIPT_DIR)
if TILELANG_DIR not in sys.path:
    sys.path.insert(0, TILELANG_DIR)

# Reuse the established input construction, precision metric, and timing rules.
from test_0721_general import (  # noqa: E402
    _bench,
    _call_official,
    _compare_outputs,
    _print_metrics,
    make_inputs,
)


CASE: Dict[str, Any] = {
    "name": "compare_b2_s1_1_k_2048",
    "suite": "compare",
    "kind": "full-valid",
    "layout": "BSND",
    "B": 2,
    "S1": 2,
    "S2": 2048,
    "N1": 128,
    "N2": 1,
    "Nidx1": 64,
    "Nidx2": 1,
    "DQuery": 512,
    "DQueryIndex": 128,
    "d_rope": 64,
    "K": 2048,
}


def _call_tilelang(module: Any, inputs: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
    lengths_q = [CASE["S1"]] * CASE["B"]
    lengths_k = [CASE["S2"]] * CASE["B"]
    return module.sparse_lightning_indexer_grad_kl_loss(
        inputs["query"],
        inputs["key"],
        inputs["query_index"],
        inputs["key_index"],
        inputs["weights"],
        inputs["sparse_indices"],
        inputs["softmax_max"],
        inputs["softmax_sum"],
        inputs["scale_value"],
        query_rope=inputs["query_rope"],
        key_rope=inputs["key_rope"],
        actual_seq_lengths_query=lengths_q,
        actual_seq_lengths_key=lengths_k,
        layout_query=CASE["layout"],
        sparse_mode=3,
    )


def _brief_case() -> str:
    return (
        f"layout={CASE['layout']} B={CASE['B']} S1={CASE['S1']} S2={CASE['S2']} "
        f"N1={CASE['N1']} Nidx1={CASE['Nidx1']} K={CASE['K']}"
    )


def _precision_check(
    label: str,
    outputs: Sequence[torch.Tensor],
    official_outputs: Sequence[torch.Tensor],
) -> bool:
    metrics, passed = _compare_outputs(outputs, official_outputs, CASE)
    print(f"[{label}] precision={'PASS' if passed else 'FAIL'}")
    _print_metrics(metrics)
    return passed


def _run_call(label: str, fn: Callable[[], Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
    print(f"[{label}] execute")
    torch.npu.synchronize()
    outputs = fn()
    torch.npu.synchronize()
    print(f"[{label}] completed")
    return outputs


def _benchmark(
    label: str,
    fn: Callable[[], Tuple[torch.Tensor, ...]],
    warmup: int,
    iters: int,
) -> Tuple[float, float, float]:
    avg, med, minimum = _bench(fn, warmup, iters)
    print(
        f"[{label}] bench avg={avg * 1e3:.3f}ms "
        f"med={med * 1e3:.3f}ms min={minimum * 1e3:.3f}ms"
    )
    return avg, med, minimum


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--profile-target",
        choices=("all", "0810", "0721_baseline", "official"),
        default="all",
        help="Which implementation is timed; use one target for msprof.",
    )
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="Skip precision calls and time only profile-target.",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.iters < 1:
        parser.error("--warmup must be non-negative and --iters must be at least one")
    if args.profile_only and args.profile_target == "all":
        parser.error("--profile-only requires --profile-target 0810, 0721_baseline, or official")
    return args


def main() -> int:
    args = _parse_args()
    baseline = importlib.import_module("0721_baseline")
    candidate = importlib.import_module("0810")
    inputs = make_inputs(CASE, args.seed)
    official_fn = lambda: _call_official(inputs, CASE)
    baseline_fn = lambda: _call_tilelang(baseline, inputs)
    candidate_fn = lambda: _call_tilelang(candidate, inputs)
    fns: Dict[str, Callable[[], Tuple[torch.Tensor, ...]]] = {
        "0810": candidate_fn,
        "0721_baseline": baseline_fn,
        "official": official_fn,
    }

    print(f"compare case: {_brief_case()} seed={args.seed}")
    if not args.profile_only:
        official_outputs = _run_call("official", official_fn)
        baseline_outputs = _run_call("0721_baseline", baseline_fn)
        candidate_outputs = _run_call("0810", candidate_fn)
        baseline_passed = _precision_check("0721_baseline", baseline_outputs, official_outputs)
        candidate_passed = _precision_check("0810", candidate_outputs, official_outputs)
        if not baseline_passed or not candidate_passed:
            print("precision summary: FAIL")
        else:
            print("precision summary: PASS")

    if args.profile_only:
        _benchmark(args.profile_target, fns[args.profile_target], args.warmup, args.iters)
        return 0

    labels = ("0810", "0721_baseline", "official")
    if args.profile_target != "all":
        labels = (args.profile_target,)
    timings = {
        label: _benchmark(label, fns[label], args.warmup, args.iters)[1]
        for label in labels
    }
    if "official" in timings:
        official_med = timings["official"]
        for label in ("0810", "0721_baseline"):
            if label in timings:
                print(f"[{label}] speed_ratio_vs_official={official_med / timings[label]:.3f}x")
    if "0810" in timings and "0721_baseline" in timings:
        print(f"[0810] speed_ratio_vs_baseline={timings['0721_baseline'] / timings['0810']:.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
