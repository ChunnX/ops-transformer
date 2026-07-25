#!/usr/bin/env python3
"""General correctness and performance tests for ``0721.py``.

Every selected case runs both TileLang and the official torch_npu operator.
The default suite covers all documented shapes and K values.  All variable
floating-point inputs use BF16.  The operator contract still requires FP32
softmax statistics and INT32 sparse indices.

Examples:
  python test_0721_general.py
  python test_0721_general.py --suite smoke
  python test_0721_general.py --case bsnd_k4096
  python test_0721_general.py --no-bench
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from importlib import import_module
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import torch
import torch_npu


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TILELANG_DIR = os.path.dirname(SCRIPT_DIR)
if TILELANG_DIR not in sys.path:
    sys.path.insert(0, TILELANG_DIR)

MODULE = import_module("0721")
OP = MODULE.sparse_lightning_indexer_grad_kl_loss

REL_TOLERANCE = 2e-3
WARMUP_ITERS = 3
BENCH_ITERS = 10


def _case(name: str, *, suite: str = "official", **kwargs: Any) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "name": name,
        "suite": suite,
        "B": 1,
        "S1": 1,
        "S2": 2048,
        "N1": 128,
        "N2": 1,
        "Nidx1": 64,
        "Nidx2": 1,
        "DQuery": 512,
        "DQueryIndex": 128,
        "d_rope": 64,
        "K": 2048,
        "layout": "BSND",
    }
    config.update(kwargs)
    return config


# K=2048 shape/layout coverage cases.
OFFICIAL_CASES = [
    _case("bsnd_baseline"),
    _case("bsnd_batch2", B=2, S1=4),
    _case("bsnd_s2_unaligned", S1=8, S2=2305, N1=64, Nidx1=16),
    _case("bsnd_n1_64_nidx32", S1=8, N1=64, Nidx1=32),
    _case("bsnd_n1_32_nidx8", S1=16, N1=32, Nidx1=8),
    _case("tnd_s1_16", S1=16, layout="TND"),
]

# The interface document specifies K=1024..8192 in 1024 increments.  K=2048
# is already covered above; every remaining value is compared with official.
SPEC_CASES = [
    _case("bsnd_k1024", suite="spec", S1=8, S2=1537, N1=64, Nidx1=32, K=1024),
    _case("bsnd_k3072", suite="spec", S1=4, S2=3329, N1=32, Nidx1=8, K=3072),
    _case("bsnd_k4096", suite="spec", S1=4, S2=4096, K=4096),
    _case("bsnd_k5120", suite="spec", S1=2, S2=5120, N1=64, Nidx1=16, K=5120),
    _case("bsnd_k6144", suite="spec", S1=2, S2=6144, K=6144),
    _case("bsnd_k7168", suite="spec", S2=7168, N1=64, Nidx1=32, K=7168),
    _case("bsnd_k8192", suite="spec", S2=8192, K=8192),
]

SMOKE_CASE_NAMES = ("bsnd_baseline", "bsnd_s2_unaligned", "tnd_s1_16")
ALL_CASES = OFFICIAL_CASES + SPEC_CASES
CASE_BY_NAME = {case["name"]: case for case in ALL_CASES}


def _validate_case(case: Dict[str, Any]) -> None:
    if case["layout"] not in ("BSND", "TND"):
        raise ValueError(f"{case['name']}: layout must be BSND or TND")
    if case["layout"] == "TND" and case["B"] != 1:
        raise ValueError(f"{case['name']}: TND test cases require B=1")
    if case["N2"] != 1 or case["Nidx2"] != 1:
        raise ValueError(f"{case['name']}: this MQA operator requires N2=Nidx2=1")
    if case["N1"] not in (32, 64, 128):
        raise ValueError(f"{case['name']}: N1 must be 32, 64, or 128")
    if case["Nidx1"] not in (8, 16, 32, 64):
        raise ValueError(f"{case['name']}: Nidx1 must be 8, 16, 32, or 64")
    if case["DQuery"] != 512 or case["DQueryIndex"] != 128:
        raise ValueError(f"{case['name']}: 0721.py requires DQuery=512 and DQueryIndex=128")
    if case["d_rope"] != 64:
        raise ValueError(f"{case['name']}: the documented test coverage uses d_rope=64")
    if case["K"] < 1024 or case["K"] > 8192 or case["K"] % 1024 != 0:
        raise ValueError(f"{case['name']}: K must be in [1024, 8192] and divisible by 1024")
    if case["S1"] < 1 or case["S2"] < case["S1"]:
        raise ValueError(f"{case['name']}: require S2 >= S1 >= 1 for causal sparse indices")


def _make_sparse_indices(case: Dict[str, Any], seed: int) -> torch.Tensor:
    """Build right-down causal sparse indices on CPU, then transfer once."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1009)
    shape = (case["B"], case["S1"], case["Nidx2"], case["K"])
    sparse_indices = torch.full(shape, -1, dtype=torch.int32)
    for batch_id in range(case["B"]):
        for query_id in range(case["S1"]):
            visible_keys = case["S2"] - case["S1"] + query_id + 1
            chosen_count = min(visible_keys, case["K"])
            selected = torch.randperm(visible_keys, generator=generator)[:chosen_count]
            sparse_indices[batch_id, query_id, 0, :chosen_count] = selected.to(torch.int32)
    return sparse_indices.npu()


def _build_softmax_stats(
    query: torch.Tensor,
    key: torch.Tensor,
    query_rope: torch.Tensor,
    key_rope: torch.Tensor,
    scale_value: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate the forward-attention softmax max/sum inputs in FP32.

    The row loop caps temporary storage at [B, N1, S2] rather than materializing
    [B, S1, N1, S2], which matters for the long-sequence cases.
    """
    batch_size, sequence_q, head_count, _ = query.shape
    group_count = head_count
    key_full = torch.cat((key.squeeze(2), key_rope.squeeze(2)), dim=-1)
    key_transposed = key_full.transpose(-1, -2).contiguous()
    query_full = torch.cat((query, query_rope), dim=-1)
    softmax_max = torch.empty(
        (batch_size, 1, sequence_q, group_count), dtype=torch.float32, device=query.device
    )
    softmax_sum = torch.empty(
        softmax_max.shape, dtype=softmax_max.dtype, device=softmax_max.device
    )

    with torch.no_grad():
        for query_id in range(sequence_q):
            scores = torch.matmul(query_full[:, query_id], key_transposed).float()
            row_max = scores.mul(scale_value).amax(dim=-1)
            row_sum = torch.exp(scores.mul(scale_value) - row_max.unsqueeze(-1)).sum(dim=-1)
            softmax_max[:, 0, query_id, :] = row_max
            softmax_sum[:, 0, query_id, :] = row_sum
    return softmax_max, softmax_sum


def make_inputs(case: Dict[str, Any], seed: int) -> Dict[str, Any]:
    _validate_case(case)
    input_dtype = torch.bfloat16
    torch.manual_seed(seed)

    bsnd_shape = (case["B"], case["S1"], case["N1"], case["DQuery"])
    key_shape = (case["B"], case["S2"], case["N2"], case["DQuery"])
    query_index_shape = (case["B"], case["S1"], case["Nidx1"], case["DQueryIndex"])
    key_index_shape = (case["B"], case["S2"], case["Nidx2"], case["DQueryIndex"])
    query_rope_shape = (case["B"], case["S1"], case["N1"], case["d_rope"])
    key_rope_shape = (case["B"], case["S2"], case["N2"], case["d_rope"])
    weight_shape = (case["B"], case["S1"], case["Nidx1"])

    query_bsnd = torch.randn(bsnd_shape, dtype=input_dtype).npu()
    key_bsnd = torch.randn(key_shape, dtype=input_dtype).npu()
    query_index_bsnd = torch.randn(query_index_shape, dtype=input_dtype).npu()
    key_index_bsnd = torch.randn(key_index_shape, dtype=input_dtype).npu()
    query_rope_bsnd = torch.randn(query_rope_shape, dtype=input_dtype).npu()
    key_rope_bsnd = torch.randn(key_rope_shape, dtype=input_dtype).npu()
    weights_bsnd = torch.randn(weight_shape, dtype=input_dtype).npu()
    sparse_indices_bsnd = _make_sparse_indices(case, seed)
    scale_value = 1.0 / math.sqrt(case["DQuery"])
    softmax_max_bsnd, softmax_sum_bsnd = _build_softmax_stats(
        query_bsnd, key_bsnd, query_rope_bsnd, key_rope_bsnd, scale_value
    )

    if case["layout"] == "TND":
        return {
            "query": query_bsnd.squeeze(0),
            "key": key_bsnd.squeeze(0),
            "query_index": query_index_bsnd.squeeze(0),
            "key_index": key_index_bsnd.squeeze(0),
            "weights": weights_bsnd.squeeze(0),
            "sparse_indices": sparse_indices_bsnd.squeeze(0),
            "softmax_max": softmax_max_bsnd.squeeze(0),
            "softmax_sum": softmax_sum_bsnd.squeeze(0),
            "query_rope": query_rope_bsnd.squeeze(0),
            "key_rope": key_rope_bsnd.squeeze(0),
            "scale_value": scale_value,
        }

    return {
        "query": query_bsnd,
        "key": key_bsnd,
        "query_index": query_index_bsnd,
        "key_index": key_index_bsnd,
        "weights": weights_bsnd,
        "sparse_indices": sparse_indices_bsnd,
        "softmax_max": softmax_max_bsnd,
        "softmax_sum": softmax_sum_bsnd,
        "query_rope": query_rope_bsnd,
        "key_rope": key_rope_bsnd,
        "scale_value": scale_value,
    }


def _call_tilelang(inputs: Dict[str, Any], case: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
    lengths_q = [case["S1"]] if case["layout"] == "TND" else [case["S1"]] * case["B"]
    lengths_k = [case["S2"]] if case["layout"] == "TND" else [case["S2"]] * case["B"]
    return OP(
        inputs["query"], inputs["key"], inputs["query_index"], inputs["key_index"],
        inputs["weights"], inputs["sparse_indices"], inputs["softmax_max"], inputs["softmax_sum"],
        inputs["scale_value"], query_rope=inputs["query_rope"], key_rope=inputs["key_rope"],
        actual_seq_lengths_query=lengths_q, actual_seq_lengths_key=lengths_k,
        layout_query=case["layout"], sparse_mode=3,
    )


def _call_official(inputs: Dict[str, Any], case: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
    lengths_q = [case["S1"]] if case["layout"] == "TND" else [case["S1"]] * case["B"]
    lengths_k = [case["S2"]] if case["layout"] == "TND" else [case["S2"]] * case["B"]
    return torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
        inputs["query"], inputs["key"], inputs["query_index"], inputs["key_index"],
        inputs["weights"], inputs["sparse_indices"], inputs["softmax_max"], inputs["softmax_sum"],
        inputs["scale_value"], query_rope=inputs["query_rope"], key_rope=inputs["key_rope"],
        actual_seq_qlen=lengths_q, actual_seq_klen=lengths_k,
        layout=case["layout"], sparse_mode=3, pre_tokens=65536, next_tokens=65536,
    )


def _bench(fn: Callable[[], Any], warmup: int, iterations: int) -> Tuple[float, float, float]:
    for _ in range(warmup):
        fn()
        torch.npu.synchronize()
    timings: List[float] = []
    for _ in range(iterations):
        begin = time.perf_counter()
        fn()
        torch.npu.synchronize()
        timings.append(time.perf_counter() - begin)
    timings.sort()
    return sum(timings) / len(timings), timings[len(timings) // 2], timings[0]


def _compare_tensor(name: str, actual: torch.Tensor, expected: torch.Tensor) -> Dict[str, float]:
    if actual.shape != expected.shape:
        raise AssertionError(f"{name}: shape mismatch: TileLang={tuple(actual.shape)}, official={tuple(expected.shape)}")
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    actual_finite = torch.isfinite(actual_f32)
    expected_finite = torch.isfinite(expected_f32)
    if not actual_finite.all().item() or not expected_finite.all().item():
        raise AssertionError(
            f"{name}: non-finite value detected "
            f"(TileLang={actual_f32.numel() - actual_finite.sum().item()}, "
            f"Official={expected_f32.numel() - expected_finite.sum().item()})"
        )
    difference = (actual_f32 - expected_f32).abs()
    return {
        "max_abs": difference.max().item(),
        "mean_abs": difference.mean().item(),
        "mean_rel": (difference / (expected_f32.abs() + 1e-6)).mean().item(),
    }


def _compare_outputs(
    tilelang_outputs: Sequence[torch.Tensor], official_outputs: Sequence[torch.Tensor]
) -> Tuple[Dict[str, Dict[str, float]], bool]:
    metrics: Dict[str, Dict[str, float]] = {}
    output_names = ("dQueryIndex", "dKeyIndex", "dWeights")
    for name, actual, expected in zip(output_names, tilelang_outputs[:3], official_outputs[:3]):
        metrics[name] = _compare_tensor(name, actual, expected)
    loss_actual = tilelang_outputs[3].float().reshape(-1)
    loss_expected = official_outputs[3].float().reshape(-1)
    if loss_actual.numel() != 1 or loss_expected.numel() != 1:
        raise AssertionError("Loss must contain exactly one scalar")
    if not torch.isfinite(loss_actual).all().item() or not torch.isfinite(loss_expected).all().item():
        raise AssertionError("Loss: non-finite value detected")
    loss_abs = (loss_actual - loss_expected).abs().item()
    metrics["Loss"] = {
        "max_abs": loss_abs,
        "mean_abs": loss_abs,
        "mean_rel": loss_abs / (loss_expected.abs().item() + 1e-6),
    }
    passed = all(value["mean_rel"] <= REL_TOLERANCE for value in metrics.values())
    return metrics, passed


def _print_metrics(metrics: Dict[str, Dict[str, float]]) -> None:
    for name, values in metrics.items():
        print(
            f"  {name:<12} max_abs={values['max_abs']:.6e} "
            f"mean_abs={values['mean_abs']:.6e} mean_rel={values['mean_rel']:.6e}"
        )


def _case_summary(case: Dict[str, Any]) -> str:
    return (
        f"layout={case['layout']} B={case['B']} S1={case['S1']} S2={case['S2']} "
        f"N1={case['N1']} Nidx1={case['Nidx1']} K={case['K']} "
        "dtype=bfloat16"
    )


def run_case(case: Dict[str, Any], args: argparse.Namespace, index: int) -> str:
    print(f"\n{'=' * 88}\n[{index}] {case['name']}: {_case_summary(case)}")
    MODULE._clear_caches()
    torch.npu.synchronize()

    try:
        inputs = make_inputs(case, args.seed + index)
    except Exception as exc:
        print(f"  INPUT FAIL: {exc}")
        return "FAIL"

    try:
        tilelang_outputs = _call_tilelang(inputs, case)
        torch.npu.synchronize()
        print("  TileLang: completed")
    except Exception as exc:
        print(f"  TileLang FAIL: {exc}")
        return "FAIL"

    try:
        official_outputs = _call_official(inputs, case)
        torch.npu.synchronize()
        print("  Official: completed")
    except Exception as exc:
        print(f"  Official FAIL: {exc}")
        return "FAIL"

    try:
        metrics, passed = _compare_outputs(tilelang_outputs, official_outputs)
        _print_metrics(metrics)
        status = "PASS" if passed else "FAIL"
        print(f"  Precision: {status} (mean relative tolerance={REL_TOLERANCE:.1e})")
    except Exception as exc:
        print(f"  PRECISION FAIL: {exc}")
        status = "FAIL"

    if args.bench:
        try:
            tl_avg, tl_med, tl_min = _bench(
                lambda: _call_tilelang(inputs, case), args.warmup, args.iters
            )
            off_avg, off_med, off_min = _bench(
                lambda: _call_official(inputs, case), args.warmup, args.iters
            )
            print(
                f"  TileLang: avg={tl_avg * 1e3:.3f}ms med={tl_med * 1e3:.3f}ms "
                f"min={tl_min * 1e3:.3f}ms"
            )
            print(
                f"  Official: avg={off_avg * 1e3:.3f}ms med={off_med * 1e3:.3f}ms "
                f"min={off_min * 1e3:.3f}ms"
            )
            print(f"  Ratio (official/tilelang median): {off_med / tl_med:.2f}x")
        except Exception as exc:
            print(f"  BENCHMARK FAIL: {exc}")
            status = "FAIL"

    return status


def _select_cases(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    if args.case:
        unknown = [name for name in args.case if name not in CASE_BY_NAME]
        if unknown:
            raise ValueError(f"Unknown case(s): {', '.join(unknown)}")
        return [CASE_BY_NAME[name] for name in args.case]
    if args.suite == "smoke":
        return [CASE_BY_NAME[name] for name in SMOKE_CASE_NAMES]
    if args.suite == "official":
        return OFFICIAL_CASES
    if args.suite == "spec":
        return SPEC_CASES
    return ALL_CASES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("smoke", "official", "spec", "all"), default="all")
    parser.add_argument("--case", action="append", help="Run exactly this named case; can be repeated")
    parser.add_argument(
        "--bench",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Benchmark both implementations after comparison (default: enabled)",
    )
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERS)
    parser.add_argument("--iters", type=int, default=BENCH_ITERS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.warmup < 0 or args.iters < 1:
        parser.error("--warmup must be non-negative and --iters must be at least one")
    return args


def main() -> int:
    args = parse_args()
    if os.environ.get("SLIG_DEBUG", "0") == "1" or os.environ.get("SLIG_PROFILE", "0") == "1":
        print("WARNING: SLIG_DEBUG/SLIG_PROFILE is enabled; disable it for benchmark numbers.")
    selected_cases = list(_select_cases(args))
    print(f"0721 general test: suite={args.suite}, cases={len(selected_cases)}, benchmark={args.bench}")
    print(
        "Supported test contract: BF16 variable inputs/weights, FP32 softmax stats, INT32 sparse indices, "
        "sparse_mode=3, N2=Nidx2=1, DQuery=512, DQueryIndex=128."
    )

    results: List[Tuple[str, str]] = []
    for index, case in enumerate(selected_cases):
        results.append((case["name"], run_case(case, args, index)))

    print(f"\n{'=' * 88}\nSummary")
    for name, status in results:
        print(f"  {status:<10} {name}")
    failures = [name for name, status in results if status == "FAIL"]
    print(f"  failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
