#!/usr/bin/env python3
"""Contract-oriented correctness and performance tests for ``0721.py``.

The complete suite contains 45 BSND cases and five TND cases.  It covers the
documented B/S1/K values without constructing infeasible full-cartesian
workspaces.  Every selected case runs both TileLang and the official operator.
All variable floating-point inputs use BF16; softmax statistics are FP32 and
sparse indices are INT32.

The current TileLang TND implementation accepts one flattened sequence only,
so the five TND cases intentionally use B=1.  Multi-batch ragged TND is an
operator capability gap rather than a test-input gap.

Examples:
  python test_0721_general.py
  python test_0721_general.py --suite smoke
  python test_0721_general.py --suite core --no-bench
  python test_0721_general.py --suite stress --no-bench
  python test_0721_general.py --case bsnd_b1_s1_64_k_4096
  python test_0721_general.py --case tnd_s1_512_k_4096 --no-bench
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time
from collections import Counter
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


def _case(name: str, *, suite: str = "core", **kwargs: Any) -> Dict[str, Any]:
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
        "kind": "coverage",
    }
    config.update(kwargs)
    return config


S1_VALUES = (1, 64, 512, 1024, 2048)
K_VALUES = (1024, 2048, 4096, 5120, 8192)
HEAD_CONFIGS = (
    {"N1": 128, "Nidx1": 64},
    {"N1": 64, "Nidx1": 32},
    {"N1": 32, "Nidx1": 16},
    {"N1": 32, "Nidx1": 8},
)

# A full B/S1/K cartesian product contains 100 cases and includes multi-TiB
# workspaces.  This 45-case BSND covering set keeps high B and high S1 apart.
_GRID_S2_OVERRIDES = {
    (1, 1024): 1281,
    (1, 4096): 1025,
    (64, 2048): 2305,
    (64, 5120): 2305,
    (512, 8192): 1025,
    (1024, 4096): 1281,
    (2048, 5120): 2305,
}


def _head_config(batch: int, s1: int, k: int, ordinal: int) -> Dict[str, int]:
    # Keep large workspace cases focused on B/S1/K coverage instead of heads.
    if batch * s1 * k >= 4 * 1024 * 1024:
        return HEAD_CONFIGS[-1]
    return HEAD_CONFIGS[ordinal % len(HEAD_CONFIGS)]


def _make_b1_grid_cases() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for s1_index, s1 in enumerate(S1_VALUES):
        for k_index, k in enumerate(K_VALUES):
            s2 = _GRID_S2_OVERRIDES.get((s1, k), max(s1, k))
            cases.append(
                _case(
                    f"bsnd_b1_s1_{s1}_k_{k}",
                    suite="core",
                    B=1,
                    S1=s1,
                    S2=s2,
                    K=k,
                    **_head_config(1, s1, k, s1_index + k_index),
                )
            )
    return cases


_B4_CASE_SPECS = (
    (1, 1024, 1024),
    (1, 8192, 1024),
    (64, 1024, 1281),
    (64, 4096, 2049),
    (512, 1024, 1024),
    (512, 2048, 2305),
    (1024, 1024, 1024),
    (1024, 2048, 1281),
    (2048, 1024, 2048),
    (2048, 2048, 2305),
)


def _make_b4_cases() -> List[Dict[str, Any]]:
    return [
        _case(
            f"bsnd_b4_s1_{s1}_k_{k}",
            suite="core",
            B=4,
            S1=s1,
            S2=s2,
            K=k,
            **_head_config(4, s1, k, ordinal),
        )
        for ordinal, (s1, k, s2) in enumerate(_B4_CASE_SPECS)
    ]


_B64_CASE_SPECS = (
    (1, 1024, 1024),
    (1, 2048, 1024),
    (1, 4096, 1024),
    (64, 1024, 1024),
    (64, 8192, 1024),
)


def _make_batch_stress_cases() -> List[Dict[str, Any]]:
    cases = [
        _case(
            f"bsnd_b64_s1_{s1}_k_{k}",
            suite="stress",
            B=64,
            S1=s1,
            S2=s2,
            K=k,
            **_head_config(64, s1, k, ordinal),
        )
        for ordinal, (s1, k, s2) in enumerate(_B64_CASE_SPECS)
    ]
    cases.extend(
        _case(
            f"bsnd_b256_s1_1_k_{k}",
            suite="stress",
            B=256,
            S1=1,
            S2=1024,
            K=k,
            **HEAD_CONFIGS[-1],
        )
        for k in K_VALUES
    )
    return cases


TND_CASES = [
    _case("tnd_s1_1_k_1024", suite="tnd", S1=1, S2=1024, K=1024, **HEAD_CONFIGS[0], layout="TND"),
    _case("tnd_s1_64_k_2048", suite="tnd", S1=64, S2=2305, K=2048, **HEAD_CONFIGS[1], layout="TND"),
    _case("tnd_s1_512_k_4096", suite="tnd", S1=512, S2=4097, K=4096, **HEAD_CONFIGS[2], layout="TND"),
    _case("tnd_s1_1024_k_5120", suite="tnd", S1=1024, S2=1281, K=5120, **HEAD_CONFIGS[3], layout="TND"),
    _case("tnd_s1_2048_k_8192", suite="tnd", S1=2048, S2=2305, K=8192, **HEAD_CONFIGS[3], layout="TND"),
]


ISOLATION_CASES = [
    _case(
        "isolate_valid_b1_s1_64_k_1024",
        suite="isolate",
        kind="valid-k",
        B=1,
        S1=64,
        S2=1087,
        N1=32,
        Nidx1=16,
        K=1024,
    ),
    _case(
        "isolate_tail_b1_s1_1_k_4096",
        suite="isolate",
        kind="tail-k",
        B=1,
        S1=1,
        S2=1025,
        N1=32,
        Nidx1=16,
        K=4096,
    ),
    _case(
        "isolate_valid_b4_s1_1_k_1024",
        suite="isolate",
        kind="valid-k",
        B=4,
        S1=1,
        S2=1024,
        N1=32,
        Nidx1=16,
        K=1024,
    ),
    _case(
        "isolate_valid_b4_s1_64_k_1024",
        suite="isolate",
        kind="valid-k",
        B=4,
        S1=64,
        S2=1087,
        N1=32,
        Nidx1=16,
        K=1024,
    ),
]

CORE_CASES = _make_b1_grid_cases() + _make_b4_cases()
STRESS_CASES = _make_batch_stress_cases()
BSND_CASES = CORE_CASES + STRESS_CASES
SMOKE_CASE_NAMES = (
    "bsnd_b1_s1_1_k_1024",
    "bsnd_b1_s1_64_k_2048",
    "bsnd_b4_s1_64_k_4096",
    "bsnd_b256_s1_1_k_1024",
    "tnd_s1_1_k_1024",
)
ALL_CASES = BSND_CASES + TND_CASES
CASE_BY_NAME = {case["name"]: case for case in ALL_CASES + ISOLATION_CASES}
assert len(BSND_CASES) == 45
assert len(TND_CASES) == 5
assert len(ALL_CASES) == 50
assert len(CASE_BY_NAME) == len(ALL_CASES) + len(ISOLATION_CASES)


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


def _valid_k_lengths(case: Dict[str, Any]) -> List[int]:
    return [
        min(case["K"], max(case["S2"] - case["S1"] + query_id + 1, 0))
        for query_id in range(case["S1"])
    ]


def _tail_summary(case: Dict[str, Any]) -> Tuple[int, int, int]:
    valid_lengths = _valid_k_lengths(case)
    valid_min = min(valid_lengths)
    valid_max = max(valid_lengths)
    invalid_per_batch = sum(case["K"] - valid_length for valid_length in valid_lengths)
    return valid_min, valid_max, invalid_per_batch * case["B"]


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


def make_inputs(case: Dict[str, Any], seed: int) -> Dict[str, Any]:
    _validate_case(case)
    input_dtype = torch.bfloat16
    torch.manual_seed(seed)

    B, S1, S2, N1, N2, Nidx1, Nidx2 = (
        case["B"], case["S1"], case["S2"], case["N1"], case["N2"], case["Nidx1"], case["Nidx2"]
    )
    DQuery, DQueryIndex, d_rope, K = case["DQuery"], case["DQueryIndex"], case["d_rope"], case["K"]
    G = N1 // N2
    scale_value = 1.0 / math.sqrt(DQuery)

    bsnd_shape = (B, S1, N1, DQuery)
    key_shape = (B, S2, N2, DQuery)
    query_index_shape = (B, S1, Nidx1, DQueryIndex)
    key_index_shape = (B, S2, Nidx2, DQueryIndex)
    query_rope_shape = (B, S1, N1, d_rope)
    key_rope_shape = (B, S2, N2, d_rope)
    weight_shape = (B, S1, Nidx1)

    query_bsnd = torch.randn(bsnd_shape, dtype=input_dtype).npu()
    key_bsnd = torch.randn(key_shape, dtype=input_dtype).npu()
    query_index_bsnd = torch.randn(query_index_shape, dtype=input_dtype).npu()
    key_index_bsnd = torch.randn(key_index_shape, dtype=input_dtype).npu()
    query_rope_bsnd = torch.randn(query_rope_shape, dtype=input_dtype).npu()
    key_rope_bsnd = torch.randn(key_rope_shape, dtype=input_dtype).npu()
    weights_bsnd = torch.randn(weight_shape, dtype=input_dtype).npu()
    sparse_indices_bsnd = _make_sparse_indices(case, seed)

    with torch.no_grad():
        if case["layout"] == "TND":
            query = query_bsnd.squeeze(0)
            key = key_bsnd.squeeze(0)
            query_index = query_index_bsnd.squeeze(0)
            key_index = key_index_bsnd.squeeze(0)
            query_rope = query_rope_bsnd.squeeze(0)
            key_rope = key_rope_bsnd.squeeze(0)
            weights = weights_bsnd.squeeze(0)
            sparse_indices = sparse_indices_bsnd.squeeze(0)

            k_mqa = key.squeeze(1)
            k_rope_mqa = key_rope.squeeze(1)
            q_full = torch.cat([query, query_rope], dim=-1)
            k_full = torch.cat([k_mqa, k_rope_mqa], dim=-1)
            k_full_t = k_full.transpose(-1, -2)
            sm_max_list = []
            sm_sum_list = []
            for s1_idx in range(S1):
                row = torch.matmul(q_full[s1_idx:s1_idx+1], k_full_t)
                row_scaled = row.float() * scale_value
                row_max = row_scaled.max(dim=-1).values
                row_sum = torch.exp(row_scaled - row_max.unsqueeze(-1)).sum(dim=-1)
                sm_max_list.append(row_max)
                sm_sum_list.append(row_sum)
            sm_max_val = torch.cat(sm_max_list, dim=0)
            sm_sum_val = torch.cat(sm_sum_list, dim=0)
            softmax_max = sm_max_val.reshape(1, S1, N2, G).permute(0, 2, 1, 3).contiguous().float()
            softmax_sum = sm_sum_val.reshape(1, S1, N2, G).permute(0, 2, 1, 3).contiguous().float()
            softmax_max = softmax_max.squeeze(0)
            softmax_sum = softmax_sum.squeeze(0)
            torch.npu.synchronize()
        else:
            query = query_bsnd
            key = key_bsnd
            query_index = query_index_bsnd
            key_index = key_index_bsnd
            query_rope = query_rope_bsnd
            key_rope = key_rope_bsnd
            weights = weights_bsnd
            sparse_indices = sparse_indices_bsnd

            k_mqa = key.squeeze(2)
            k_rope_mqa = key_rope.squeeze(2)
            q_full = torch.cat([query, query_rope], dim=-1)
            k_full = torch.cat([k_mqa, k_rope_mqa], dim=-1)
            k_full_t = k_full.transpose(-1, -2).unsqueeze(1)
            sm_max_list = []
            sm_sum_list = []
            for s1_idx in range(S1):
                row = torch.matmul(q_full[:, s1_idx:s1_idx+1], k_full_t)
                row_scaled = row.float() * scale_value
                row_max = row_scaled.max(dim=-1).values
                row_sum = torch.exp(row_scaled - row_max.unsqueeze(-1)).sum(dim=-1)
                sm_max_list.append(row_max)
                sm_sum_list.append(row_sum)
            sm_max_val = torch.cat(sm_max_list, dim=1)
            sm_sum_val = torch.cat(sm_sum_list, dim=1)
            softmax_max = sm_max_val.reshape(B, S1, N2, G).permute(0, 2, 1, 3).contiguous().float()
            softmax_sum = sm_sum_val.reshape(B, S1, N2, G).permute(0, 2, 1, 3).contiguous().float()

    return {
        "query": query,
        "key": key,
        "query_index": query_index,
        "key_index": key_index,
        "weights": weights,
        "sparse_indices": sparse_indices,
        "softmax_max": softmax_max,
        "softmax_sum": softmax_sum,
        "scale_value": scale_value,
        "query_rope": query_rope,
        "key_rope": key_rope,
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
    reference_norm = torch.norm(expected_f32).item()
    return {
        "max_abs": difference.max().item(),
        "mean_abs": difference.mean().item(),
        "mean_rel": (difference / (expected_f32.abs() + 1e-6)).mean().item(),
        "norm_rel": torch.linalg.vector_norm(difference).item() / (reference_norm + 1e-6),
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
        "norm_rel": loss_abs / (torch.norm(loss_expected).item() + 1e-6),
    }
    passed = all(value["mean_rel"] <= REL_TOLERANCE for value in metrics.values())
    return metrics, passed


def _print_metrics(metrics: Dict[str, Dict[str, float]]) -> None:
    for name, values in metrics.items():
        print(
            f"  {name:<12} max_abs={values['max_abs']:.6e} "
            f"mean_abs={values['mean_abs']:.6e} mean_rel={values['mean_rel']:.6e} "
            f"norm_rel={values['norm_rel']:.6e}"
        )


def _brief_metrics(metrics: Dict[str, Dict[str, float]]) -> str:
    return " ".join(
        f"{name}={metrics[name]['mean_rel']:.2e}"
        for name in ("dQueryIndex", "dKeyIndex", "dWeights", "Loss")
    )


def _case_summary(case: Dict[str, Any]) -> str:
    valid_min, valid_max, invalid_count = _tail_summary(case)
    return (
        f"suite={case['suite']} kind={case['kind']} layout={case['layout']} B={case['B']} S1={case['S1']} S2={case['S2']} "
        f"N1={case['N1']} Nidx1={case['Nidx1']} K={case['K']} "
        f"valid_k=[{valid_min},{valid_max}] invalid={invalid_count} dtype=bfloat16"
    )


def _print_batch_diagnostics(
    case: Dict[str, Any], tilelang_outputs: Sequence[torch.Tensor], official_outputs: Sequence[torch.Tensor]
) -> None:
    if case["layout"] != "BSND":
        return
    output_names = ("dQueryIndex", "dKeyIndex", "dWeights")
    for batch_id in range(case["B"]):
        values = []
        for name, actual, expected in zip(output_names, tilelang_outputs[:3], official_outputs[:3]):
            actual_batch = actual[batch_id].float()
            expected_batch = expected[batch_id].float()
            finite = torch.isfinite(actual_batch).all().item() and torch.isfinite(expected_batch).all().item()
            if finite:
                difference = (actual_batch - expected_batch).abs()
                relative = torch.norm(difference).item() / (torch.norm(expected_batch).item() + 1e-6)
                values.append(f"{name}={relative:.2e}")
            else:
                values.append(f"{name}=nonfinite")
        print(f"  batch={batch_id} {' '.join(values)}")


def _release_case_memory() -> None:
    """Release tensors from the previous case before compiling the next one."""
    gc.collect()
    torch.npu.empty_cache()


def run_case(case: Dict[str, Any], args: argparse.Namespace, index: int) -> str:
    case_prefix = f"[{index:02d}] {case['name']}"
    if args.verbose:
        print(f"\n{'=' * 88}\n{case_prefix}: {_case_summary(case)}")
    MODULE._clear_caches()
    torch.npu.synchronize()

    try:
        inputs = make_inputs(case, args.seed + index)
    except Exception as exc:
        print(f"{case_prefix} INPUT FAIL: {exc}")
        return "INPUT_FAIL"

    try:
        tilelang_outputs = _call_tilelang(inputs, case)
        torch.npu.synchronize()
        if args.verbose:
            print("  TileLang: completed")
    except Exception as exc:
        print(f"{case_prefix} TileLang FAIL: {exc}")
        return "TILELANG_FAIL"

    try:
        official_outputs = _call_official(inputs, case)
        torch.npu.synchronize()
        if args.verbose:
            print("  Official: completed")
    except Exception as exc:
        print(f"{case_prefix} Official FAIL: {exc}")
        return "OFFICIAL_FAIL"

    try:
        metrics, passed = _compare_outputs(tilelang_outputs, official_outputs)
        status = "PASS" if passed else "PRECISION_FAIL"
        if args.verbose:
            _print_metrics(metrics)
        if args.diagnose:
            _print_batch_diagnostics(case, tilelang_outputs, official_outputs)
        # Keep the normal 50-case run compact: a passing case is represented
        # by its benchmark line below, while every failure keeps its metrics.
        if not passed or (not args.bench and not args.summary_only):
            print(f"{case_prefix} {status} {_brief_metrics(metrics)}")
    except Exception as exc:
        print(f"{case_prefix} PRECISION FAIL: {exc}")
        status = "PRECISION_FAIL"

    if args.bench and status == "PASS":
        try:
            tl_avg, tl_med, tl_min = _bench(
                lambda: _call_tilelang(inputs, case), args.warmup, args.iters
            )
            off_avg, off_med, off_min = _bench(
                lambda: _call_official(inputs, case), args.warmup, args.iters
            )
            print(
                f"{case_prefix} bench tl_med={tl_med * 1e3:.3f}ms "
                f"off_med={off_med * 1e3:.3f}ms ratio={off_med / tl_med:.2f}x"
            )
            if args.verbose:
                print(
                    f"  TileLang: avg={tl_avg * 1e3:.3f}ms min={tl_min * 1e3:.3f}ms; "
                    f"Official: avg={off_avg * 1e3:.3f}ms min={off_min * 1e3:.3f}ms"
                )
        except Exception as exc:
            print(f"{case_prefix} BENCHMARK FAIL: {exc}")
            status = "BENCHMARK_FAIL"

    return status


def _select_cases(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    if args.case:
        unknown = [name for name in args.case if name not in CASE_BY_NAME]
        if unknown:
            raise ValueError(f"Unknown case(s): {', '.join(unknown)}")
        return [CASE_BY_NAME[name] for name in args.case]
    if args.suite == "smoke":
        return [CASE_BY_NAME[name] for name in SMOKE_CASE_NAMES]
    if args.suite == "core":
        return CORE_CASES
    if args.suite == "stress":
        return STRESS_CASES
    if args.suite == "tnd":
        return TND_CASES
    if args.suite == "isolate":
        return ISOLATION_CASES
    return ALL_CASES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("smoke", "core", "stress", "tnd", "isolate", "all"), default="all")
    parser.add_argument("--case", action="append", help="Run exactly this named case; can be repeated")
    parser.add_argument(
        "--bench",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Benchmark only precision-passing cases (default: enabled)",
    )
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERS)
    parser.add_argument("--iters", type=int, default=BENCH_ITERS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diagnose", action="store_true", help="Print per-batch normalized error diagnostics")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--verbose", action="store_true", help="Print full per-case diagnostics")
    output_group.add_argument(
        "--summary-only",
        action="store_true",
        help="Print failures, successful benchmark lines, and final totals",
    )
    parser.add_argument("--list-failures", action="store_true", help="List every non-passing case in the final summary")
    args = parser.parse_args()
    if args.warmup < 0 or args.iters < 1:
        parser.error("--warmup must be non-negative and --iters must be at least one")
    if args.diagnose:
        args.bench = False
    return args


def main() -> int:
    args = parse_args()
    if os.environ.get("SLIG_DEBUG", "0") == "1" or os.environ.get("SLIG_PROFILE", "0") == "1":
        print("WARNING: SLIG_DEBUG/SLIG_PROFILE is enabled; disable it for benchmark numbers.")
    selected_cases = list(_select_cases(args))
    if not args.summary_only:
        print(f"0721 general test: suite={args.suite}, cases={len(selected_cases)}, benchmark={args.bench}")

    results: List[Tuple[str, str]] = []
    for index, case in enumerate(selected_cases):
        results.append((case["name"], run_case(case, args, index)))
        _release_case_memory()

    status_counts = Counter(status for _, status in results)
    passed_count = status_counts["PASS"]
    failed_count = len(results) - passed_count
    print(
        f"Summary: total={len(results)} pass={passed_count} fail={failed_count} "
        f"precision={status_counts['PRECISION_FAIL']} "
        f"tilelang_error={status_counts['TILELANG_FAIL']} "
        f"official_error={status_counts['OFFICIAL_FAIL']} "
        f"input_error={status_counts['INPUT_FAIL']} "
        f"bench_error={status_counts['BENCHMARK_FAIL']}"
    )
    if args.list_failures:
        for status in ("PRECISION_FAIL", "TILELANG_FAIL", "OFFICIAL_FAIL", "INPUT_FAIL", "BENCHMARK_FAIL"):
            names = [name for name, case_status in results if case_status == status]
            if names:
                print(f"{status}: {', '.join(names)}")
    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
