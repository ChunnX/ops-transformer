# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Accuracy test for the PyTorch BlockAttnResPrepare interface.

Examples:
    python3 test_torch_block_attn_res_prepare.py
    python3 test_torch_block_attn_res_prepare.py --suite network
    python3 test_torch_block_attn_res_prepare.py --shape 254 8 24 7168 --valid-blocks 1 4 8 9

The custom shape arguments are ordered as T, N, S, D.  The golden result is
calculated in FP64 on CPU and rounded to FP32 before comparison.
"""

import argparse
import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import torch


TensorTriple = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class ShapeCase:
    name: str
    total_t: int
    total_n: int
    total_s: int
    total_d: int


@dataclass(frozen=True)
class AccuracyMetrics:
    passed: bool
    max_abs: float
    max_rel: float
    rel_l2: float
    bad_elements: int
    total_elements: int


SUITES: Dict[str, Sequence[ShapeCase]] = {
    "smoke": (
        # Exercises a non-64-aligned D tail in the Vector implementation.
        ShapeCase("vector_tail", total_t=2, total_n=8, total_s=3, total_d=257),
        # Meets the minimum T/S/D requirements of the Cube candidate path.
        ShapeCase("cube_candidate", total_t=32, total_n=8, total_s=16, total_d=256),
    ),
    "network": (
        ShapeCase("decode_s18", total_t=1, total_n=8, total_s=18, total_d=7168),
        ShapeCase("decode_s24", total_t=4, total_n=8, total_s=24, total_d=7168),
        ShapeCase("prefill_cube", total_t=32, total_n=8, total_s=24, total_d=7168),
    ),
}

OUTPUT_NAMES = ("softmax_max", "weighted_output", "softmax_sum")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare BlockAttnResPrepare NPU output with an FP64 CPU golden."
    )
    parser.add_argument(
        "--suite", choices=tuple(SUITES), default="smoke", help="Preset shape suite."
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=4,
        metavar=("T", "N", "S", "D"),
        help="Run one custom [T,N,D]/[S,D] shape instead of the selected suite.",
    )
    parser.add_argument(
        "--valid-blocks",
        type=int,
        nargs="+",
        default=None,
        help="Runtime valid_blocks values. Default covers negative, zero, one, partial, full, and clamped values.",
    )
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--golden-chunk-t",
        type=int,
        default=8,
        help="Number of T rows processed per FP64 golden chunk.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not math.isfinite(args.epsilon) or args.epsilon <= 0.0:
        raise ValueError("epsilon must be finite and greater than zero")
    if args.rtol < 0.0 or args.atol < 0.0:
        raise ValueError("rtol and atol must be non-negative")
    if args.golden_chunk_t <= 0:
        raise ValueError("golden-chunk-t must be greater than zero")
    if args.shape is not None:
        total_t, total_n, total_s, total_d = args.shape
        if (
            total_t <= 0
            or total_s <= 0
            or total_n < 1
            or total_n > 64
            or total_d < 1
            or total_d > 8192
        ):
            raise ValueError("shape must satisfy T>0, S>0, 1<=N<=64, and 1<=D<=8192")


def load_npu_operator() -> Callable[..., TensorTriple]:
    try:
        import torch_npu  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import
        from cann_ops_transformer.ops import block_attn_res_prepare
    except ImportError as error:
        raise RuntimeError(
            "torch_npu and cann_ops_transformer must be installed before running this test. "
            "Build/install the package under torch_extension first."
        ) from error
    return block_attn_res_prepare


def make_inputs(case: ShapeCase, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    # Scaling Q by sqrt(D) keeps logits in a useful range and avoids a trivially
    # saturated softmax for large network head dimensions.
    block_residual = 0.25 * torch.randn(
        case.total_t,
        case.total_n,
        case.total_d,
        generator=generator,
        dtype=torch.float32,
    )
    effective_query = torch.randn(
        case.total_s,
        case.total_d,
        generator=generator,
        dtype=torch.float32,
    ) / math.sqrt(case.total_d)
    return block_residual.contiguous(), effective_query.contiguous()


def default_valid_blocks(total_n: int) -> List[int]:
    values = (-1, 0, 1, max(1, total_n // 2), total_n, total_n + 3)
    return list(dict.fromkeys(values))


def cpu_golden(
    block_residual: torch.Tensor,
    effective_query: torch.Tensor,
    valid_blocks: int,
    epsilon: float,
    chunk_t: int,
) -> TensorTriple:
    """Calculate the online-softmax state in FP64 and return FP32 tensors."""
    total_t, total_n, total_d = block_residual.shape
    total_s = effective_query.shape[0]
    valid_n = min(max(valid_blocks, 0), total_n)

    if valid_n == 0:
        softmax_max = torch.full(
            (total_s, total_t),
            torch.finfo(torch.float32).min,
            dtype=torch.float32,
        )
        weighted_output = torch.zeros((total_s, total_t, total_d), dtype=torch.float32)
        softmax_sum = torch.ones((total_s, total_t), dtype=torch.float32)
        return softmax_max, weighted_output, softmax_sum

    # Match the float attribute received by the C++/ACLNN interface, while all
    # following reference arithmetic remains FP64.
    epsilon_fp32 = float(torch.tensor(epsilon, dtype=torch.float32).item())
    query_fp64 = effective_query.to(torch.float64)
    softmax_max = torch.empty((total_s, total_t), dtype=torch.float32)
    weighted_output = torch.empty((total_s, total_t, total_d), dtype=torch.float32)
    softmax_sum = torch.empty((total_s, total_t), dtype=torch.float32)

    for t_start in range(0, total_t, chunk_t):
        t_end = min(t_start + chunk_t, total_t)
        residual_fp64 = block_residual[t_start:t_end, :valid_n, :].to(torch.float64)

        square_sum = residual_fp64.square().sum(dim=-1)
        rms = torch.sqrt(square_sum / float(total_d) + epsilon_fp32)
        dot = torch.einsum("sd,tnd->stn", query_fp64, residual_fp64)
        logits = dot / rms.unsqueeze(0)

        max_chunk = logits.amax(dim=-1)
        exp_chunk = torch.exp(logits - max_chunk.unsqueeze(-1))
        sum_chunk = exp_chunk.sum(dim=-1)
        # Phase 1 intentionally returns the unnormalized numerator.  A later
        # online-softmax merge consumes it together with softmax_sum.
        output_chunk = torch.einsum("stn,tnd->std", exp_chunk, residual_fp64)

        softmax_max[:, t_start:t_end] = max_chunk.to(torch.float32)
        softmax_sum[:, t_start:t_end] = sum_chunk.to(torch.float32)
        weighted_output[:, t_start:t_end, :] = output_chunk.to(torch.float32)

    return softmax_max, weighted_output, softmax_sum


def accuracy_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float,
    atol: float,
) -> AccuracyMetrics:
    if actual.shape != expected.shape:
        return AccuracyMetrics(
            False, math.inf, math.inf, math.inf, expected.numel(), expected.numel()
        )

    actual_fp64 = actual.to(torch.float64)
    expected_fp64 = expected.to(torch.float64)
    finite = torch.isfinite(actual_fp64) & torch.isfinite(expected_fp64)
    diff = (actual_fp64 - expected_fp64).abs()
    allowed = atol + rtol * expected_fp64.abs()
    bad = (~finite) | (diff > allowed)

    if not bool(finite.all()):
        max_abs = math.inf
        max_rel = math.inf
        rel_l2 = math.inf
    else:
        max_abs = float(diff.max().item()) if diff.numel() else 0.0
        relative = diff / expected_fp64.abs().clamp_min(1e-12)
        max_rel = float(relative.max().item()) if relative.numel() else 0.0
        ref_norm = torch.linalg.vector_norm(expected_fp64.reshape(-1))
        diff_norm = torch.linalg.vector_norm(diff.reshape(-1))
        rel_l2 = float((diff_norm / ref_norm.clamp_min(1e-30)).item())

    return AccuracyMetrics(
        passed=not bool(bad.any()),
        max_abs=max_abs,
        max_rel=max_rel,
        rel_l2=rel_l2,
        bad_elements=int(bad.sum().item()),
        total_elements=expected.numel(),
    )


def check_output_contract(outputs: TensorTriple, case: ShapeCase) -> None:
    expected_shapes = (
        (case.total_s, case.total_t),
        (case.total_s, case.total_t, case.total_d),
        (case.total_s, case.total_t),
    )
    if len(outputs) != len(expected_shapes):
        raise AssertionError(f"expected three outputs, got {len(outputs)}")
    for name, output, expected_shape in zip(OUTPUT_NAMES, outputs, expected_shapes):
        if tuple(output.shape) != expected_shape:
            raise AssertionError(
                f"{name}: expected shape {expected_shape}, got {tuple(output.shape)}"
            )
        if output.dtype != torch.float32:
            raise AssertionError(f"{name}: expected torch.float32, got {output.dtype}")


def run_case(
    case: ShapeCase,
    valid_values: Iterable[int],
    block_attn_res_prepare: Callable[..., TensorTriple],
    args: argparse.Namespace,
    seed: int,
) -> Tuple[int, int]:
    block_residual, effective_query = make_inputs(case, seed)
    block_residual_npu = block_residual.to(args.device)
    effective_query_npu = effective_query.to(args.device)
    passed_checks = 0
    total_checks = 0

    print(
        f"\n[{case.name}] V=[{case.total_t},{case.total_n},{case.total_d}] "
        f"Q=[{case.total_s},{case.total_d}]"
    )
    for valid_value in valid_values:
        valid_blocks_npu = torch.tensor(
            [valid_value], dtype=torch.int32, device=args.device
        )
        with torch.inference_mode():
            npu_outputs = block_attn_res_prepare(
                block_residual_npu,
                effective_query_npu,
                valid_blocks_npu,
                epsilon=args.epsilon,
            )
            torch.npu.synchronize()
            actual_outputs = tuple(output.cpu() for output in npu_outputs)

        check_output_contract(actual_outputs, case)
        golden_outputs = cpu_golden(
            block_residual,
            effective_query,
            valid_value,
            args.epsilon,
            args.golden_chunk_t,
        )

        clamped_value = min(max(valid_value, 0), case.total_n)
        print(f"  valid_blocks={valid_value} (effective={clamped_value})")
        for name, actual, expected in zip(OUTPUT_NAMES, actual_outputs, golden_outputs):
            metrics = accuracy_metrics(actual, expected, args.rtol, args.atol)
            status = "PASS" if metrics.passed else "FAIL"
            print(
                f"    {status} {name:<16} max_abs={metrics.max_abs:.6e} "
                f"max_rel={metrics.max_rel:.6e} rel_l2={metrics.rel_l2:.6e} "
                f"bad={metrics.bad_elements}/{metrics.total_elements}"
            )
            total_checks += 1
            passed_checks += int(metrics.passed)
            if args.fail_fast and not metrics.passed:
                return passed_checks, total_checks

    return passed_checks, total_checks


def selected_cases(args: argparse.Namespace) -> Sequence[ShapeCase]:
    if args.shape is None:
        return SUITES[args.suite]
    total_t, total_n, total_s, total_d = args.shape
    return (ShapeCase("custom", total_t, total_n, total_s, total_d),)


def main() -> int:
    args = parse_args()
    validate_args(args)
    block_attn_res_prepare = load_npu_operator()
    torch.npu.set_device(args.device)

    total_passed = 0
    total_checks = 0
    print(
        f"device={args.device}, epsilon={args.epsilon}, rtol={args.rtol}, atol={args.atol}"
    )
    for case_index, case in enumerate(selected_cases(args)):
        valid_values = args.valid_blocks or default_valid_blocks(case.total_n)
        passed, checked = run_case(
            case,
            valid_values,
            block_attn_res_prepare,
            args,
            args.seed + case_index,
        )
        total_passed += passed
        total_checks += checked
        if args.fail_fast and passed != checked:
            break

    failed = total_checks - total_passed
    print(f"\nsummary: passed={total_passed}/{total_checks}, failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
