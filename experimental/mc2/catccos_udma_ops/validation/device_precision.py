# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Bounded-memory precision checks that keep model-sized tensors on device."""

from __future__ import annotations

import math

import torch

try:
    from .precision import _hard_limit_at_scalar, precision_standard
except ImportError:  # Direct torchrun execution.
    from precision import _hard_limit_at_scalar, precision_standard


def _finite_or_none(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _segment_for_row(segments, row):
    for segment in segments:
        if segment["row_start"] <= row < segment["row_end"]:
            return segment
    return {
        "local_expert": None,
        "global_expert": None,
        "source_rank": None,
    }


def _aggregate_breakdown(rows, key):
    aggregate = {}
    for row in rows:
        value = row.get(key)
        current = aggregate.setdefault(value, {
            key: value,
            "rows": 0,
            "elements": 0,
            "mixed_mismatches": 0,
            "hard_violations": 0,
            "max_abs_error": 0.0,
        })
        current["rows"] += row["rows"]
        current["elements"] += row["elements"]
        current["mixed_mismatches"] += row["mixed_mismatches"]
        current["hard_violations"] += row["hard_violations"]
        maximum = row["max_abs_error"]
        if maximum is None:
            current["max_abs_error"] = None
        elif current["max_abs_error"] is not None:
            current["max_abs_error"] = max(
                current["max_abs_error"], maximum
            )
    return [aggregate[value] for value in sorted(aggregate)]


def _error_tensors(actual, expected, standard):
    actual_fp32 = actual.float()
    expected_fp32 = expected.float()
    absolute_error = (actual_fp32 - expected_fp32).abs()
    mixed_limit = standard["atol"] + standard["rtol"] * expected_fp32.abs()
    mixed_mismatch = ~(absolute_error <= mixed_limit)
    golden_abs = expected_fp32.abs()
    tiny = torch.finfo(torch.float32).tiny
    exponents = torch.floor(torch.log2(golden_abs.clamp_min(tiny)))
    exponents = torch.clamp_min(
        exponents, float(standard["ulp_min_exponent"])
    )
    ulp = torch.pow(2.0, exponents - standard["ulp_mantissa_bits"])
    hard_limit = torch.clamp_min(
        32.0 * ulp, standard["max_abs_error_limit_fixed"]
    )
    hard_mismatch = ~(absolute_error <= hard_limit)
    return (
        actual_fp32,
        expected_fp32,
        absolute_error,
        mixed_mismatch,
        hard_mismatch,
    )


def analyze_accuracy_on_device(
    actual,
    expected,
    segments,
    *,
    topk,
    chunk_elements,
    include_breakdown,
):
    """Apply the opbase criteria with bounded device working memory.

    Only scalar reductions and the final Top-K values are copied to Host. This
    keeps a model-sized comparison from materializing two FP32 CPU tensors.
    """
    if tuple(actual.shape) != tuple(expected.shape):
        raise ValueError(
            "shape mismatch: actual={} expected={}".format(
                tuple(actual.shape), tuple(expected.shape)
            )
        )
    if actual.dim() != 2:
        raise ValueError("device accuracy diagnostics require a 2D output")
    if topk <= 0 or chunk_elements <= 0:
        raise ValueError("topk and chunk_elements must be positive")

    standard = precision_standard(actual.dtype)
    actual_flat = actual.detach().reshape(-1)
    expected_flat = expected.detach().reshape(-1)
    element_count = actual_flat.numel()
    mixed_counts = []
    hard_counts = []
    actual_nonfinite_counts = []
    expected_nonfinite_counts = []
    maximum_errors = []
    top_values = []
    top_indices = []

    for start in range(0, element_count, chunk_elements):
        stop = min(element_count, start + chunk_elements)
        (
            actual_fp32,
            expected_fp32,
            absolute_error,
            mixed_mismatch,
            hard_mismatch,
        ) = _error_tensors(
            actual_flat[start:stop], expected_flat[start:stop], standard
        )
        mixed_counts.append(mixed_mismatch.sum(dtype=torch.int64))
        hard_counts.append(hard_mismatch.sum(dtype=torch.int64))
        actual_nonfinite_counts.append(
            (~torch.isfinite(actual_fp32)).sum(dtype=torch.int64)
        )
        expected_nonfinite_counts.append(
            (~torch.isfinite(expected_fp32)).sum(dtype=torch.int64)
        )
        maximum_errors.append(
            torch.nan_to_num(
                absolute_error,
                nan=math.inf,
                posinf=math.inf,
                neginf=math.inf,
            ).max()
        )
        current_topk = min(topk, stop - start)
        values, indices = torch.topk(
            torch.nan_to_num(
                absolute_error,
                nan=math.inf,
                posinf=math.inf,
                neginf=math.inf,
            ),
            current_topk,
        )
        top_values.append(values)
        top_indices.append(indices.to(torch.int64) + start)

    def _sum_scalars(values):
        if not values:
            return 0
        return int(torch.stack(values).sum().item())

    mixed_count = _sum_scalars(mixed_counts)
    hard_count = _sum_scalars(hard_counts)
    actual_nonfinite = _sum_scalars(actual_nonfinite_counts)
    expected_nonfinite = _sum_scalars(expected_nonfinite_counts)
    max_abs_error = (
        float(torch.stack(maximum_errors).max().item())
        if maximum_errors else 0.0
    )
    matched_ratio = (
        1.0 if element_count == 0
        else (element_count - mixed_count) / element_count
    )
    passed = (
        matched_ratio >= standard["required_matched_ratio"]
        and hard_count == 0
        and actual_nonfinite == 0
        and expected_nonfinite == 0
    )

    top_mismatches = []
    if top_values:
        candidate_values = torch.cat(top_values)
        candidate_indices = torch.cat(top_indices)
        final_topk = min(topk, candidate_values.numel())
        selected_values, selection = torch.topk(
            candidate_values, final_topk
        )
        selected_indices = candidate_indices.index_select(0, selection)
        selected_actual = actual_flat.index_select(
            0, selected_indices
        ).float().cpu().tolist()
        selected_expected = expected_flat.index_select(
            0, selected_indices
        ).float().cpu().tolist()
        values_host = selected_values.cpu().tolist()
        indices_host = selected_indices.cpu().tolist()
        columns = actual.shape[1]
        for flat_index, error, actual_value, expected_value in zip(
            indices_host, values_host, selected_actual, selected_expected
        ):
            row, column = divmod(int(flat_index), columns)
            segment = _segment_for_row(segments, row)
            ulp, hard_limit = _hard_limit_at_scalar(
                float(expected_value), standard
            )
            mixed_limit = standard["atol"] + standard["rtol"] * abs(
                float(expected_value)
            )
            top_mismatches.append({
                "flat_index": int(flat_index),
                "row": row,
                "column": column,
                "local_expert": segment.get("local_expert"),
                "global_expert": segment.get("global_expert"),
                "source_rank": segment.get("source_rank"),
                "actual": _finite_or_none(actual_value),
                "expected": _finite_or_none(expected_value),
                "abs_error": _finite_or_none(error),
                "mixed_limit": _finite_or_none(mixed_limit),
                "hard_limit": _finite_or_none(hard_limit),
                "ulp_at_abs_expected": _finite_or_none(ulp),
                "mixed_tolerance_pass": bool(error <= mixed_limit),
                "hard_limit_pass": bool(error <= hard_limit),
            })

    by_segment = []
    if include_breakdown and not passed:
        segment_mixed = []
        segment_hard = []
        segment_max = []
        for segment in segments:
            start = segment["row_start"]
            stop = segment["row_end"]
            if start == stop:
                segment_mixed.append(
                    torch.zeros((), dtype=torch.int64, device=actual.device)
                )
                segment_hard.append(
                    torch.zeros((), dtype=torch.int64, device=actual.device)
                )
                segment_max.append(
                    torch.zeros((), dtype=torch.float32, device=actual.device)
                )
                continue
            _, _, errors, mixed, hard = _error_tensors(
                actual[start:stop], expected[start:stop], standard
            )
            segment_mixed.append(mixed.sum(dtype=torch.int64))
            segment_hard.append(hard.sum(dtype=torch.int64))
            segment_max.append(
                torch.nan_to_num(
                    errors, nan=math.inf, posinf=math.inf, neginf=math.inf
                ).max()
            )
        mixed_host = torch.stack(segment_mixed).cpu().tolist()
        hard_host = torch.stack(segment_hard).cpu().tolist()
        max_host = torch.stack(segment_max).cpu().tolist()
        for segment, mixed, hard, maximum in zip(
            segments, mixed_host, hard_host, max_host
        ):
            by_segment.append({
                **segment,
                "elements": (
                    segment["row_end"] - segment["row_start"]
                ) * actual.shape[1],
                "mixed_mismatches": int(mixed),
                "hard_violations": int(hard),
                "max_abs_error": _finite_or_none(maximum),
            })

    reasons = []
    if matched_ratio < standard["required_matched_ratio"]:
        reasons.append("FAIL_MATCHED_RATIO")
    if hard_count:
        reasons.append("FAIL_MAX_ABS_ERROR")
    if actual_nonfinite:
        reasons.append("FAIL_ACTUAL_NONFINITE")
    if expected_nonfinite:
        reasons.append("FAIL_EXPECTED_NONFINITE")
    return {
        "passed": passed,
        "failure_reasons": reasons,
        "precision_standard": standard,
        "shape": list(actual.shape),
        "elements": element_count,
        "mixed_mismatch_count": mixed_count,
        "matched_ratio": matched_ratio,
        "hard_limit_violation_count": hard_count,
        "max_abs_error": _finite_or_none(max_abs_error),
        "actual_nonfinite": actual_nonfinite,
        "expected_nonfinite": expected_nonfinite,
        "top_mismatches": top_mismatches,
        "by_route_segment": by_segment,
        "by_source_rank": (
            _aggregate_breakdown(by_segment, "source_rank")
            if by_segment else []
        ),
        "by_local_expert": (
            _aggregate_breakdown(by_segment, "local_expert")
            if by_segment else []
        ),
        "comparison_device": str(actual.device),
        "comparison_strategy": (
            "chunked device opbase reductions; only Top-K and scalar "
            "summaries are copied to Host"
        ),
    }
