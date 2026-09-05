# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Opbase-compliant mixed-tolerance checks and actionable error reports."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union


OPBASE_STANDARD_URL = (
    "https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/"
    "experimental_standard.md"
)
OPBASE_REVISION = "7a4b727744c443bfa84d0efc12e54fc18100ab83"

_TOLERANCE_TABLE = {
    # rtol, atol, required ratio, fixed hard limit, mantissa bits, minimum exponent
    "float16": (2 ** -9, 2 ** -9, 0.99, 1e-1, 10, -14),
    "bfloat16": (2 ** -6, 2 ** -6, 0.99, 1e0, 7, -126),
    "float32": (2 ** -10, 2 ** -16, 0.99, 1e-2, 23, -126),
    "hifloat32": (2 ** -9, 2 ** -10, 0.99, 1e-1, 23, -126),
    "float8e4m3": (2 ** -2, 2 ** -4, 0.99, 1e0, 3, -6),
    "float8e5m2": (2 ** -1, 2 ** -3, 0.99, 1e-1, 2, -14),
}

_QUANTILE_SAMPLE_LIMIT = 1 << 20
_REDUCTION_CHUNK_ELEMENTS = 1 << 22


def _normalize_dtype(dtype: Any) -> str:
    value = str(dtype).lower().replace("torch.", "").replace("_", "").replace(" ", "")
    if "float8e4m3" in value:
        return "float8e4m3"
    if "float8e5m2" in value:
        return "float8e5m2"
    return value


def precision_standard(dtype: Any) -> Dict[str, Any]:
    normalized = _normalize_dtype(dtype)
    if normalized not in _TOLERANCE_TABLE:
        raise ValueError(f"unsupported opbase precision dtype: {dtype}")
    rtol, atol, required_ratio, fixed_limit, mantissa_bits, emin = _TOLERANCE_TABLE[normalized]
    return {
        "name": "opbase_experimental",
        "source": OPBASE_STANDARD_URL,
        "opbase_revision": OPBASE_REVISION,
        "dtype": normalized,
        "rtol": rtol,
        "atol": atol,
        "required_matched_ratio": required_ratio,
        "max_abs_error_limit_fixed": fixed_limit,
        "max_abs_error_limit": "max(fixed, 32 * ULP_at_abs_golden)",
        "max_abs_error_limit_scope": "per_element",
        "ulp_mantissa_bits": mantissa_bits,
        "ulp_min_exponent": emin,
    }


def _hard_limit_values(golden_abs: Any, standard: Mapping[str, Any]) -> Any:
    """Compute max(fixed, 32 * ULP) at each Golden magnitude."""
    import torch

    values = golden_abs.double()
    exponents = torch.floor(torch.log2(values))
    exponents = torch.clamp_min(exponents, standard["ulp_min_exponent"])
    ulp = torch.pow(2.0, exponents - standard["ulp_mantissa_bits"])
    return torch.clamp_min(32.0 * ulp, standard["max_abs_error_limit_fixed"])


def _hard_limit_statistics(
    abs_error: Any, expected: Any, standard: Mapping[str, Any]
) -> Tuple[int, Optional[float]]:
    """Check the per-element hard limit without materializing another full tensor."""
    import torch

    error_flat = abs_error.flatten()
    expected_flat = expected.flatten()
    violations = 0
    maximum_ratio: Optional[float] = None
    for start in range(0, error_flat.numel(), _REDUCTION_CHUNK_ELEMENTS):
        error_chunk = error_flat[start:start + _REDUCTION_CHUNK_ELEMENTS]
        golden_abs = expected_flat[start:start + _REDUCTION_CHUNK_ELEMENTS].abs()
        limit = _hard_limit_values(golden_abs, standard)
        violations += int((~(error_chunk <= limit)).sum().item())
        ratio = error_chunk / limit
        finite_ratio = ratio[torch.isfinite(ratio)]
        if finite_ratio.numel():
            chunk_max = float(finite_ratio.max().item())
            maximum_ratio = chunk_max if maximum_ratio is None else max(maximum_ratio, chunk_max)
    return violations, maximum_ratio


def _hard_limit_at_scalar(golden: float, standard: Mapping[str, Any]) -> Tuple[float, float]:
    golden_abs = abs(golden)
    if not math.isfinite(golden_abs):
        return math.nan, math.nan
    if golden_abs == 0.0:
        exponent = int(standard["ulp_min_exponent"])
    else:
        exponent = max(math.floor(math.log2(golden_abs)), int(standard["ulp_min_exponent"]))
    ulp = math.ldexp(1.0, exponent - int(standard["ulp_mantissa_bits"]))
    return ulp, max(float(standard["max_abs_error_limit_fixed"]), 32.0 * ulp)


def _safe_float(value: Any) -> Optional[float]:
    result = float(value)
    return result if math.isfinite(result) else None


def _quantiles(
    values: Any, *, max_quantile_samples: int = _QUANTILE_SAMPLE_LIMIT
) -> Dict[str, Any]:
    """Return exact reductions and deterministic sampled quantiles at large scale."""
    import torch

    if max_quantile_samples <= 0:
        raise ValueError("max_quantile_samples must be positive")
    flat = values.double().flatten()
    total_elements = flat.numel()
    finite_elements = 0
    finite_sum = 0.0
    finite_max: Optional[float] = None
    for start in range(0, total_elements, _REDUCTION_CHUNK_ELEMENTS):
        chunk = flat[start:start + _REDUCTION_CHUNK_ELEMENTS]
        finite_mask = torch.isfinite(chunk)
        if bool(finite_mask.all()):
            finite_chunk = chunk
        else:
            finite_chunk = chunk[finite_mask]
        if finite_chunk.numel() == 0:
            continue
        finite_elements += finite_chunk.numel()
        finite_sum += float(finite_chunk.sum().item())
        chunk_max = float(finite_chunk.max().item())
        finite_max = chunk_max if finite_max is None else max(finite_max, chunk_max)

    if total_elements <= max_quantile_samples:
        sample = flat
        quantiles_approximate = False
    else:
        indices = torch.div(
            torch.arange(max_quantile_samples, dtype=torch.int64) * total_elements,
            max_quantile_samples,
            rounding_mode="floor",
        )
        sample = flat[indices]
        quantiles_approximate = True
    sample = sample[torch.isfinite(sample)]

    if sample.numel() == 0 and finite_elements:
        for start in range(0, total_elements, _REDUCTION_CHUNK_ELEMENTS):
            chunk = flat[start:start + _REDUCTION_CHUNK_ELEMENTS]
            finite_chunk = chunk[torch.isfinite(chunk)]
            if finite_chunk.numel():
                sample = finite_chunk[:max_quantile_samples]
                quantiles_approximate = finite_elements > sample.numel()
                break

    metadata = {
        "total_elements": total_elements,
        "finite_elements": finite_elements,
        "quantile_sample_elements": sample.numel(),
        "quantiles_approximate": quantiles_approximate,
    }
    if finite_elements == 0:
        return {
            **{name: None for name in ("mean", "median", "p90", "p99", "max")},
            **metadata,
        }
    points = torch.tensor([0.5, 0.9, 0.99], dtype=torch.float64)
    quantiles = torch.quantile(sample, points)
    return {
        "mean": _safe_float(finite_sum / finite_elements),
        "median": _safe_float(quantiles[0]),
        "p90": _safe_float(quantiles[1]),
        "p99": _safe_float(quantiles[2]),
        "max": _safe_float(finite_max),
        **metadata,
    }


def _build_segments(global_table: Sequence[Sequence[int]]) -> List[Dict[str, int]]:
    if not global_table:
        return []
    local_experts = len(global_table[0])
    segments = []
    offset = 0
    for local_expert in range(local_experts):
        for source_rank, row in enumerate(global_table):
            count = int(row[local_expert])
            segments.append(
                {
                    "local_expert": local_expert,
                    "source_rank": source_rank,
                    "row_start": offset,
                    "row_end": offset + count,
                    "rows": count,
                }
            )
            offset += count
    return segments


def _segment_for_row(segments: Sequence[Mapping[str, int]], row: int) -> Tuple[Optional[int], Optional[int]]:
    for segment in segments:
        if segment["row_start"] <= row < segment["row_end"]:
            return int(segment["local_expert"]), int(segment["source_rank"])
    return None, None


def _breakdown(
    abs_error: Any,
    mismatch: Any,
    segments: Sequence[Mapping[str, int]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    expert_stats: Dict[int, Dict[str, Any]] = {}
    source_stats: Dict[int, Dict[str, Any]] = {}
    for segment in segments:
        start, end = segment["row_start"], segment["row_end"]
        segment_elements = max(0, end - start) * abs_error.shape[1]
        segment_mismatch = int(mismatch[start:end].sum().item()) if segment_elements else 0
        segment_max = _safe_float(abs_error[start:end].max()) if segment_elements else 0.0
        for key, value, target in (
            ("local_expert", segment["local_expert"], expert_stats),
            ("source_rank", segment["source_rank"], source_stats),
        ):
            stats = target.setdefault(
                int(value),
                {key: int(value), "elements": 0, "mismatches": 0, "max_abs_error": 0.0},
            )
            stats["elements"] += segment_elements
            stats["mismatches"] += segment_mismatch
            stats["max_abs_error"] = max(float(stats["max_abs_error"]), float(segment_max or 0.0))
    for collection in (expert_stats, source_stats):
        for stats in collection.values():
            stats["mismatch_ratio"] = (
                stats["mismatches"] / stats["elements"] if stats["elements"] else 0.0
            )
    return list(expert_stats.values()), list(source_stats.values())


def _error_ratio_histogram(error_ratio: Any) -> List[Dict[str, Any]]:
    flat = error_ratio.flatten()
    boundaries = [0.25, 0.5, 1.0, 2.0, 4.0]
    labels = ["[0,0.25]", "(0.25,0.5]", "(0.5,1]", "(1,2]", "(2,4]", ">4"]
    total = max(1, flat.numel())
    result = []
    for index, label in enumerate(labels):
        if index == 0:
            mask = flat <= boundaries[0]
        elif index == len(labels) - 1:
            mask = flat > boundaries[-1]
        else:
            mask = (flat > boundaries[index - 1]) & (flat <= boundaries[index])
        count = int(mask.sum().item())
        result.append({"range": label, "count": count, "ratio": count / total})
    return result


def _heatmap(error_ratio: Any, bins: int) -> Dict[str, Any]:
    rows, columns = error_ratio.shape
    row_bins = min(max(1, bins), rows)
    column_bins = min(max(1, bins), columns)
    values: List[List[Optional[float]]] = []
    for row_bucket in range(row_bins):
        row_start = row_bucket * rows // row_bins
        row_end = (row_bucket + 1) * rows // row_bins
        current = []
        for column_bucket in range(column_bins):
            column_start = column_bucket * columns // column_bins
            column_end = (column_bucket + 1) * columns // column_bins
            current.append(_safe_float(error_ratio[row_start:row_end, column_start:column_end].max()))
        values.append(current)
    return {
        "row_bins": row_bins,
        "column_bins": column_bins,
        "value": "max_error_ratio",
        "values": values,
    }


def analyze_accuracy(
    actual: Any,
    expected: Any,
    *,
    global_table: Optional[Sequence[Sequence[int]]] = None,
    segments_override: Optional[Sequence[Mapping[str, int]]] = None,
    topk: int = 20,
    heatmap_bins: int = 32,
) -> Dict[str, Any]:
    """Compare meaningful output rows using the official opbase BF16 standard."""
    import torch

    if tuple(actual.shape) != tuple(expected.shape):
        raise ValueError(f"shape mismatch: actual={tuple(actual.shape)}, expected={tuple(expected.shape)}")
    if actual.dim() != 2:
        raise ValueError("accuracy diagnostics currently require a two-dimensional output")

    standard = precision_standard(actual.dtype)
    actual_cpu = actual.detach().float().cpu()
    expected_cpu = expected.detach().float().cpu()
    abs_error = (actual_cpu - expected_cpu).abs().double()
    threshold = standard["atol"] + standard["rtol"] * expected_cpu.abs().double()
    mismatch = ~(abs_error <= threshold)
    hard_violation_count, maximum_hard_error_ratio = _hard_limit_statistics(
        abs_error, expected_cpu, standard
    )
    total = actual_cpu.numel()
    mismatch_count = int(mismatch.sum().item())
    matched_ratio = 1.0 if total == 0 else (total - mismatch_count) / total
    all_abs_error_finite = total == 0 or bool(torch.isfinite(abs_error).all())
    if total == 0:
        max_abs_error = 0.0
    elif all_abs_error_finite:
        max_abs_error = float(abs_error.max().item())
    else:
        max_abs_error = math.inf
    ratio_pass = matched_ratio >= standard["required_matched_ratio"]
    max_error_pass = hard_violation_count == 0
    passed = ratio_pass and max_error_pass

    tiny = torch.finfo(torch.float64).tiny
    relative_error = abs_error / expected_cpu.abs().double().clamp_min(tiny)
    error_ratio = abs_error / threshold.clamp_min(tiny)
    finite_pairs = torch.isfinite(actual_cpu) & torch.isfinite(expected_cpu)
    if total and bool(finite_pairs.all()):
        actual_flat = actual_cpu.double().flatten()
        expected_flat = expected_cpu.double().flatten()
        denom = torch.linalg.vector_norm(actual_flat) * torch.linalg.vector_norm(expected_flat)
        cosine = float(torch.dot(actual_flat, expected_flat) / denom) if denom > 0 else 1.0
        l2_norm = float(torch.linalg.vector_norm(actual_flat - expected_flat))
        rmse = float(torch.sqrt(torch.mean((actual_flat - expected_flat) ** 2)))
    else:
        cosine = math.nan
        l2_norm = math.nan
        rmse = math.nan

    if (global_table is None) == (segments_override is None):
        raise ValueError("provide exactly one of global_table or segments_override")
    segments = (
        [dict(segment) for segment in segments_override]
        if segments_override is not None
        else _build_segments(global_table or [])
    )
    expected_rows = segments[-1]["row_end"] if segments else 0
    if expected_rows != actual.shape[0]:
        raise ValueError(
            f"route table describes {expected_rows} valid rows but compared tensor has {actual.shape[0]} rows"
        )
    expert_stats, source_stats = _breakdown(abs_error, mismatch, segments)

    top_mismatches = []
    if total:
        top_count = min(max(1, int(topk)), total)
        score = abs_error.flatten() if all_abs_error_finite else torch.nan_to_num(
            abs_error.flatten(), nan=math.inf, posinf=math.inf, neginf=math.inf
        )
        top_values, top_indices = torch.topk(score, top_count)
        columns = actual.shape[1]
        for score_value, flat_index in zip(top_values.tolist(), top_indices.tolist()):
            row, column = divmod(int(flat_index), columns)
            local_expert, source_rank = _segment_for_row(segments, row)
            expected_value = float(expected_cpu[row, column])
            ulp, hard_limit = _hard_limit_at_scalar(expected_value, standard)
            hard_error_ratio = float(score_value) / hard_limit
            top_mismatches.append(
                {
                    "flat_index": int(flat_index),
                    "row": row,
                    "column": column,
                    "local_expert": local_expert,
                    "source_rank": source_rank,
                    "actual": _safe_float(actual_cpu[row, column]),
                    "expected": _safe_float(expected_value),
                    "abs_error": _safe_float(score_value),
                    "rel_error": _safe_float(relative_error[row, column]),
                    "error_ratio": _safe_float(error_ratio[row, column]),
                    "ulp_at_abs_expected": _safe_float(ulp),
                    "hard_limit": _safe_float(hard_limit),
                    "hard_error_ratio": _safe_float(hard_error_ratio),
                    "mixed_tolerance_pass": not bool(mismatch[row, column]),
                    "hard_limit_pass": bool(float(score_value) <= hard_limit),
                }
            )

    reasons = []
    if not ratio_pass:
        reasons.append("FAIL_MATCHED_RATIO")
    if not max_error_pass:
        reasons.append("FAIL_MAX_ABS_ERROR")
    return {
        "passed": passed,
        "failure_reasons": reasons,
        "precision_standard": standard,
        "shape": list(actual.shape),
        "elements": total,
        "matched_elements": total - mismatch_count,
        "mismatch_count": mismatch_count,
        "mismatch_ratio": mismatch_count / total if total else 0.0,
        "matched_ratio": matched_ratio,
        "ratio_pass": ratio_pass,
        "max_abs_error": _safe_float(max_abs_error),
        "max_abs_error_pass": max_error_pass,
        "hard_limit_violation_count": hard_violation_count,
        "maximum_hard_error_ratio": _safe_float(maximum_hard_error_ratio),
        "nan": {
            "actual": int(torch.isnan(actual_cpu).sum().item()),
            "expected": int(torch.isnan(expected_cpu).sum().item()),
        },
        "positive_inf": {
            "actual": int(torch.isposinf(actual_cpu).sum().item()),
            "expected": int(torch.isposinf(expected_cpu).sum().item()),
        },
        "negative_inf": {
            "actual": int(torch.isneginf(actual_cpu).sum().item()),
            "expected": int(torch.isneginf(expected_cpu).sum().item()),
        },
        "absolute_error": _quantiles(abs_error),
        "relative_error": _quantiles(relative_error),
        "rmse": _safe_float(rmse),
        "l2_norm": _safe_float(l2_norm),
        "cosine_similarity": _safe_float(cosine),
        "error_ratio_histogram": _error_ratio_histogram(error_ratio),
        "by_local_expert": expert_stats,
        "by_source_rank": source_stats,
        "segments": segments,
        "top_mismatches": top_mismatches,
        "heatmap": _heatmap(error_ratio, heatmap_bins),
    }


def _write_heatmap(path: Path, report: Mapping[str, Any]) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return "matplotlib is not installed; heatmap PNG was skipped"

    values = np.asarray(report["heatmap"]["values"], dtype=np.float64)
    display = np.log2(np.maximum(values, 2 ** -8))
    figure, axis = plt.subplots(figsize=(9, 6))
    image = axis.imshow(display, aspect="auto", interpolation="nearest", cmap="magma")
    axis.set_title("max log2(error_ratio) per output bucket; >0 exceeds mixed tolerance")
    axis.set_xlabel("output N bucket")
    axis.set_ylabel("packed token row bucket")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("log2(error_ratio)")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return None


def write_accuracy_artifacts(
    output_dir: Union[Path, str],
    report: Dict[str, Any],
    *,
    dump_payload: Optional[Mapping[str, Any]] = None,
    max_dump_bytes: int = 512 * 1024 * 1024,
) -> Dict[str, Any]:
    """Write JSON, Top-K CSV, heatmap and an optional bounded failure dump."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    heatmap_warning = _write_heatmap(output_path / "error_heatmap.png", report)
    if heatmap_warning:
        report["heatmap_warning"] = heatmap_warning

    with (output_path / "top_mismatches.csv").open("w", encoding="utf-8", newline="") as stream:
        rows = report["top_mismatches"]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["row"])
        writer.writeheader()
        writer.writerows(rows)

    if dump_payload is not None and not report["passed"]:
        import torch

        estimated_bytes = sum(
            value.numel() * value.element_size()
            for value in dump_payload.values()
            if isinstance(value, torch.Tensor)
        )
        if estimated_bytes <= max_dump_bytes:
            cpu_payload = {
                key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                for key, value in dump_payload.items()
            }
            torch.save(cpu_payload, output_path / "failure_dump.pt")
            report["failure_dump"] = "failure_dump.pt"
        else:
            report["failure_dump_skipped"] = {
                "estimated_bytes": estimated_bytes,
                "limit_bytes": max_dump_bytes,
            }

    with (output_path / "report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report
