# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Persistent one-process-per-NPU worker for GMM->AllToAllV root-cause tests."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu

import ops_transformer_catccos_udma as udma

try:
    from .accuracy_worker import (
        _grouped_mat_mul_all_to_allv_golden,
        _grouped_mat_mul_all_to_allv_segments,
        _make_tensor,
        _rank_route_values,
        _route_matrix,
        _tensor_metadata,
    )
    from .case_schema import read_resolved_cases
    from .common import (
        udma_init_method_from_env,
        validation_udma_heap_size,
        write_json,
    )
    from .memory_safety import compare_fingerprints, sampled_fingerprint
    from .precision import _hard_limit_at_scalar, precision_standard
except ImportError:  # Direct torchrun execution.
    from accuracy_worker import (
        _grouped_mat_mul_all_to_allv_golden,
        _grouped_mat_mul_all_to_allv_segments,
        _make_tensor,
        _rank_route_values,
        _route_matrix,
        _tensor_metadata,
    )
    from case_schema import read_resolved_cases
    from common import (
        udma_init_method_from_env,
        validation_udma_heap_size,
        write_json,
    )
    from memory_safety import compare_fingerprints, sampled_fingerprint
    from precision import _hard_limit_at_scalar, precision_standard


STATUS_CACHE_LINE_BYTES = 64


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
    mixed_limit = (
        standard["atol"] + standard["rtol"] * expected_fp32.abs()
    )
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


def _device_accuracy_report(
    actual,
    expected,
    segments,
    *,
    topk,
    chunk_elements,
    include_breakdown,
):
    """Apply the opbase BF16 criteria with bounded NPU working memory."""
    if tuple(actual.shape) != tuple(expected.shape):
        raise ValueError(
            "shape mismatch: actual={} expected={}".format(
                tuple(actual.shape), tuple(expected.shape)
            )
        )
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
                absolute_error, nan=math.inf, posinf=math.inf, neginf=math.inf
            ).max()
        )
        current_topk = min(topk, stop - start)
        values, indices = torch.topk(
            torch.nan_to_num(
                absolute_error, nan=math.inf, posinf=math.inf, neginf=math.inf
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
        "by_source_rank": _aggregate_breakdown(
            by_segment, "source_rank"
        ) if by_segment else [],
        "by_local_expert": _aggregate_breakdown(
            by_segment, "local_expert"
        ) if by_segment else [],
        "comparison_device": str(actual.device),
        "comparison_strategy": (
            "chunked NPU opbase reductions; only Top-K and scalar summaries "
            "are copied to Host"
        ),
    }


def _write_report(output_dir, report, actual=None, expected=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "report.json", report)
    for filename, rows, empty_field in (
        ("top_mismatches.csv", report["top_mismatches"], "row"),
        ("by_route_segment.csv", report["by_route_segment"], "row_start"),
        ("by_source_rank.csv", report["by_source_rank"], "source_rank"),
        ("by_local_expert.csv", report["by_local_expert"], "local_expert"),
    ):
        with (output_dir / filename).open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(rows[0]) if rows else [empty_field],
            )
            writer.writeheader()
            writer.writerows(rows)
    if not report["passed"] and actual is not None and expected is not None:
        selected_rows = sorted({
            item["row"] for item in report["top_mismatches"]
        })
        if selected_rows:
            row_index = torch.tensor(
                selected_rows, dtype=torch.long, device=actual.device
            )
            torch.save({
                "selected_rows": selected_rows,
                "actual_selected_rows": actual.detach().index_select(
                    0, row_index
                ).cpu(),
                "expected_selected_rows": expected.detach().index_select(
                    0, row_index
                ).cpu(),
                "top_mismatches": report["top_mismatches"],
            }, output_dir / "failure_rows.pt")


def _read_cq_status(device_anchor, world_size):
    raw = (
        torch.ops.ops_transformer_catccos_udma
        ._grouped_mat_mul_all_to_allv_diagnostic_status(device_anchor)
    )
    torch_npu.npu.synchronize()
    payload = raw.cpu().contiguous().numpy().tobytes()
    records = []
    for source_rank in range(world_size):
        status, comm_plus_one, source_plus_one, mode = struct.unpack_from(
            "<IIII", payload, source_rank * STATUS_CACHE_LINE_BYTES
        )
        if status:
            records.append({
                "status": status,
                "comm_index": comm_plus_one - 1,
                "source_rank": source_plus_one - 1,
                "diagnostic_mode_id": mode,
            })
    return records


def _fingerprints(x, weight, local_table, global_table, max_elements):
    return {
        "x": sampled_fingerprint(x, max_elements=max_elements),
        "weight": sampled_fingerprint(weight, max_elements=max_elements),
        "local_tokens_per_expert": sampled_fingerprint(
            local_table, max_elements=max_elements
        ),
        "global_tokens_per_local_expert": sampled_fingerprint(
            global_table, max_elements=max_elements
        ),
    }


def _identity(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _progress(path, payload):
    payload = dict(payload)
    payload["updated_unix"] = time.time()
    write_json(path, payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolved-case", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", action="append", required=True)
    parser.add_argument("--launches-per-mode", type=int, default=2)
    parser.add_argument("--fingerprint-elements", type=int, default=1 << 16)
    parser.add_argument("--comparison-chunk-elements", type=int, default=1 << 22)
    args = parser.parse_args()
    if args.launches_per_mode <= 0:
        raise ValueError("--launches-per-mode must be positive")
    if args.fingerprint_elements <= 0 or args.comparison_chunk_elements <= 0:
        raise ValueError("fingerprint and comparison chunk sizes must be positive")

    cases = read_resolved_cases(args.resolved_case)
    if len(cases) != 1:
        raise ValueError("root-cause worker requires exactly one resolved case")
    case = cases[0]
    if case["operator"] != "grouped_mat_mul_all_to_allv":
        raise ValueError("root-cause worker supports grouped_mat_mul_all_to_allv only")
    if case["route"] != "balanced":
        raise ValueError(
            "official_barrier_exact requires a balanced route with identical "
            "communication-loop counts on every rank"
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if case["world_size"] != world_size:
        raise ValueError("case world_size does not match torchrun")
    output_root = Path(args.output_dir).resolve()
    rank_root = output_root / "rank_{}".format(rank)
    progress_path = output_root / "progress.json"
    device = "npu:{}".format(local_rank)
    initialized_dist = False
    initialized_udma = False
    try:
        torch_npu.npu.set_device(local_rank)
        if not dist.is_gloo_available():
            raise RuntimeError("Gloo is required for persistent diagnostics")
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        initialized_dist = True
        dist.barrier()
        if rank == 0:
            _progress(progress_path, {
                "phase": "udma_initialization",
                "mode_order": args.mode,
                "completed_modes": [],
            })
        init_method = udma_init_method_from_env()
        heap_size = validation_udma_heap_size()
        udma.init_udma_group(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            init_method=init_method,
            heap_size=heap_size,
        )
        initialized_udma = True

        if rank == 0:
            _progress(progress_path, {
                "phase": "input_generation",
                "mode_order": args.mode,
                "completed_modes": [],
            })
        shape = case["shape"]
        m, n, k = shape["m"], shape["n"], shape["k"]
        expert_num = case["expert_num"]
        route_matrix = _route_matrix(case)
        local_values, global_values, input_rows = _rank_route_values(
            route_matrix, rank, world_size, expert_num
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(case["seed"] + rank)
        x = _make_tensor(
            (input_rows, k), case["input_distribution"], device, generator
        )
        weight = _make_tensor(
            (expert_num // world_size, k, n),
            case["input_distribution"], device, generator,
        )
        local_table = torch.tensor(
            local_values, dtype=torch.int64, device=device
        )
        global_table = torch.tensor(
            global_values, dtype=torch.int64, device=device
        )
        torch_npu.npu.synchronize()
        fingerprints_before = _fingerprints(
            x, weight, local_table, global_table,
            args.fingerprint_elements,
        )
        dataset_id = _identity({
            "case_id": case["id"],
            "rank": rank,
            "fingerprints": fingerprints_before,
        })

        if rank == 0:
            _progress(progress_path, {
                "phase": "golden_generation",
                "mode_order": args.mode,
                "completed_modes": [],
            })
        golden_started = time.monotonic()
        expected = _grouped_mat_mul_all_to_allv_golden(
            x, weight, route_matrix, rank, world_size, expert_num
        )
        torch_npu.npu.synchronize()
        golden_seconds = time.monotonic() - golden_started
        golden_fingerprint = sampled_fingerprint(
            expected, max_elements=args.fingerprint_elements
        )
        golden_id = _identity({
            "case_id": case["id"],
            "rank": rank,
            "fingerprint": golden_fingerprint,
        })
        segments = _grouped_mat_mul_all_to_allv_segments(
            route_matrix, rank, world_size, expert_num
        )
        reuse_manifest = {
            "case_id": case["id"],
            "rank": rank,
            "pid": os.getpid(),
            "input_generation_count": 1,
            "golden_generation_count": 1,
            "golden_generation_seconds": golden_seconds,
            "dataset_id": dataset_id,
            "golden_id": golden_id,
            "tensor_metadata": {
                "x": _tensor_metadata(x),
                "weight": _tensor_metadata(weight),
                "local_tokens_per_expert": _tensor_metadata(local_table),
                "global_tokens_per_local_expert": _tensor_metadata(global_table),
                "golden": _tensor_metadata(expected),
            },
            "input_fingerprints": fingerprints_before,
            "golden_fingerprint": golden_fingerprint,
            "mode_order": args.mode,
            "reuse_contract": (
                "The same resident tensors and Golden are used by every mode "
                "and launch in this worker process."
            ),
        }
        write_json(rank_root / "reuse_manifest.json", reuse_manifest)
        gathered_reuse = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_reuse, reuse_manifest)
        if rank == 0:
            write_json(output_root / "reuse_manifest.json", {
                "case_id": case["id"],
                "input_generation_count_per_rank": 1,
                "golden_generation_count_per_rank": 1,
                "ranks": gathered_reuse,
                "group_dataset_id": _identity([
                    item["dataset_id"] for item in gathered_reuse
                ]),
                "group_golden_id": _identity([
                    item["golden_id"] for item in gathered_reuse
                ]),
            })

        completed_modes = []
        mode_group_results = []
        for mode in args.mode:
            os.environ["CATCCOS_UDMA_GMM_A2AV_DIAGNOSTIC_MODE"] = mode
            torch_npu.npu.synchronize()
            dist.barrier()
            if rank == 0:
                _progress(progress_path, {
                    "phase": "mode_started",
                    "active_mode": mode,
                    "active_mode_started_unix": time.time(),
                    "mode_order": args.mode,
                    "completed_modes": completed_modes,
                    "modes": mode_group_results,
                })
            dist.barrier()

            launch_results = []
            for launch_index in range(args.launches_per_mode):
                launch_started = time.monotonic()
                actual = udma.grouped_mat_mul_all_to_allv(
                    x, weight, local_table, global_table,
                    ep_size=world_size, expert_num=expert_num,
                )
                torch_npu.npu.synchronize()
                elapsed = time.monotonic() - launch_started
                cq_failures = (
                    [] if mode == "production"
                    else _read_cq_status(x, world_size)
                )
                report = _device_accuracy_report(
                    actual, expected, segments,
                    topk=case["accuracy"]["topk"],
                    chunk_elements=args.comparison_chunk_elements,
                    include_breakdown=True,
                )
                report.update({
                    "case_id": case["id"],
                    "rank": rank,
                    "mode": mode,
                    "launch_index": launch_index,
                    "elapsed_seconds": elapsed,
                    "dataset_id": dataset_id,
                    "golden_id": golden_id,
                    "cq_failures": cq_failures,
                    "cq_status_passed": not cq_failures,
                    "output": _tensor_metadata(actual),
                })
                launch_dir = (
                    rank_root / "modes" / mode /
                    "launch_{}".format(launch_index)
                )
                _write_report(
                    launch_dir, report,
                    actual=actual,
                    expected=expected,
                )
                launch_results.append({
                    "launch_index": launch_index,
                    "passed": report["passed"],
                    "matched_ratio": report["matched_ratio"],
                    "hard_limit_violation_count":
                        report["hard_limit_violation_count"],
                    "max_abs_error": report["max_abs_error"],
                    "actual_nonfinite": report["actual_nonfinite"],
                    "cq_failures": cq_failures,
                    "report": str(launch_dir / "report.json"),
                })
                del actual
                gc.collect()

            fingerprints_after = _fingerprints(
                x, weight, local_table, global_table,
                args.fingerprint_elements,
            )
            mutation_checks = {
                name: compare_fingerprints(
                    fingerprints_before[name], fingerprints_after[name]
                )
                for name in fingerprints_before
            }
            mutation_checks["golden"] = compare_fingerprints(
                golden_fingerprint,
                sampled_fingerprint(
                    expected, max_elements=args.fingerprint_elements
                ),
            )
            rank_mode = {
                "case_id": case["id"],
                "rank": rank,
                "mode": mode,
                "passed": (
                    all(item["passed"] for item in launch_results)
                    and all(not item["cq_failures"] for item in launch_results)
                    and all(item["passed"] for item in mutation_checks.values())
                ),
                "launches": launch_results,
                "input_mutation": mutation_checks,
                "dataset_id": dataset_id,
                "golden_id": golden_id,
            }
            write_json(rank_root / "modes" / mode / "summary.json", rank_mode)
            gathered_modes = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_modes, rank_mode)
            group_mode = {
                "mode": mode,
                "passed": all(item["passed"] for item in gathered_modes),
                "ranks": gathered_modes,
                "failed_ranks": [
                    item["rank"] for item in gathered_modes
                    if not item["passed"]
                ],
                "cq_failure_count": sum(
                    len(launch["cq_failures"])
                    for item in gathered_modes
                    for launch in item["launches"]
                ),
            }
            if rank == 0:
                write_json(
                    output_root / "modes" / mode / "summary.json",
                    group_mode,
                )
                completed_modes.append(mode)
                mode_group_results.append(group_mode)
                _progress(progress_path, {
                    "phase": "mode_completed",
                    "active_mode": None,
                    "mode_order": args.mode,
                    "completed_modes": completed_modes,
                    "modes": mode_group_results,
                })
                print(
                    "[gmm-a2av-root-cause] mode={} passed={} "
                    "failed_ranks={} cq_failures={}".format(
                        mode, group_mode["passed"],
                        group_mode["failed_ranks"],
                        group_mode["cq_failure_count"],
                    ),
                    flush=True,
                )
            dist.barrier()

        if rank == 0:
            _progress(progress_path, {
                "phase": "worker_completed",
                "active_mode": None,
                "mode_order": args.mode,
                "completed_modes": completed_modes,
                "modes": mode_group_results,
            })
        return 0
    finally:
        if initialized_dist:
            dist.destroy_process_group()
        gc.collect()
        if initialized_udma:
            udma.destroy_udma_group()


if __name__ == "__main__":
    sys.exit(main())
