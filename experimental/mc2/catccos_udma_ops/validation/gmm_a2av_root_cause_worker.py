# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Persistent one-process-per-NPU worker for GMM->AllToAllV root-cause tests."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu

import ops_transformer_catccos_udma as udma

try:
    from .accuracy_worker import (
        _grouped_mat_mul_all_to_allv_golden_with_local,
        _grouped_mat_mul_all_to_allv_producer_trace_golden,
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
    from .device_precision import analyze_accuracy_on_device
    from .memory_safety import compare_fingerprints, sampled_fingerprint
except ImportError:  # Direct torchrun execution.
    from accuracy_worker import (
        _grouped_mat_mul_all_to_allv_golden_with_local,
        _grouped_mat_mul_all_to_allv_producer_trace_golden,
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
    from device_precision import analyze_accuracy_on_device
    from memory_safety import compare_fingerprints, sampled_fingerprint


DIAGNOSTIC_TENSOR_WORDS = 131072
DIAGNOSTIC_HEADER_WORDS = 32
DIAGNOSTIC_RECORD_WORDS = 24
DIAGNOSTIC_MAGIC = 0x474D4D4132415631
DIAGNOSTIC_VERSION = 3
DIAGNOSTIC_MODE_IDS = {
    "baseline": 1,
    "producer_barrier": 2,
    "full_barrier": 3,
    "full_barrier_mte3": 4,
    "full_barrier_official_block": 5,
    "pre_get_delay_medium": 6,
    "post_quiet_delay_short": 7,
    "post_quiet_delay_medium": 7,
    "post_quiet_delay_long": 7,
    "no_reuse_trace": 8,
}
DIAGNOSTIC_DELAY_CYCLES = {
    "baseline": 0,
    "producer_barrier": 0,
    "full_barrier": 0,
    "full_barrier_mte3": 0,
    "full_barrier_official_block": 0,
    "pre_get_delay_medium": 20_000,
    "post_quiet_delay_short": 2_000,
    "post_quiet_delay_medium": 20_000,
    "post_quiet_delay_long": 200_000,
    "no_reuse_trace": 0,
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


def _parse_diagnostic_tensor(raw, expected_mode, world_size):
    torch_npu.npu.synchronize()
    words = [int(value) for value in raw.cpu().tolist()]
    if words[0] != DIAGNOSTIC_MAGIC:
        raise RuntimeError(
            "diagnostic tensor magic mismatch: 0x{:x}".format(words[0])
        )
    if words[1] != DIAGNOSTIC_VERSION:
        raise RuntimeError(
            "unsupported diagnostic tensor version: {}".format(words[1])
        )
    if words[2] != DIAGNOSTIC_MODE_IDS[expected_mode]:
        raise RuntimeError(
            "diagnostic mode mismatch: device={} host={}".format(
                words[2], DIAGNOSTIC_MODE_IDS[expected_mode]
            )
        )
    expected_delay = DIAGNOSTIC_DELAY_CYCLES[expected_mode]
    if words[20] != expected_delay:
        raise RuntimeError(
            "diagnostic delay mismatch: device={} host={}".format(
                words[20], expected_delay
            )
        )
    if words[4] != world_size:
        raise RuntimeError(
            "diagnostic world size mismatch: device={} host={}".format(
                words[4], world_size
            )
        )
    if words[14] != DIAGNOSTIC_RECORD_WORDS:
        raise RuntimeError(
            "diagnostic record ABI mismatch: device={} host={}".format(
                words[14], DIAGNOSTIC_RECORD_WORDS
            )
        )
    record_count = words[15]
    record_capacity = (
        len(words) - DIAGNOSTIC_HEADER_WORDS
    ) // DIAGNOSTIC_RECORD_WORDS
    if words[16] or record_count > record_capacity:
        raise RuntimeError(
            "diagnostic record buffer overflow: count={} capacity={}".format(
                record_count, record_capacity
            )
        )
    records = []
    for record_index in range(record_count):
        base = (
            DIAGNOSTIC_HEADER_WORDS +
            record_index * DIAGNOSTIC_RECORD_WORDS
        )
        values = words[base:base + DIAGNOSTIC_RECORD_WORDS]
        records.append({
            "comm_index": values[0],
            "source_rank": values[1],
            "stage_index": values[2],
            "diagnostic_mode_id": values[3],
            "has_input": bool(values[4]),
            "actual_rows": values[5],
            "output_row_offset": values[6],
            "udma_bytes": values[7],
            "quiet_completed": bool(values[8]),
            "completion_published": bool(values[9]),
            "producer_barrier": bool(values[10]),
            "recycle_barrier": bool(values[11]),
            "stage_released": bool(values[12]),
            "official_block_remote_copy": bool(values[13]),
            "delay_cycles": values[14],
            "delay_placement": {
                0: "none", 1: "pre_get", 2: "post_quiet",
            }.get(values[15], "unknown"),
            "stage_byte_offset": values[16],
            "stage_payload_bytes": values[17],
            "producer_trace_row_offset": values[18],
            "producer_trace_copy_completed": bool(values[19]),
        })
    completion_failures = [
        {
            "reason": "quiet_did_not_complete",
            "comm_index": item["comm_index"],
            "source_rank": item["source_rank"],
            "diagnostic_mode_id": item["diagnostic_mode_id"],
        }
        for item in records
        if item["has_input"] and not item["quiet_completed"]
    ]
    trace_copy_failures = [
        {
            "reason": "trace_copy_did_not_complete",
            "comm_index": item["comm_index"],
            "destination_rank": item["source_rank"],
        }
        for item in records
        if expected_mode == "no_reuse_trace"
        and item["stage_payload_bytes"] != 0
        and not item["producer_trace_copy_completed"]
    ]
    return {
        "header": {
            "magic": "0x{:016x}".format(words[0]),
            "version": words[1],
            "diagnostic_mode_id": words[2],
            "rank": words[3],
            "world_size": words[4],
            "aiv_block_num": words[5],
            "aiv_sub_block_num": words[6],
            "active_aiv_tasks": words[7],
            "comm_loops": words[8],
            "stage_rows": words[9],
            "workspace_stages": words[10],
            "n": words[11],
            "aic_block_num": words[12],
            "input_rows": words[13],
            "record_words": words[14],
            "record_count": record_count,
            "overflow": bool(words[16]),
            "symmetric_base_address": "0x{:x}".format(words[17]),
            "producer_trace_address": "0x{:x}".format(words[18]),
            "producer_trace_rows": words[19],
            "delay_cycles": words[20],
            "no_reuse_comm_loops_over_capacity": words[21],
        },
        "records": records,
        "completion_failures": completion_failures,
        "producer_trace_copy_failures": trace_copy_failures,
    }


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


def _exact_reuse_check(reference, current, *, chunk_elements):
    if chunk_elements <= 0:
        raise ValueError("chunk_elements must be positive")
    structure_equal = (
        tuple(reference.shape) == tuple(current.shape)
        and tuple(reference.stride()) == tuple(current.stride())
        and reference.dtype == current.dtype
        and reference.device == current.device
    )
    mismatch_count = None
    if structure_equal:
        reference_flat = reference.detach().reshape(-1)
        current_flat = current.detach().reshape(-1)
        mismatch_counts = []
        for start in range(0, current_flat.numel(), chunk_elements):
            stop = min(current_flat.numel(), start + chunk_elements)
            mismatch_counts.append(torch.count_nonzero(
                reference_flat[start:stop] != current_flat[start:stop]
            ))
        mismatch_count = (
            int(torch.stack(mismatch_counts).sum().item())
            if mismatch_counts else 0
        )
    values_equal = mismatch_count == 0
    return {
        "structure_equal": structure_equal,
        "values_equal": values_equal,
        "mismatch_count": mismatch_count,
        "reference_data_ptr": "0x{:x}".format(reference.data_ptr()),
        "current_data_ptr": "0x{:x}".format(current.data_ptr()),
        "elements_checked": current.numel() if structure_equal else 0,
        "passed": structure_equal and values_equal,
    }


def _reuse_check(
    reference,
    current,
    fingerprint_before,
    *,
    max_elements,
    chunk_elements,
):
    fingerprint_after = sampled_fingerprint(
        current, max_elements=max_elements
    )
    sampled = compare_fingerprints(fingerprint_before, fingerprint_after)
    exact = _exact_reuse_check(
        reference, current, chunk_elements=chunk_elements
    )
    return {
        "sampled_fingerprint": sampled,
        "full_tensor_exact": exact,
        "passed": sampled["passed"] and exact["passed"],
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
    parser.add_argument("--launches-per-mode", type=int, default=3)
    parser.add_argument("--fingerprint-elements", type=int, default=1 << 16)
    parser.add_argument("--comparison-chunk-elements", type=int, default=1 << 22)
    args = parser.parse_args()
    if args.launches_per_mode <= 0:
        raise ValueError("--launches-per-mode must be positive")
    if args.fingerprint_elements <= 0 or args.comparison_chunk_elements <= 0:
        raise ValueError("fingerprint and comparison chunk sizes must be positive")
    unknown_modes = sorted(set(args.mode) - set(DIAGNOSTIC_MODE_IDS))
    if unknown_modes:
        raise ValueError("unknown diagnostic modes: {}".format(unknown_modes))

    cases = read_resolved_cases(args.resolved_case)
    if len(cases) != 1:
        raise ValueError("root-cause worker requires exactly one resolved case")
    case = cases[0]
    if case["operator"] != "grouped_mat_mul_all_to_allv":
        raise ValueError("root-cause worker supports grouped_mat_mul_all_to_allv only")
    if case["route"] != "balanced":
        raise ValueError(
            "the per-round barrier matrix requires a balanced route with identical "
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
        if "no_reuse_trace" in args.mode:
            no_reuse_minimum_heap = 704 * 1024 * 1024
            if heap_size < no_reuse_minimum_heap:
                raise RuntimeError(
                    "no_reuse_trace requires UDMA_HEAP_SIZE >= {} bytes "
                    "(640 MiB staging plus signal/headroom); got {}".format(
                        no_reuse_minimum_heap, heap_size
                    )
                )
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
        resident_inputs = {
            "x": x,
            "weight": weight,
            "local_tokens_per_expert": local_table,
            "global_tokens_per_local_expert": global_table,
        }
        input_references = {
            name: tensor.detach().clone()
            for name, tensor in resident_inputs.items()
        }
        torch_npu.npu.synchronize()
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
        needs_producer_trace = "no_reuse_trace" in args.mode
        expected, local_gmm_expected = (
            _grouped_mat_mul_all_to_allv_golden_with_local(
                x, weight, route_matrix, rank, world_size, expert_num,
                include_local=needs_producer_trace,
            )
        )
        producer_trace_expected = None
        producer_trace_segments = None
        if needs_producer_trace:
            producer_trace_expected, producer_trace_segments = (
                _grouped_mat_mul_all_to_allv_producer_trace_golden(
                    local_gmm_expected, route_matrix, rank, world_size,
                    expert_num,
                )
            )
            del local_gmm_expected
        torch_npu.npu.synchronize()
        golden_seconds = time.monotonic() - golden_started
        golden_fingerprint = sampled_fingerprint(
            expected, max_elements=args.fingerprint_elements
        )
        producer_trace_golden_fingerprint = (
            sampled_fingerprint(
                producer_trace_expected,
                max_elements=args.fingerprint_elements,
            )
            if producer_trace_expected is not None else None
        )
        golden_reference = expected.detach().clone()
        torch_npu.npu.synchronize()
        golden_id = _identity({
            "case_id": case["id"],
            "rank": rank,
            "fingerprint": golden_fingerprint,
            "producer_trace_fingerprint":
                producer_trace_golden_fingerprint,
        })
        prelaunch_input_validation = {
            name: _reuse_check(
                input_references[name],
                tensor,
                fingerprints_before[name],
                max_elements=args.fingerprint_elements,
                chunk_elements=args.comparison_chunk_elements,
            )
            for name, tensor in resident_inputs.items()
        }
        prelaunch_validation_passed = all(
            item["passed"]
            for item in prelaunch_input_validation.values()
        )
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
                "producer_trace_golden": (
                    _tensor_metadata(producer_trace_expected)
                    if producer_trace_expected is not None else None
                ),
            },
            "input_fingerprints": fingerprints_before,
            "golden_fingerprint": golden_fingerprint,
            "producer_trace_golden_fingerprint":
                producer_trace_golden_fingerprint,
            "prelaunch_input_validation": prelaunch_input_validation,
            "prelaunch_validation_passed": prelaunch_validation_passed,
            "mode_order": args.mode,
            "reuse_contract": (
                "The same resident tensors and Golden are used by every mode "
                "and launch in this worker process."
            ),
            "mutation_validation": (
                "Full-tensor exact reference clones plus sampled SHA-256 "
                "fingerprints are checked throughout the matrix."
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
                "prelaunch_validation_passed": all(
                    item["prelaunch_validation_passed"]
                    for item in gathered_reuse
                ),
            })
        if not all(
            item["prelaunch_validation_passed"]
            for item in gathered_reuse
        ):
            raise RuntimeError(
                "input tensors changed while generating the shared Golden; "
                "see reuse_manifest.json"
            )

        completed_modes = []
        mode_group_results = []
        for mode in args.mode:
            diagnostic_mode = DIAGNOSTIC_MODE_IDS[mode]
            delay_cycles = DIAGNOSTIC_DELAY_CYCLES[mode]
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
            detailed_failure_written = False
            for launch_index in range(args.launches_per_mode):
                diagnostic_out = torch.empty_like(expected)
                diagnostic_out.fill_(float("nan"))
                diagnostic_status = torch.zeros(
                    DIAGNOSTIC_TENSOR_WORDS,
                    dtype=torch.int64,
                    device=device,
                )
                if mode == "no_reuse_trace":
                    producer_trace = torch.empty_like(
                        producer_trace_expected
                    )
                    producer_trace.fill_(float("nan"))
                else:
                    producer_trace = torch.empty(
                        (0, n), dtype=torch.bfloat16, device=device
                    )
                torch_npu.npu.synchronize()
                # Every rank starts the kernel only after all diagnostic
                # outputs contain NaNs. This keeps Host-side report cost from
                # shifting the following collective launch.
                dist.barrier()
                launch_started = time.monotonic()
                actual = (
                    torch.ops.ops_transformer_catccos_udma
                    ._grouped_mat_mul_all_to_allv_diagnostic(
                        x, weight, local_table, global_table,
                        world_size, expert_num, diagnostic_out,
                        diagnostic_status, producer_trace,
                        diagnostic_mode, delay_cycles,
                    )
                )
                torch_npu.npu.synchronize()
                elapsed = time.monotonic() - launch_started
                output_aliases_prefilled_buffer = (
                    actual.data_ptr() == diagnostic_out.data_ptr()
                )
                # Do not let a fast rank inspect output or completion records while a
                # slower rank is still executing the communication kernel.
                dist.barrier()
                diagnostic_report = _parse_diagnostic_tensor(
                    diagnostic_status, mode, world_size
                )
                completion_failures = diagnostic_report["completion_failures"]
                trace_copy_failures = diagnostic_report[
                    "producer_trace_copy_failures"
                ]
                no_reuse_capacity_valid = not diagnostic_report["header"][
                    "no_reuse_comm_loops_over_capacity"
                ]
                report = analyze_accuracy_on_device(
                    actual, expected, segments,
                    topk=case["accuracy"]["topk"],
                    chunk_elements=args.comparison_chunk_elements,
                    include_breakdown=False,
                )
                detailed_failure = False
                if not report["passed"] and not detailed_failure_written:
                    report = analyze_accuracy_on_device(
                        actual, expected, segments,
                        topk=case["accuracy"]["topk"],
                        chunk_elements=args.comparison_chunk_elements,
                        include_breakdown=True,
                    )
                    detailed_failure = True
                    detailed_failure_written = True
                launch_reuse_checks = {
                    name: _exact_reuse_check(
                        input_references[name],
                        tensor,
                        chunk_elements=args.comparison_chunk_elements,
                    )
                    for name, tensor in resident_inputs.items()
                }
                launch_reuse_checks["golden"] = _exact_reuse_check(
                    golden_reference,
                    expected,
                    chunk_elements=args.comparison_chunk_elements,
                )
                resident_tensors_unchanged = all(
                    item["passed"]
                    for item in launch_reuse_checks.values()
                )
                if not resident_tensors_unchanged:
                    report["passed"] = False
                    report["failure_reasons"].append(
                        "FAIL_RESIDENT_TENSOR_MUTATION"
                    )
                if not output_aliases_prefilled_buffer:
                    report["passed"] = False
                    report["failure_reasons"].append(
                        "FAIL_OUTPUT_ALIAS_CONTRACT"
                    )
                producer_report = None
                if mode == "no_reuse_trace":
                    producer_report = analyze_accuracy_on_device(
                        producer_trace, producer_trace_expected,
                        producer_trace_segments,
                        topk=case["accuracy"]["topk"],
                        chunk_elements=args.comparison_chunk_elements,
                        include_breakdown=True,
                    )
                    producer_report.update({
                        "case_id": case["id"],
                        "rank": rank,
                        "mode": mode,
                        "launch_index": launch_index,
                        "route_axis_role": (
                            "source_rank field denotes destination_rank for "
                            "producer staging"
                        ),
                        "copy_failures": trace_copy_failures,
                        "workspace_capacity_valid":
                            no_reuse_capacity_valid,
                        "producer_trace": _tensor_metadata(producer_trace),
                    })
                report.update({
                    "case_id": case["id"],
                    "rank": rank,
                    "mode": mode,
                    "launch_index": launch_index,
                    "elapsed_seconds": elapsed,
                    "dataset_id": dataset_id,
                    "golden_id": golden_id,
                    "completion_failures": completion_failures,
                    "completion_passed": not completion_failures,
                    "delay_cycles": delay_cycles,
                    "producer_trace_passed": (
                        producer_report["passed"]
                        if producer_report is not None else None
                    ),
                    "producer_trace_copy_failures": trace_copy_failures,
                    "no_reuse_workspace_capacity_valid":
                        no_reuse_capacity_valid,
                    "output": _tensor_metadata(actual),
                    "output_isolation": {
                        "allocation": "caller_provided_per_launch",
                        "prefill": "nan",
                        "returned_tensor_aliases_prefilled_buffer":
                            output_aliases_prefilled_buffer,
                        "purpose": (
                            "Expose any output element the kernel leaves "
                            "unwritten and prevent stale allocator contents "
                            "from satisfying accuracy checks."
                        ),
                    },
                    "resident_tensor_integrity": launch_reuse_checks,
                    "resident_tensors_unchanged":
                        resident_tensors_unchanged,
                    "detailed_failure_artifacts": detailed_failure,
                })
                launch_dir = (
                    rank_root / "modes" / mode /
                    "launch_{}".format(launch_index)
                )
                write_json(
                    launch_dir / "diagnostic.json", diagnostic_report
                )
                _write_report(
                    launch_dir, report,
                    actual=actual if detailed_failure else None,
                    expected=expected if detailed_failure else None,
                )
                if producer_report is not None:
                    producer_dir = launch_dir / "producer_trace"
                    _write_report(
                        producer_dir, producer_report,
                        actual=(
                            producer_trace
                            if not producer_report["passed"] else None
                        ),
                        expected=(
                            producer_trace_expected
                            if not producer_report["passed"] else None
                        ),
                    )
                    write_json(producer_dir / "layout.json", {
                        "symmetric_base_address": diagnostic_report[
                            "header"
                        ]["symmetric_base_address"],
                        "producer_trace_address": diagnostic_report[
                            "header"
                        ]["producer_trace_address"],
                        "workspace_stages": diagnostic_report[
                            "header"
                        ]["workspace_stages"],
                        "stage_rows": diagnostic_report[
                            "header"
                        ]["stage_rows"],
                        "records": [{
                            "comm_index": item["comm_index"],
                            "destination_rank": item["source_rank"],
                            "stage_index": item["stage_index"],
                            "stage_byte_offset": item[
                                "stage_byte_offset"
                            ],
                            "stage_payload_bytes": item[
                                "stage_payload_bytes"
                            ],
                            "trace_row_offset": item[
                                "producer_trace_row_offset"
                            ],
                            "trace_copy_completed": item[
                                "producer_trace_copy_completed"
                            ],
                        } for item in diagnostic_report["records"]],
                    })
                launch_results.append({
                    "launch_index": launch_index,
                    "final_output_passed": report["passed"],
                    "passed": (
                        report["passed"]
                        and resident_tensors_unchanged
                        and output_aliases_prefilled_buffer
                        and not trace_copy_failures
                        and no_reuse_capacity_valid
                        and (
                            producer_report is None or
                            producer_report["passed"]
                        )
                    ),
                    "matched_ratio": report["matched_ratio"],
                    "hard_limit_violation_count":
                        report["hard_limit_violation_count"],
                    "max_abs_error": report["max_abs_error"],
                    "actual_nonfinite": report["actual_nonfinite"],
                    "resident_tensors_unchanged":
                        resident_tensors_unchanged,
                    "completion_failures": completion_failures,
                    "delay_cycles": delay_cycles,
                    "producer_trace_passed": (
                        producer_report["passed"]
                        if producer_report is not None else None
                    ),
                    "producer_trace_report": (
                        str(launch_dir / "producer_trace" / "report.json")
                        if producer_report is not None else None
                    ),
                    "diagnostic": str(launch_dir / "diagnostic.json"),
                    "diagnostic_header": diagnostic_report["header"],
                    "report": str(launch_dir / "report.json"),
                })
                del actual, diagnostic_out, diagnostic_status, producer_trace
                gc.collect()
                # Reporting and failure breakdowns take different time on
                # different ranks. Align them before the next launch.
                dist.barrier()

            mutation_checks = {
                name: _reuse_check(
                    input_references[name],
                    resident_inputs[name],
                    fingerprints_before[name],
                    max_elements=args.fingerprint_elements,
                    chunk_elements=args.comparison_chunk_elements,
                )
                for name in fingerprints_before
            }
            mutation_checks["golden"] = _reuse_check(
                golden_reference,
                expected,
                golden_fingerprint,
                max_elements=args.fingerprint_elements,
                chunk_elements=args.comparison_chunk_elements,
            )
            rank_mode = {
                "case_id": case["id"],
                "rank": rank,
                "mode": mode,
                "passed": (
                    all(item["passed"] for item in launch_results)
                    and all(not item["completion_failures"] for item in launch_results)
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
                "completion_failure_count": sum(
                    len(launch["completion_failures"])
                    for item in gathered_modes
                    for launch in item["launches"]
                ),
                "final_output_passed": all(
                    launch["final_output_passed"]
                    for item in gathered_modes
                    for launch in item["launches"]
                ),
                "producer_trace_passed": (
                    all(
                        launch["producer_trace_passed"] is True
                        for item in gathered_modes
                        for launch in item["launches"]
                    )
                    if mode == "no_reuse_trace" else None
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
                    "failed_ranks={} completion_failures={}".format(
                        mode, group_mode["passed"],
                        group_mode["failed_ranks"],
                        group_mode["completion_failure_count"],
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
