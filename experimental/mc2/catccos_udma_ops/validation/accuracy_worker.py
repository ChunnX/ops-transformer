# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""One-process-per-NPU accuracy worker for resolved validation cases."""

from __future__ import annotations

import argparse
import gc
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
    from .case_schema import read_resolved_cases
    from .common import (
        udma_init_method_from_env,
        validation_udma_heap_size,
        write_json,
    )
    from .precision import analyze_accuracy, write_accuracy_artifacts
    from .memory_safety import (
        GuardedTensor,
        compare_fingerprints,
        nonfinite_summary,
        poison_allocatable_hbm,
        sampled_fingerprint,
    )
except ImportError:  # Direct execution keeps the source tree out of package lookup.
    from case_schema import read_resolved_cases
    from common import (
        udma_init_method_from_env,
        validation_udma_heap_size,
        write_json,
    )
    from precision import analyze_accuracy, write_accuracy_artifacts
    from memory_safety import (
        GuardedTensor,
        compare_fingerprints,
        nonfinite_summary,
        poison_allocatable_hbm,
        sampled_fingerprint,
    )


def _make_tensor(shape, distribution, device, generator, out=None):
    kind = distribution["kind"]
    if kind == "uniform":
        result = torch.empty(shape, dtype=torch.float32)
        result.uniform_(distribution["low"], distribution["high"], generator=generator)
    elif kind == "normal":
        result = torch.empty(shape, dtype=torch.float32)
        result.normal_(distribution["mean"], distribution["std"], generator=generator)
    else:
        raise ValueError("unsupported distribution {}".format(kind))
    result = result.to(dtype=torch.bfloat16)
    if out is not None:
        out.copy_(result)
        return out
    return result.to(device=device)


def _route_matrix(case):
    m = case["shape"]["m"]
    world_size = case["world_size"]
    expert_num = case["expert_num"]
    if case["route"] == "balanced":
        return udma.build_global_route_matrix(m, world_size, expert_num)

    # Deterministic adversarial route: every source sends at least one token to
    # every destination rank, then concentrates the residue on rank 0. This
    # keeps all ranks active while creating different receive-row counts and
    # many zero-token experts.
    if m < world_size:
        raise ValueError(
            "unbalanced validation route requires shape.m >= world_size"
        )
    local_experts = expert_num // world_size
    matrix = []
    for source_rank in range(world_size):
        row = [0] * expert_num
        for destination_rank in range(world_size):
            local_expert = (source_rank + destination_rank) % local_experts
            row[destination_rank * local_experts + local_expert] += 1
        hot_local_expert = source_rank % local_experts
        row[hot_local_expert] += m - world_size
        matrix.append(row)
    return matrix


def _rank_route_values(route_matrix, rank, world_size, expert_num):
    local_experts = expert_num // world_size
    local = list(route_matrix[rank])
    global_for_local = [
        route_matrix[source_rank][rank * local_experts + local_expert]
        for source_rank in range(world_size)
        for local_expert in range(local_experts)
    ]
    return local, global_for_local, sum(global_for_local)


def _guarded_tensor(shape, dtype, device, name, memory_options, *, output=False):
    return GuardedTensor.allocate(
        shape,
        dtype=dtype,
        device=device,
        name=name,
        guard_bytes=memory_options["guard_bytes"],
        guard_kind="exact" if output else "nan",
        guard_value=-12352.0 if output else None,
    )


def _prepare_memory_safety_launch(output_guards):
    # All ranks must complete the diagnostic memset before any peer begins to
    # consume another rank's symmetric workspace.
    udma.poison_udma_workspace_for_testing()
    for guarded in output_guards:
        guarded.reset_guards()
        guarded.fill_payload(float("nan"))
    torch_npu.npu.synchronize()
    dist.barrier()


def _golden(x, weight, route_matrix, rank, world_size, expert_num):
    # Keep the validation control plane on the host.  Initializing an HCCL
    # communicator after ACLSHMEM has claimed the UDMA/HDC resources can fail,
    # and an HCCL collective would make the Golden depend on a second device
    # communication implementation.  Gloo gathers FP32 CPU inputs instead.
    x_cpu = x.detach().float().cpu()
    weight_cpu = weight.detach().float().cpu()
    gathered = [torch.empty_like(x_cpu) for _ in range(world_size)]
    dist.all_gather(gathered, x_cpu)
    local_experts = expert_num // world_size
    expert_outputs = []
    for local_expert in range(local_experts):
        global_expert = rank * local_experts + local_expert
        pieces = []
        for source_rank, source_x in enumerate(gathered):
            counts = route_matrix[source_rank]
            start = sum(counts[:global_expert])
            end = start + counts[global_expert]
            pieces.append(source_x[start:end])
        expert_input = torch.cat(pieces, dim=0)
        expert_outputs.append(expert_input.matmul(weight_cpu[local_expert]))
    return torch.cat(expert_outputs, dim=0).to(dtype=torch.bfloat16, device=x.device)


def _all_to_allv_golden(x, route_matrix, rank, world_size, expert_num):
    """Return the activation rows consumed by local experts."""
    x_cpu = x.detach().float().cpu()
    gathered = [torch.empty_like(x_cpu) for _ in range(world_size)]
    dist.all_gather(gathered, x_cpu)
    local_experts = expert_num // world_size
    pieces = []
    for local_expert in range(local_experts):
        global_expert = rank * local_experts + local_expert
        for source_rank, source_x in enumerate(gathered):
            counts = route_matrix[source_rank]
            start = sum(counts[:global_expert])
            pieces.append(source_x[start:start + counts[global_expert]])
    return torch.cat(pieces, dim=0).to(dtype=torch.bfloat16, device=x.device)


def _grouped_mat_mul_all_to_allv_golden(
    x, weight, route_matrix, rank, world_size, expert_num
):
    expected, _ = _grouped_mat_mul_all_to_allv_golden_with_local(
        x, weight, route_matrix, rank, world_size, expert_num,
        include_local=False,
    )
    return expected


def _grouped_mat_mul_all_to_allv_golden_with_local(
    x, weight, route_matrix, rank, world_size, expert_num, *,
    include_local=True,
):
    """Compute local GMM in FP32, then route its pieces through CPU Gloo."""
    x_cpu = x.detach().float().cpu()
    weight_cpu = weight.detach().float().cpu()
    local_experts = expert_num // world_size
    local_outputs = []
    offset = 0
    for local_expert in range(local_experts):
        global_expert = rank * local_experts + local_expert
        rows = sum(route_matrix[source][global_expert] for source in range(world_size))
        local_outputs.append(
            x_cpu[offset:offset + rows].matmul(weight_cpu[local_expert])
        )
        offset += rows
    if offset != x_cpu.shape[0]:
        raise RuntimeError("route matrix does not describe the grouped input rows")
    local_output = torch.cat(local_outputs, dim=0)
    expert_rank_rows = [
        sum(
            route_matrix[source_rank][global_expert]
            for source_rank in range(world_size)
            for global_expert in range(
                expert_rank * local_experts,
                (expert_rank + 1) * local_experts,
            )
        )
        for expert_rank in range(world_size)
    ]
    if local_output.shape[0] != expert_rank_rows[rank]:
        raise RuntimeError("local GMM rows do not match the route matrix")
    max_rows = max(expert_rank_rows)
    padded = torch.zeros((max_rows, local_output.shape[1]), dtype=local_output.dtype)
    padded[:local_output.shape[0]].copy_(local_output)
    gathered_padded = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered_padded, padded)
    gathered = [
        gathered_padded[expert_rank][:expert_rank_rows[expert_rank]]
        for expert_rank in range(world_size)
    ]

    pieces = []
    for expert_rank, expert_output in enumerate(gathered):
        expert_offset = 0
        for local_expert in range(local_experts):
            global_expert = expert_rank * local_experts + local_expert
            counts = [route_matrix[source][global_expert] for source in range(world_size)]
            start = expert_offset + sum(counts[:rank])
            pieces.append(expert_output[start:start + counts[rank]])
            expert_offset += sum(counts)
    expected = torch.cat(pieces, dim=0)
    local_result = (
        local_output.to(dtype=torch.bfloat16, device=x.device)
        if include_local else None
    )
    return expected.to(dtype=torch.bfloat16, device=x.device), local_result


def _grouped_mat_mul_all_to_allv_producer_trace_golden(
    local_output, route_matrix, rank, world_size, expert_num
):
    """Repack local GMM output into destination-major staging order."""
    local_experts = expert_num // world_size
    expert_bases = []
    offset = 0
    for local_expert in range(local_experts):
        global_expert = rank * local_experts + local_expert
        expert_bases.append(offset)
        offset += sum(
            route_matrix[source][global_expert]
            for source in range(world_size)
        )
    if offset != local_output.shape[0]:
        raise RuntimeError("local GMM rows do not match producer trace routes")

    pieces = []
    segments = []
    trace_offset = 0
    for destination_rank in range(world_size):
        for local_expert in range(local_experts):
            global_expert = rank * local_experts + local_expert
            counts = [
                route_matrix[source][global_expert]
                for source in range(world_size)
            ]
            rows = int(counts[destination_rank])
            start = expert_bases[local_expert] + sum(
                counts[:destination_rank]
            )
            pieces.append(local_output[start:start + rows])
            segments.append({
                "local_expert": local_expert,
                "global_expert": global_expert,
                # The generic precision reporter groups this field. In a
                # producer trace it denotes the destination, not the source.
                "source_rank": destination_rank,
                "destination_rank": destination_rank,
                "row_start": trace_offset,
                "row_end": trace_offset + rows,
                "rows": rows,
            })
            trace_offset += rows
    trace = torch.cat(pieces, dim=0)
    return trace, segments


def _grouped_mat_mul_all_to_allv_segments(
    route_matrix, rank, world_size, expert_num
):
    local_experts = expert_num // world_size
    segments = []
    offset = 0
    for expert_rank in range(world_size):
        for local_expert in range(local_experts):
            global_expert = expert_rank * local_experts + local_expert
            rows = int(route_matrix[rank][global_expert])
            segments.append({
                "local_expert": global_expert,
                "global_expert": global_expert,
                "source_rank": expert_rank,
                "row_start": offset,
                "row_end": offset + rows,
                "rows": rows,
            })
            offset += rows
    return segments


def _padding_stats(tensor, valid_rows):
    padding = tensor[valid_rows:].detach().float()
    if padding.numel() == 0:
        return {"elements": 0, "nonzero_count": 0, "max_abs": 0.0}
    return {
        "elements": padding.numel(),
        "nonzero_count": int(torch.count_nonzero(padding).item()),
        "max_abs": float(padding.abs().max().item()),
    }


def _tensor_metadata(tensor):
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "contiguous": bool(tensor.is_contiguous()),
        "numel": tensor.numel(),
        "element_size": tensor.element_size(),
        "data_ptr": "0x{:x}".format(tensor.data_ptr()),
    }


def _report_brief(report):
    return {
        "path": report["path"],
        "launch_index": report.get("launch_index"),
        "global_launch_index": report.get("global_launch_index"),
        "cross_rank_aligned": report.get("cross_rank_aligned", False),
        "allocation_mode": report.get("allocation_mode"),
        "output_role": report.get("output_role"),
        "passed": report["passed"],
        "matched_ratio": report["matched_ratio"],
        "max_abs_error": report["max_abs_error"],
        "failure_reasons": report["failure_reasons"],
        "actual_nonfinite": report.get("actual_nonfinite", {}).get(
            "total_nonfinite", 0
        ),
        "memory_safety_passed": report.get("memory_safety", {}).get("passed"),
    }


def _failure_scope(reports):
    """Classify a failure without replacing the detailed per-path reports."""
    by_path = {report["path"]: report for report in reports}
    first_output = by_path.get("export_output") or by_path.get("ordinary")
    first_export = by_path.get("all_to_allv_export")
    auto_outputs = [
        report for path, report in by_path.items()
        if path.startswith("auto_repeat_")
    ]
    out_outputs = [
        report for path, report in by_path.items()
        if path.startswith("out_repeat_")
    ]
    repeat_exports = [
        report for path, report in by_path.items()
        if path.startswith("all_to_allv_export_") and path != "all_to_allv_export"
    ]
    failed_auto = [report["path"] for report in auto_outputs if not report["passed"]]
    failed_out = [report["path"] for report in out_outputs if not report["passed"]]
    failed_exports = [
        report["path"] for report in repeat_exports if not report["passed"]
    ]

    if first_output is not None and not first_output["passed"]:
        likely_scope = "base_kernel_or_first_collective"
    elif first_export is not None and not first_export["passed"]:
        likely_scope = "first_alltoallv_transfer_or_export_materialization"
    elif failed_exports:
        likely_scope = "alltoallv_transfer_export_or_stage_materialization"
    elif failed_auto:
        failed_reports = [by_path[path] for path in failed_auto]
        likely_scope = (
            "cross_invocation_stage_readiness_or_workspace_reuse"
            if all(
                report.get("cross_rank_aligned", False)
                for report in failed_reports
            )
            else "cross_rank_launch_skew_or_cross_invocation_state"
        )
    elif failed_out:
        likely_scope = "caller_provided_out_path"
    else:
        likely_scope = "none"
    return {
        "likely_scope": likely_scope,
        "first_output_passed": first_output is None or first_output["passed"],
        "first_export_passed": first_export is None or first_export["passed"],
        "failed_auto_outputs": failed_auto,
        "failed_caller_out_outputs": failed_out,
        "failed_repeat_exports": failed_exports,
        "note": (
            "This is an evidence-based triage hint; report.json and compact "
            "failure dumps remain authoritative."
        ),
    }


def _run_case(
    case, rank, world_size, device, output_root, memory_options,
    mssanitizer_options, diagnostic_options,
):
    case_started = time.perf_counter()
    timing_events = []

    def add_timing(phase, started, *, path=None, launch_index=None, **details):
        event = {
            "phase": phase,
            "elapsed_seconds": time.perf_counter() - started,
        }
        if path is not None:
            event["path"] = path
        if launch_index is not None:
            event["launch_index"] = launch_index
        event.update(details)
        timing_events.append(event)
        return event

    input_preparation_started = time.perf_counter()
    shape = case["shape"]
    m, n, k = shape["m"], shape["n"], shape["k"]
    expert_num = case["expert_num"]
    route_matrix = _route_matrix(case)
    local_values, global_values, received_rows = _rank_route_values(
        route_matrix, rank, world_size, expert_num
    )
    input_rows = (
        received_rows
        if case["operator"] == "grouped_mat_mul_all_to_allv"
        else m
    )
    memory_enabled = memory_options["enabled"]
    mssanitizer_enabled = mssanitizer_options["enabled"]
    automatic_only = diagnostic_options["automatic_only"]
    synchronize_launches = not automatic_only
    check_reusable_out = diagnostic_options["check_reusable_out"]
    max_failure_dumps_per_role = diagnostic_options[
        "max_failure_dumps_per_role"
    ]
    if automatic_only and (
        case["operator"] != "all_to_allv_grouped_mat_mul"
        or not case["output_all_to_allv"]
    ):
        raise ValueError(
            "automatic-only diagnostics require all_to_allv_grouped_mat_mul "
            "with output_all_to_allv enabled"
        )
    memory_report = None
    if memory_enabled:
        memory_report = {
            "enabled": True,
            "hbm_poison": poison_allocatable_hbm(
                device,
                reserve_bytes=memory_options["hbm_reserve_bytes"],
                chunk_bytes=memory_options["hbm_poison_chunk_bytes"],
            ),
            "workspace_poison_byte": 255,
            "workspace_poison_data_bytes": 200 * 1024 * 1024 * 2,
            "launches": [],
        }
        dist.barrier()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(case["seed"] + rank)
    guarded_inputs = []
    if memory_enabled:
        x_guard = _guarded_tensor(
            (input_rows, k), torch.bfloat16, device, "x", memory_options
        )
        weight_guard = _guarded_tensor(
            (expert_num // world_size, k, n), torch.bfloat16, device,
            "weight", memory_options,
        )
        x = _make_tensor(
            (input_rows, k), case["input_distribution"], device, generator,
            out=x_guard.tensor,
        )
        weight = _make_tensor(
            (expert_num // world_size, k, n), case["input_distribution"],
            device, generator, out=weight_guard.tensor,
        )
        guarded_inputs.extend([x_guard, weight_guard])
    else:
        x = _make_tensor((input_rows, k), case["input_distribution"], device, generator)
        weight = _make_tensor(
            (expert_num // world_size, k, n), case["input_distribution"], device, generator
        )
    local_table = torch.tensor(local_values, dtype=torch.int64, device=device)
    global_table = torch.tensor(global_values, dtype=torch.int64, device=device)
    if memory_enabled:
        local_route_guard = GuardedTensor.allocate(
            local_table.shape, dtype=torch.int64, device=device,
            name="local_tokens_per_expert", guard_elements=256,
            guard_kind="exact", guard_value=0x5A5A5A5A5A5A5A5A,
        )
        global_route_guard = GuardedTensor.allocate(
            global_table.shape, dtype=torch.int64, device=device,
            name="global_tokens_per_local_expert", guard_elements=256,
            guard_kind="exact", guard_value=0x5A5A5A5A5A5A5A5A,
        )
        local_route_guard.tensor.copy_(local_table)
        global_route_guard.tensor.copy_(global_table)
        local_table = local_route_guard.tensor
        global_table = global_route_guard.tensor
        guarded_inputs.extend([local_route_guard, global_route_guard])
        torch_npu.npu.synchronize()
        memory_report["inputs_before"] = {
            item.name: {
                "guard": item.check(),
                "fingerprint": sampled_fingerprint(
                    item.tensor, max_elements=memory_options["fingerprint_elements"]
                ),
                "nonfinite": (
                    nonfinite_summary(item.tensor)
                    if item.tensor.dtype.is_floating_point else None
                ),
            }
            for item in guarded_inputs
        }
    global_table_2d = global_table.view(world_size, expert_num // world_size)
    launch_local_table = local_table
    launch_global_table = global_table
    launch_route_path = "device_tensor"
    if mssanitizer_enabled and mssanitizer_options["route_input"] == "list":
        # Use the Host values that already exist instead of introducing a D2H
        # solely to prepare the list-mode sanitizer launch.
        launch_local_table = list(local_values)
        launch_global_table = list(global_values)
        launch_route_path = "python_list"
    candidates = []
    candidate_guards = {}
    segments_override = None
    additional_reports = []
    launch_records = []
    launch_timings = {}
    launch_snapshot_metadata = {}
    failure_dump_counts = {
        "operator_output": 0,
        "all_to_allv_export": 0,
    }
    diagnostic_fingerprint_elements = memory_options["fingerprint_elements"]
    input_fingerprints_before = {}
    if not mssanitizer_enabled:
        input_fingerprints_before = {
            "x": sampled_fingerprint(
                x, max_elements=diagnostic_fingerprint_elements
            ),
            "weight": sampled_fingerprint(
                weight, max_elements=diagnostic_fingerprint_elements
            ),
            "local_tokens_per_expert": sampled_fingerprint(
                local_table, max_elements=diagnostic_fingerprint_elements
            ),
            "global_tokens_per_local_expert": sampled_fingerprint(
                global_table, max_elements=diagnostic_fingerprint_elements
            ),
        }
    add_timing("input_preparation", input_preparation_started)

    def execute_launch(path, launch_callable):
        """Run one globally identified launch and freeze its outputs."""
        launch_index = len(launch_records)
        barrier_seconds = 0.0
        if synchronize_launches:
            started = time.perf_counter()
            dist.barrier()
            barrier_seconds = time.perf_counter() - started
            timing_events.append({
                "phase": "pre_launch_barrier",
                "path": path,
                "launch_index": launch_index,
                "elapsed_seconds": barrier_seconds,
            })

        started = time.perf_counter()
        result = launch_callable()
        torch_npu.npu.synchronize()
        operator_seconds = time.perf_counter() - started
        timing_events.append({
            "phase": "operator_launch",
            "path": path,
            "launch_index": launch_index,
            "elapsed_seconds": operator_seconds,
        })

        outputs = result if isinstance(result, tuple) else (result,)
        started = time.perf_counter()
        snapshots = tuple(output.detach().clone() for output in outputs)
        # A completed immutable snapshot is the boundary between global launch
        # N and N+1.  No rank can enter the next aligned launch before this
        # rank reaches its next pre-launch Gloo barrier.
        torch_npu.npu.synchronize()
        snapshot_seconds = time.perf_counter() - started
        timing_events.append({
            "phase": "output_snapshot",
            "path": path,
            "launch_index": launch_index,
            "elapsed_seconds": snapshot_seconds,
        })
        launch_timings[path] = {
            "pre_launch_barrier_seconds": barrier_seconds,
            "operator_seconds": operator_seconds,
            "snapshot_seconds": snapshot_seconds,
        }
        launch_snapshot_metadata[path] = [
            _tensor_metadata(snapshot) for snapshot in snapshots
        ]
        snapshot_result = snapshots if isinstance(result, tuple) else snapshots[0]
        return result, snapshot_result

    def record_launch(
        path, output, *, allocation_mode, export_output=None,
        output_prefill=None,
    ):
        launch_records.append({
            "launch_index": len(launch_records),
            "path": path,
            "allocation_mode": allocation_mode,
            "caller_provided_out": allocation_mode == "caller_provided",
            "caller_provided_all_to_allv_out": (
                allocation_mode == "caller_provided" and export_output is not None
            ),
            "cross_rank_aligned": synchronize_launches,
            "global_launch_index": (
                len(launch_records) if synchronize_launches else None
            ),
            "completion_protocol": "production",
            "output_prefill": output_prefill,
            "output": _tensor_metadata(output),
            "all_to_allv_output": (
                _tensor_metadata(export_output)
                if export_output is not None else None
            ),
            "timings": launch_timings.get(path, {}),
            "report_snapshots": launch_snapshot_metadata.get(path, []),
        })

    def reserve_failure_dump(output_role, report):
        if not case["accuracy"]["dump_on_failure"] or report["passed"]:
            return False
        current = failure_dump_counts[output_role]
        if (
            max_failure_dumps_per_role >= 0
            and current >= max_failure_dumps_per_role
        ):
            report["failure_dump"] = {
                "written": False,
                "reason": "per-role dump limit reached",
                "limit": max_failure_dumps_per_role,
            }
            return False
        failure_dump_counts[output_role] = current + 1
        report["failure_dump"] = {
            "written": True,
            "limit": max_failure_dumps_per_role,
        }
        return True
    if case["operator"] == "grouped_mat_mul_all_to_allv":
        started = time.perf_counter()
        expected = _grouped_mat_mul_all_to_allv_golden(
            x, weight, route_matrix, rank, world_size, expert_num
        )
        add_timing("golden_generation", started, operator=case["operator"])
        valid_rows = expected.shape[0]
        segments_override = _grouped_mat_mul_all_to_allv_segments(
            route_matrix, rank, world_size, expert_num
        )
        if memory_enabled:
            _prepare_memory_safety_launch([])
            memory_report["launches"].append({"path": "ordinary", "workspace_poisoned": True})
        ordinary, ordinary_snapshot = execute_launch(
            "ordinary",
            lambda: udma.grouped_mat_mul_all_to_allv(
                x, weight, launch_local_table, launch_global_table,
                ep_size=world_size, expert_num=expert_num,
            ),
        )
        record_launch("ordinary", ordinary, allocation_mode="automatic")
        candidates.append((
            "ordinary", ordinary_snapshot,
            _padding_stats(ordinary_snapshot, valid_rows),
        ))
        del ordinary
        gc.collect()

        if not mssanitizer_enabled:
            for repeat in range(case["accuracy"]["auto_repeats"]):
                path = "auto_repeat_{}".format(repeat)
                if memory_enabled:
                    _prepare_memory_safety_launch([])
                    memory_report["launches"].append({
                        "path": path, "workspace_poisoned": True,
                    })
                auto_actual, auto_snapshot = execute_launch(
                    path,
                    lambda: udma.grouped_mat_mul_all_to_allv(
                        x, weight, local_table, global_table,
                        ep_size=world_size, expert_num=expert_num,
                    ),
                )
                record_launch(path, auto_actual, allocation_mode="automatic")
                candidates.append((
                    path, auto_snapshot,
                    _padding_stats(auto_snapshot, valid_rows),
                ))
                del auto_actual

            caller_repeats = (
                case["accuracy"]["repeats"] if check_reusable_out else 0
            )
            reusable_guard = None
            reusable_out = None
            if caller_repeats:
                if memory_enabled:
                    reusable_guard = _guarded_tensor(
                        expected.shape, torch.bfloat16, device, "gmm_a2av_out",
                        memory_options, output=True,
                    )
                    reusable_out = reusable_guard.tensor
                else:
                    reusable_out = torch.empty(
                        expected.shape, dtype=torch.bfloat16, device=device
                    )
            for repeat in range(caller_repeats):
                path = "out_repeat_{}".format(repeat)
                if memory_enabled:
                    _prepare_memory_safety_launch([reusable_guard])
                    memory_report["launches"].append({
                        "path": path, "workspace_poisoned": True,
                    })
                else:
                    reusable_out.fill_(float("nan"))
                out_actual, out_snapshot = execute_launch(
                    path,
                    lambda: udma.grouped_mat_mul_all_to_allv(
                        x, weight, local_table, global_table,
                        ep_size=world_size, expert_num=expert_num,
                        out=reusable_out,
                    ),
                )
                record_launch(
                    path, out_actual, allocation_mode="caller_provided",
                    output_prefill="nan",
                )
                candidates.append((
                    path, out_snapshot,
                    _padding_stats(out_snapshot, valid_rows),
                ))
                if memory_enabled:
                    candidate_guards[path] = [reusable_guard.check()]

            if memory_enabled and caller_repeats:
                local_host = local_table.cpu().tolist()
                global_host = global_table.cpu().tolist()
                _prepare_memory_safety_launch([reusable_guard])
                memory_report["launches"].append({
                    "path": "host_route", "workspace_poisoned": True,
                    "route_path": "host_list_with_h2d",
                })
                out_actual, out_snapshot = execute_launch(
                    "host_route",
                    lambda: udma.grouped_mat_mul_all_to_allv(
                        x, weight, local_host, global_host,
                        ep_size=world_size, expert_num=expert_num,
                        out=reusable_out,
                    ),
                )
                record_launch(
                    "host_route", out_actual,
                    allocation_mode="caller_provided",
                    output_prefill="nan",
                )
                candidates.append((
                    "host_route", out_snapshot,
                    _padding_stats(out_snapshot, valid_rows),
                ))
                candidate_guards["host_route"] = [reusable_guard.check()]
            if caller_repeats:
                del out_actual, reusable_out
            gc.collect()
    else:
        started = time.perf_counter()
        expected = _golden(x, weight, route_matrix, rank, world_size, expert_num)
        expected_all_to_allv = _all_to_allv_golden(
            x, route_matrix, rank, world_size, expert_num
        )
        add_timing("golden_generation", started, operator=case["operator"])
        valid_rows = received_rows
        if valid_rows != expected.shape[0]:
            raise RuntimeError("golden rows do not match the global route table")
        export_enabled = case["output_all_to_allv"]

        def add_export_report(export_tensor, path, launch_index):
            started = time.perf_counter()
            report = analyze_accuracy(
                export_tensor, expected_all_to_allv,
                global_table=global_table_2d.cpu().tolist(),
                topk=case["accuracy"]["topk"],
                heatmap_bins=case["accuracy"]["heatmap_bins"],
            )
            add_timing(
                "accuracy_analysis", started, path=path,
                launch_index=launch_index, output_role="all_to_allv_export",
            )
            report.update(
                case_id=case["id"], rank=rank, path=path,
                launch_index=launch_index,
                global_launch_index=(
                    launch_records[launch_index]["global_launch_index"]
                ),
                cross_rank_aligned=(
                    launch_records[launch_index]["cross_rank_aligned"]
                ),
                allocation_mode=(
                    launch_records[launch_index]["allocation_mode"]
                ),
                output_role="all_to_allv_export",
                valid_rows=expected_all_to_allv.shape[0],
                padding={"elements": 0, "nonzero_count": 0, "max_abs": 0.0},
            )
            dump = None
            if reserve_failure_dump("all_to_allv_export", report):
                selected_rows = sorted({
                    int(item["row"])
                    for item in report.get("top_mismatches", [])
                })
                row_index = torch.tensor(
                    selected_rows, dtype=torch.long, device=export_tensor.device
                )
                dump = {
                    "case": case,
                    "rank": rank,
                    "launch": launch_records[launch_index],
                    "selected_output_rows": selected_rows,
                    "actual_selected_rows": export_tensor.detach().index_select(
                        0, row_index
                    ).cpu(),
                    "expected_selected_rows": (
                        expected_all_to_allv.detach().index_select(0, row_index).cpu()
                    ),
                    "local_table": local_table,
                    "global_table": global_table,
                }
            additional_reports.append((path, report, dump))

        if memory_enabled:
            _prepare_memory_safety_launch([])
            memory_report["launches"].append({
                "path": "export_output" if export_enabled else "ordinary",
                "workspace_poisoned": True,
            })
        ordinary_path = "export_output" if export_enabled else "ordinary"
        if export_enabled:
            ordinary_result, ordinary_snapshot_result = execute_launch(
                ordinary_path,
                lambda: udma.all_to_allv_grouped_mat_mul(
                    x, weight, launch_local_table, launch_global_table,
                    ep_size=world_size, expert_num=expert_num,
                    received_rows=received_rows,
                    output_all_to_allv=True,
                ),
            )
            ordinary, exported_all_to_allv = ordinary_result
            ordinary_snapshot, exported_snapshot = ordinary_snapshot_result
        else:
            ordinary, ordinary_snapshot = execute_launch(
                ordinary_path,
                lambda: udma.all_to_allv_grouped_mat_mul(
                    x, weight, launch_local_table, launch_global_table,
                    ep_size=world_size, expert_num=expert_num,
                    received_rows=received_rows,
                ),
            )
            exported_all_to_allv = None
            exported_snapshot = None
        record_launch(
            ordinary_path, ordinary, allocation_mode="automatic",
            export_output=exported_all_to_allv,
        )
        if export_enabled:
            add_export_report(
                exported_snapshot, "all_to_allv_export",
                launch_records[-1]["launch_index"],
            )
        candidates.append((
            ordinary_path,
            ordinary_snapshot, _padding_stats(ordinary_snapshot, valid_rows),
        ))

        if not mssanitizer_enabled:
            # Consecutive automatically allocated launches isolate persistent
            # kernel/workspace state from the caller-provided output overload.
            for repeat in range(case["accuracy"]["auto_repeats"]):
                path = "auto_repeat_{}".format(repeat)
                if memory_enabled:
                    _prepare_memory_safety_launch([])
                    memory_report["launches"].append({
                        "path": path, "workspace_poisoned": True,
                    })
                if export_enabled:
                    auto_result, auto_snapshot_result = execute_launch(
                        path,
                        lambda: udma.all_to_allv_grouped_mat_mul(
                            x, weight, local_table, global_table,
                            ep_size=world_size, expert_num=expert_num,
                            received_rows=received_rows,
                            output_all_to_allv=True,
                        ),
                    )
                    auto_actual, auto_export = auto_result
                    auto_snapshot, auto_export_snapshot = auto_snapshot_result
                else:
                    auto_actual, auto_snapshot = execute_launch(
                        path,
                        lambda: udma.all_to_allv_grouped_mat_mul(
                            x, weight, local_table, global_table,
                            ep_size=world_size, expert_num=expert_num,
                            received_rows=received_rows,
                        ),
                    )
                    auto_export = None
                    auto_export_snapshot = None
                record_launch(
                    path, auto_actual, allocation_mode="automatic",
                    export_output=auto_export,
                )
                candidates.append((
                    path, auto_snapshot,
                    _padding_stats(auto_snapshot, valid_rows),
                ))
                if export_enabled:
                    add_export_report(
                        auto_export_snapshot,
                        "all_to_allv_export_{}".format(path),
                        launch_records[-1]["launch_index"],
                    )

            reusable_guard = None
            reusable_export_guard = None
            reusable_out = None
            reusable_all_to_allv_out = None
            caller_repeats = (
                case["accuracy"]["repeats"]
                if check_reusable_out and not automatic_only else 0
            )
            if caller_repeats:
                if memory_enabled:
                    reusable_guard = _guarded_tensor(
                        ordinary.shape, torch.bfloat16, device, "a2av_gmm_out",
                        memory_options, output=True,
                    )
                    reusable_out = reusable_guard.tensor
                    if export_enabled:
                        reusable_export_guard = _guarded_tensor(
                            expected_all_to_allv.shape, torch.bfloat16, device,
                            "all_to_allv_export", memory_options, output=True,
                        )
                        reusable_all_to_allv_out = reusable_export_guard.tensor
                else:
                    reusable_out = torch.zeros_like(ordinary)
                    reusable_all_to_allv_out = (
                        torch.empty_like(expected_all_to_allv)
                        if export_enabled else None
                    )
            for repeat in range(caller_repeats):
                path = "out_repeat_{}".format(repeat)
                if memory_enabled:
                    launch_guards = [reusable_guard]
                    if reusable_export_guard is not None:
                        launch_guards.append(reusable_export_guard)
                    _prepare_memory_safety_launch(launch_guards)
                    memory_report["launches"].append({
                        "path": path, "workspace_poisoned": True,
                    })
                else:
                    reusable_out.fill_(float("nan"))
                    if reusable_all_to_allv_out is not None:
                        reusable_all_to_allv_out.fill_(float("nan"))
                if export_enabled:
                    out_result, out_snapshot_result = execute_launch(
                        path,
                        lambda: udma.all_to_allv_grouped_mat_mul(
                            x, weight, local_table, global_table,
                            ep_size=world_size, expert_num=expert_num,
                            out=reusable_out, received_rows=received_rows,
                            output_all_to_allv=True,
                            all_to_allv_out=reusable_all_to_allv_out,
                        ),
                    )
                    out_actual, export_actual = out_result
                    out_snapshot, export_snapshot = out_snapshot_result
                else:
                    out_actual, out_snapshot = execute_launch(
                        path,
                        lambda: udma.all_to_allv_grouped_mat_mul(
                            x, weight, local_table, global_table,
                            ep_size=world_size, expert_num=expert_num,
                            out=reusable_out, received_rows=received_rows,
                        ),
                    )
                    export_actual = None
                    export_snapshot = None
                record_launch(
                    path, out_actual, allocation_mode="caller_provided",
                    export_output=export_actual if export_enabled else None,
                    output_prefill="nan",
                )
                candidates.append((
                    path, out_snapshot,
                    _padding_stats(out_snapshot, valid_rows),
                ))
                if memory_enabled:
                    candidate_guards[path] = [guard.check() for guard in launch_guards]
                if export_enabled:
                    add_export_report(
                        export_snapshot,
                        "all_to_allv_export_out_repeat_{}".format(repeat),
                        launch_records[-1]["launch_index"],
                    )

            if memory_enabled and caller_repeats:
                local_host = local_table.cpu().tolist()
                global_host = global_table.cpu().tolist()
                launch_guards = [reusable_guard]
                if reusable_export_guard is not None:
                    launch_guards.append(reusable_export_guard)
                _prepare_memory_safety_launch(launch_guards)
                memory_report["launches"].append({
                    "path": "host_route", "workspace_poisoned": True,
                    "route_path": "host_list_with_h2d_and_route_metadata",
                })
                if export_enabled:
                    out_result, out_snapshot_result = execute_launch(
                        "host_route",
                        lambda: udma.all_to_allv_grouped_mat_mul(
                            x, weight, local_host, global_host,
                            ep_size=world_size, expert_num=expert_num,
                            out=reusable_out, received_rows=received_rows,
                            output_all_to_allv=True,
                            all_to_allv_out=reusable_all_to_allv_out,
                        ),
                    )
                    out_actual, export_actual = out_result
                    out_snapshot, export_snapshot = out_snapshot_result
                else:
                    out_actual, out_snapshot = execute_launch(
                        "host_route",
                        lambda: udma.all_to_allv_grouped_mat_mul(
                            x, weight, local_host, global_host,
                            ep_size=world_size, expert_num=expert_num,
                            out=reusable_out, received_rows=received_rows,
                        ),
                    )
                    export_actual = None
                    export_snapshot = None
                record_launch(
                    "host_route", out_actual,
                    allocation_mode="caller_provided",
                    export_output=export_actual if export_enabled else None,
                    output_prefill="nan",
                )
                candidates.append((
                    "host_route", out_snapshot,
                    _padding_stats(out_snapshot, valid_rows),
                ))
                candidate_guards["host_route"] = [
                    guard.check() for guard in launch_guards
                ]
                if export_enabled:
                    add_export_report(
                        export_snapshot, "all_to_allv_export_host_route",
                        launch_records[-1]["launch_index"],
                    )

    input_diagnostics_started = time.perf_counter()
    input_diagnostics = {
        "skipped": mssanitizer_enabled,
        "skip_reason": (
            "mssanitizer mode keeps exactly one target operator launch"
            if mssanitizer_enabled else None
        ),
        "tensor_metadata": {
            "x": _tensor_metadata(x),
            "weight": _tensor_metadata(weight),
            "local_tokens_per_expert": _tensor_metadata(local_table),
            "global_tokens_per_local_expert": _tensor_metadata(global_table),
        },
        "route_values": {
            "local_tokens_per_expert": local_values,
            "global_tokens_per_local_expert": global_values,
            "received_rows": received_rows,
        },
    }
    if not mssanitizer_enabled:
        torch_npu.npu.synchronize()
        input_fingerprints_after = {
            "x": sampled_fingerprint(
                x, max_elements=diagnostic_fingerprint_elements
            ),
            "weight": sampled_fingerprint(
                weight, max_elements=diagnostic_fingerprint_elements
            ),
            "local_tokens_per_expert": sampled_fingerprint(
                local_table, max_elements=diagnostic_fingerprint_elements
            ),
            "global_tokens_per_local_expert": sampled_fingerprint(
                global_table, max_elements=diagnostic_fingerprint_elements
            ),
        }
        input_diagnostics["fingerprints"] = {
            name: compare_fingerprints(
                input_fingerprints_before[name], input_fingerprints_after[name]
            )
            for name in input_fingerprints_before
        }
        input_diagnostics["mutation_detected"] = any(
            not value["passed"]
            for value in input_diagnostics["fingerprints"].values()
        )

    if memory_enabled:
        torch_npu.npu.synchronize()
        inputs_after = {}
        input_mutation = {}
        for item in guarded_inputs:
            after_fingerprint = sampled_fingerprint(
                item.tensor, max_elements=memory_options["fingerprint_elements"]
            )
            inputs_after[item.name] = {
                "guard": item.check(),
                "fingerprint": after_fingerprint,
                "nonfinite": (
                    nonfinite_summary(item.tensor)
                    if item.tensor.dtype.is_floating_point else None
                ),
            }
            input_mutation[item.name] = compare_fingerprints(
                memory_report["inputs_before"][item.name]["fingerprint"],
                after_fingerprint,
            )
        memory_report["inputs_after"] = inputs_after
        memory_report["input_mutation"] = input_mutation
        memory_report["candidate_guards"] = candidate_guards
        memory_report["candidate_nonfinite"] = {
            name: nonfinite_summary(actual)
            for name, actual, _ in candidates
        }
        memory_report["common_passed"] = (
            memory_report["hbm_poison"]["passed"]
            and all(value["guard"]["passed"] for value in inputs_after.values())
            and all(value["passed"] for value in input_mutation.values())
            and all(
                value["nonfinite"] is None or value["nonfinite"]["passed"]
                for value in inputs_after.values()
            )
        )
        memory_report["passed"] = memory_report["common_passed"] and all(
            guard["passed"]
            for guards in candidate_guards.values()
            for guard in guards
        ) and all(
            value["passed"]
            for value in memory_report["candidate_nonfinite"].values()
        )
    add_timing("input_and_memory_diagnostics", input_diagnostics_started)

    case_root = output_root / case["id"] / "rank_{}".format(rank)

    def add_memory_result(report, path):
        if not memory_enabled:
            return
        guard_path = path
        if path.startswith("all_to_allv_export_out_repeat_"):
            guard_path = path.replace(
                "all_to_allv_export_out_repeat_", "out_repeat_", 1
            )
        elif path == "all_to_allv_export_host_route":
            guard_path = "host_route"
        guards = candidate_guards.get(guard_path, [])
        output_nonfinite_passed = report.get("actual_nonfinite", {}).get(
            "passed", True
        )
        memory_passed = memory_report["common_passed"] and all(
            guard["passed"] for guard in guards
        ) and output_nonfinite_passed
        report["memory_safety"] = {
            "enabled": True,
            "passed": memory_passed,
            "output_guards": guards,
            "coverage": "red_zones" if guards else "allocator_and_workspace_poison_only",
            "details": "../memory_safety.json",
        }
        if not memory_passed:
            report["passed"] = False
            if "FAIL_MEMORY_SAFETY" not in report["failure_reasons"]:
                report["failure_reasons"].append("FAIL_MEMORY_SAFETY")

    reports = []
    for name, report, dump in additional_reports:
        add_memory_result(report, name)
        started = time.perf_counter()
        write_accuracy_artifacts(
            case_root / name, report, dump_payload=dump
        )
        add_timing(
            "artifact_write", started, path=name,
            launch_index=report.get("launch_index"),
            output_role="all_to_allv_export",
        )
        reports.append(report)
    for name, actual, padding in candidates:
        analyze_arguments = {
            "topk": case["accuracy"]["topk"],
            "heatmap_bins": case["accuracy"]["heatmap_bins"],
        }
        if segments_override is None:
            analyze_arguments["global_table"] = global_table_2d.cpu().tolist()
        else:
            analyze_arguments["segments_override"] = segments_override
        started = time.perf_counter()
        report = analyze_accuracy(actual, expected, **analyze_arguments)
        launch_record = next(
            (item for item in launch_records if item["path"] == name), None
        )
        add_timing(
            "accuracy_analysis", started, path=name,
            launch_index=(
                launch_record["launch_index"]
                if launch_record is not None else None
            ),
            output_role="operator_output",
        )
        report.update(
            case_id=case["id"], rank=rank, path=name,
            launch_index=(
                launch_record["launch_index"]
                if launch_record is not None else None
            ),
            allocation_mode=(
                launch_record["allocation_mode"]
                if launch_record is not None else None
            ),
            global_launch_index=(
                launch_record["global_launch_index"]
                if launch_record is not None else None
            ),
            cross_rank_aligned=(
                launch_record["cross_rank_aligned"]
                if launch_record is not None else False
            ),
            output_role="operator_output",
            valid_rows=valid_rows, padding=padding,
        )
        add_memory_result(report, name)
        dump = None
        if reserve_failure_dump("operator_output", report):
            dump = {
                "case": case,
                "rank": rank,
                "launch": launch_record,
                "x": x,
                "weight": weight,
                "local_table": local_table,
                "global_table": global_table,
                "expected": expected,
                "actual": actual,
            }
        started = time.perf_counter()
        write_accuracy_artifacts(case_root / name, report, dump_payload=dump)
        add_timing(
            "artifact_write", started, path=name,
            launch_index=report.get("launch_index"),
            output_role="operator_output",
        )
        reports.append(report)

    any_accuracy_failure = any(not report["passed"] for report in reports)
    if any_accuracy_failure and not mssanitizer_enabled:
        # Inputs are generated finite. Exact post-failure scans tell whether an
        # operator launch mutated them or introduced NaN/Inf outside its output.
        input_diagnostics["post_failure_nonfinite"] = {
            "x": nonfinite_summary(x),
            "weight": nonfinite_summary(weight),
        }
    input_diagnostics["passed"] = (
        not input_diagnostics.get("mutation_detected", False)
        and all(
            value.get("passed", True)
            for value in input_diagnostics.get(
                "post_failure_nonfinite", {}
            ).values()
        )
    )
    for launch in launch_records:
        launch["report_paths"] = [
            report["path"] for report in reports
            if report.get("launch_index") == launch["launch_index"]
        ]
    write_json(case_root / "launch_diagnostics.json", {
        "case_id": case["id"],
        "rank": rank,
        "launches": launch_records,
    })
    write_json(case_root / "input_diagnostics.json", input_diagnostics)

    if memory_enabled:
        memory_report["passed"] = memory_report["passed"] and all(
            item.get("memory_safety", {}).get("passed", True)
            for item in reports
        )
        write_json(case_root / "memory_safety.json", memory_report)

    ordinary_valid = candidates[0][1]
    repeat_consistency = []
    for name, actual, _ in candidates[1:]:
        different = int(torch.count_nonzero(actual != ordinary_valid).item())
        repeat_consistency.append({"path": name, "different_elements_vs_ordinary": different})
    diagnosis = _failure_scope(reports)
    if not input_diagnostics["passed"]:
        diagnosis["likely_scope"] = "input_or_route_tensor_mutation"
    timing_events.append({
        "phase": "case_total",
        "elapsed_seconds": time.perf_counter() - case_started,
    })
    timing_totals = {}
    for event in timing_events:
        phase = event["phase"]
        timing_totals[phase] = timing_totals.get(phase, 0.0) + float(
            event["elapsed_seconds"]
        )
    timing_report = {
        "case_id": case["id"],
        "rank": rank,
        "cross_rank_launch_alignment": synchronize_launches,
        "check_reusable_out": check_reusable_out,
        "totals_seconds": timing_totals,
        "events": timing_events,
    }
    write_json(case_root / "timings.json", timing_report)
    summary = {
        "case_id": case["id"],
        "rank": rank,
        "passed": (
            all(item["passed"] for item in reports)
            and input_diagnostics["passed"]
        ),
        "reports": [_report_brief(item) for item in reports],
        "repeat_consistency": repeat_consistency,
        "diagnosis": diagnosis,
        "diagnostic_artifacts": {
            "launches": "launch_diagnostics.json",
            "inputs": "input_diagnostics.json",
            "timings": "timings.json",
        },
        "cross_rank_launch_alignment": synchronize_launches,
        "check_reusable_out": check_reusable_out,
        "timings": {
            "path": "timings.json",
            "totals_seconds": timing_totals,
        },
        "memory_safety": (
            {
                "enabled": True,
                "passed": memory_report["passed"],
                "details": "memory_safety.json",
            }
            if memory_enabled else {"enabled": False}
        ),
        "mssanitizer": {
            "enabled": mssanitizer_enabled,
            "route_input": launch_route_path if mssanitizer_enabled else None,
            "target_launches_per_rank": 1 if mssanitizer_enabled else None,
            "caller_provided_out": False if mssanitizer_enabled else None,
        },
        "sync_diagnosis": {
            "mode": "production",
            "automatic_only": automatic_only,
            "max_failure_dumps_per_role": max_failure_dumps_per_role,
            "failure_dumps_written": failure_dump_counts,
        },
    }
    write_json(case_root / "summary.json", summary)
    print(
        "[accuracy-diagnosis] case={} rank={} likely_scope={} "
        "input_mutation={} failed_auto={} failed_out={} failed_export={}".format(
            case["id"], rank, diagnosis["likely_scope"],
            not input_diagnostics["passed"],
            diagnosis["failed_auto_outputs"],
            diagnosis["failed_caller_out_outputs"],
            diagnosis["failed_repeat_exports"],
        ),
        flush=True,
    )
    print(
        "[accuracy-timing] case={} rank={} input={:.3f}s golden={:.3f}s "
        "barrier={:.3f}s operator={:.3f}s snapshot={:.3f}s "
        "analysis={:.3f}s artifacts={:.3f}s total={:.3f}s".format(
            case["id"], rank,
            timing_totals.get("input_preparation", 0.0),
            timing_totals.get("golden_generation", 0.0),
            timing_totals.get("pre_launch_barrier", 0.0),
            timing_totals.get("operator_launch", 0.0),
            timing_totals.get("output_snapshot", 0.0),
            timing_totals.get("accuracy_analysis", 0.0),
            timing_totals.get("artifact_write", 0.0),
            timing_totals.get("case_total", 0.0),
        ),
        flush=True,
    )
    for report in reports:
        failed_experts = [
            item["local_expert"]
            for item in report.get("by_local_expert", [])
            if item.get("mismatches", 0) > 0
        ]
        failed_sources = [
            item["source_rank"]
            for item in report.get("by_source_rank", [])
            if item.get("mismatches", 0) > 0
        ]
        row_runs = report.get("mismatch_row_runs", {})
        print(
            "[accuracy] case={} rank={} launch={} path={} passed={} "
            "matched_ratio={:.9f} max_abs_error={} nonfinite={} "
            "failed_rows={}-{} row_runs={} failed_experts={} "
            "failed_sources={}".format(
                case["id"], rank, report.get("launch_index"), report["path"],
                report["passed"], report["matched_ratio"],
                report["max_abs_error"],
                report.get("actual_nonfinite", {}).get("total_nonfinite", 0),
                row_runs.get("first_row"), row_runs.get("last_row"),
                row_runs.get("total_runs", 0),
                failed_experts, failed_sources,
            ),
            flush=True,
        )
    del candidates
    gc.collect()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-cases", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--memory-safety", action="store_true")
    parser.add_argument("--mssanitizer", action="store_true")
    parser.add_argument(
        "--mssanitizer-route-input", choices=("tensor", "list"), default="tensor"
    )
    parser.add_argument("--hbm-reserve-bytes", type=int, default=2 << 30)
    parser.add_argument("--hbm-poison-chunk-bytes", type=int, default=256 << 20)
    parser.add_argument("--guard-bytes", type=int, default=16 << 20)
    parser.add_argument("--fingerprint-elements", type=int, default=1 << 20)
    parser.add_argument("--automatic-only", action="store_true")
    parser.add_argument("--check-reusable-out", action="store_true")
    parser.add_argument("--max-failure-dumps-per-role", type=int, default=-1)
    args = parser.parse_args()
    if args.memory_safety and args.mssanitizer:
        raise ValueError("--memory-safety and --mssanitizer are mutually exclusive")
    if args.automatic_only and (args.memory_safety or args.mssanitizer):
        raise ValueError(
            "--automatic-only cannot be combined with memory-safety or mssanitizer"
        )
    if args.automatic_only and args.check_reusable_out:
        raise ValueError(
            "--automatic-only and --check-reusable-out are mutually exclusive"
        )
    if args.mssanitizer and args.check_reusable_out:
        raise ValueError(
            "--mssanitizer keeps one automatic launch and cannot check reusable out"
        )
    if args.max_failure_dumps_per_role < -1:
        raise ValueError("--max-failure-dumps-per-role must be -1 or non-negative")
    memory_options = {
        "enabled": args.memory_safety,
        "hbm_reserve_bytes": args.hbm_reserve_bytes,
        "hbm_poison_chunk_bytes": args.hbm_poison_chunk_bytes,
        "guard_bytes": args.guard_bytes,
        "fingerprint_elements": args.fingerprint_elements,
    }
    mssanitizer_options = {
        "enabled": args.mssanitizer,
        "route_input": args.mssanitizer_route_input,
    }
    diagnostic_options = {
        "automatic_only": args.automatic_only,
        "check_reusable_out": args.check_reusable_out or args.memory_safety,
        "max_failure_dumps_per_role": args.max_failure_dumps_per_role,
    }
    cases = read_resolved_cases(args.resolved_cases)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if any(case["world_size"] != world_size for case in cases):
        raise ValueError("all resolved cases must match torchrun world size")

    output_root = Path(args.output_dir).resolve()
    device = "npu:{}".format(local_rank)
    initialized_dist = False
    initialized_udma = False
    all_passed = True
    try:
        torch_npu.npu.set_device(local_rank)
        if not dist.is_gloo_available():
            raise RuntimeError(
                "validation requires PyTorch Gloo support for host-side rank coordination"
            )
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        initialized_dist = True
        dist.barrier()
        init_method = udma_init_method_from_env()
        heap_size = validation_udma_heap_size()
        if rank == 0:
            print("UDMA bootstrap: {}".format(init_method), flush=True)
            print("UDMA symmetric heap: {} bytes".format(heap_size), flush=True)
        udma.init_udma_group(
            rank=rank, world_size=world_size, local_rank=local_rank,
            init_method=init_method, heap_size=heap_size,
        )
        initialized_udma = True
        for case in cases:
            summary = _run_case(
                case, rank, world_size, device, output_root, memory_options,
                mssanitizer_options, diagnostic_options,
            )
            dist.barrier()
            if rank == 0:
                gathered = [
                    json.loads(
                        (output_root / case["id"] / "rank_{}".format(item_rank) / "summary.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    for item_rank in range(world_size)
                ]
                group = {
                    "case_id": case["id"],
                    "passed": all(item["passed"] for item in gathered),
                    "ranks": gathered,
                }
                write_json(output_root / case["id"] / "summary.json", group)
                print(json.dumps(group), flush=True)
            all_passed = all_passed and summary["passed"]
            dist.barrier()
    finally:
        if initialized_dist:
            dist.destroy_process_group()
        gc.collect()
        if initialized_udma:
            udma.destroy_udma_group()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
