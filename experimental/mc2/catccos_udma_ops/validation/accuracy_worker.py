# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""One-process-per-NPU accuracy worker for resolved validation cases."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu

import ops_transformer_catccos_udma as udma

try:
    from .case_schema import read_resolved_cases
    from .common import udma_init_method_from_env, validation_udma_heap_size, write_json
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
    from common import udma_init_method_from_env, validation_udma_heap_size, write_json
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
    return expected.to(dtype=torch.bfloat16, device=x.device)


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


def _run_case(case, rank, world_size, device, output_root, memory_options):
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
    candidates = []
    candidate_guards = {}
    segments_override = None
    additional_reports = []
    if case["operator"] == "grouped_mat_mul_all_to_allv":
        expected = _grouped_mat_mul_all_to_allv_golden(
            x, weight, route_matrix, rank, world_size, expert_num
        )
        valid_rows = expected.shape[0]
        segments_override = _grouped_mat_mul_all_to_allv_segments(
            route_matrix, rank, world_size, expert_num
        )
        if memory_enabled:
            _prepare_memory_safety_launch([])
            memory_report["launches"].append({"path": "ordinary", "workspace_poisoned": True})
        ordinary = udma.grouped_mat_mul_all_to_allv(
            x, weight, local_table, global_table,
            ep_size=world_size, expert_num=expert_num,
        )
        torch_npu.npu.synchronize()
        candidates.append(("ordinary", ordinary.clone(), _padding_stats(ordinary, valid_rows)))
        del ordinary
        gc.collect()

        reusable_guard = None
        if memory_enabled:
            reusable_guard = _guarded_tensor(
                expected.shape, torch.bfloat16, device, "gmm_a2av_out",
                memory_options, output=True,
            )
            reusable_out = reusable_guard.tensor
        else:
            reusable_out = torch.empty(expected.shape, dtype=torch.bfloat16, device=device)
        for repeat in range(case["accuracy"]["repeats"]):
            path = "out_repeat_{}".format(repeat)
            if memory_enabled:
                _prepare_memory_safety_launch([reusable_guard])
                memory_report["launches"].append({"path": path, "workspace_poisoned": True})
            out_actual = udma.grouped_mat_mul_all_to_allv(
                x, weight, local_table, global_table,
                ep_size=world_size, expert_num=expert_num, out=reusable_out,
            )
            torch_npu.npu.synchronize()
            candidates.append((
                path, out_actual.clone(),
                _padding_stats(out_actual, valid_rows),
            ))
            if memory_enabled:
                candidate_guards[path] = [reusable_guard.check()]

        if memory_enabled:
            local_host = local_table.cpu().tolist()
            global_host = global_table.cpu().tolist()
            _prepare_memory_safety_launch([reusable_guard])
            memory_report["launches"].append({
                "path": "host_route", "workspace_poisoned": True,
                "route_path": "host_list_with_h2d",
            })
            out_actual = udma.grouped_mat_mul_all_to_allv(
                x, weight, local_host, global_host,
                ep_size=world_size, expert_num=expert_num, out=reusable_out,
            )
            torch_npu.npu.synchronize()
            candidates.append((
                "host_route", out_actual.clone(),
                _padding_stats(out_actual, valid_rows),
            ))
            candidate_guards["host_route"] = [reusable_guard.check()]
        del out_actual, reusable_out
        gc.collect()
    else:
        expected = _golden(x, weight, route_matrix, rank, world_size, expert_num)
        expected_all_to_allv = _all_to_allv_golden(
            x, route_matrix, rank, world_size, expert_num
        )
        valid_rows = received_rows
        if valid_rows != expected.shape[0]:
            raise RuntimeError("golden rows do not match the global route table")
        export_enabled = case["output_all_to_allv"]
        if memory_enabled:
            _prepare_memory_safety_launch([])
            memory_report["launches"].append({
                "path": "export_output" if export_enabled else "ordinary",
                "workspace_poisoned": True,
            })
        if export_enabled:
            ordinary, exported_all_to_allv = udma.all_to_allv_grouped_mat_mul(
                x, weight, local_table, global_table,
                ep_size=world_size, expert_num=expert_num,
                received_rows=received_rows,
                output_all_to_allv=True,
            )
            torch_npu.npu.synchronize()
            export_report = analyze_accuracy(
                exported_all_to_allv, expected_all_to_allv,
                global_table=global_table_2d.cpu().tolist(),
                topk=case["accuracy"]["topk"],
                heatmap_bins=case["accuracy"]["heatmap_bins"],
            )
            export_report.update(
                case_id=case["id"], rank=rank, path="all_to_allv_export",
                valid_rows=expected_all_to_allv.shape[0],
                padding={"elements": 0, "nonzero_count": 0, "max_abs": 0.0},
            )
            additional_reports.append(("all_to_allv_export", export_report))
        else:
            ordinary = udma.all_to_allv_grouped_mat_mul(
                x, weight, local_table, global_table,
                ep_size=world_size, expert_num=expert_num,
                received_rows=received_rows,
            )
            torch_npu.npu.synchronize()
        candidates.append((
            "export_output" if export_enabled else "ordinary",
            ordinary.clone(), _padding_stats(ordinary, valid_rows),
        ))

        reusable_guard = None
        reusable_export_guard = None
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
                reusable_all_to_allv_out = None
        else:
            reusable_out = torch.zeros_like(ordinary)
            reusable_all_to_allv_out = (
                torch.empty_like(expected_all_to_allv) if export_enabled else None
            )
        for repeat in range(case["accuracy"]["repeats"]):
            path = "out_repeat_{}".format(repeat)
            if memory_enabled:
                launch_guards = [reusable_guard]
                if reusable_export_guard is not None:
                    launch_guards.append(reusable_export_guard)
                _prepare_memory_safety_launch(launch_guards)
                memory_report["launches"].append({"path": path, "workspace_poisoned": True})
            else:
                reusable_out.zero_()
            if export_enabled:
                out_actual, export_actual = udma.all_to_allv_grouped_mat_mul(
                    x, weight, local_table, global_table,
                    ep_size=world_size, expert_num=expert_num, out=reusable_out,
                    received_rows=received_rows,
                    output_all_to_allv=True,
                    all_to_allv_out=reusable_all_to_allv_out,
                )
            else:
                out_actual = udma.all_to_allv_grouped_mat_mul(
                    x, weight, local_table, global_table,
                    ep_size=world_size, expert_num=expert_num, out=reusable_out,
                    received_rows=received_rows,
                )
            torch_npu.npu.synchronize()
            candidates.append((
                path, out_actual.clone(),
                _padding_stats(out_actual, valid_rows),
            ))
            if memory_enabled:
                candidate_guards[path] = [guard.check() for guard in launch_guards]
            if export_enabled and memory_enabled:
                repeat_export_report = analyze_accuracy(
                    export_actual, expected_all_to_allv,
                    global_table=global_table_2d.cpu().tolist(),
                    topk=case["accuracy"]["topk"],
                    heatmap_bins=case["accuracy"]["heatmap_bins"],
                )
                repeat_export_report.update(
                    case_id=case["id"], rank=rank,
                    path="all_to_allv_export_repeat_{}".format(repeat),
                    valid_rows=expected_all_to_allv.shape[0],
                    padding={"elements": 0, "nonzero_count": 0, "max_abs": 0.0},
                )
                additional_reports.append((repeat_export_report["path"], repeat_export_report))

        if memory_enabled:
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
                out_actual, export_actual = udma.all_to_allv_grouped_mat_mul(
                    x, weight, local_host, global_host,
                    ep_size=world_size, expert_num=expert_num, out=reusable_out,
                    received_rows=received_rows, output_all_to_allv=True,
                    all_to_allv_out=reusable_all_to_allv_out,
                )
            else:
                out_actual = udma.all_to_allv_grouped_mat_mul(
                    x, weight, local_host, global_host,
                    ep_size=world_size, expert_num=expert_num, out=reusable_out,
                    received_rows=received_rows,
                )
            torch_npu.npu.synchronize()
            candidates.append((
                "host_route", out_actual.clone(),
                _padding_stats(out_actual, valid_rows),
            ))
            candidate_guards["host_route"] = [guard.check() for guard in launch_guards]
            if export_enabled:
                host_export_report = analyze_accuracy(
                    export_actual, expected_all_to_allv,
                    global_table=global_table_2d.cpu().tolist(),
                    topk=case["accuracy"]["topk"],
                    heatmap_bins=case["accuracy"]["heatmap_bins"],
                )
                host_export_report.update(
                    case_id=case["id"], rank=rank, path="all_to_allv_export_host_route",
                    valid_rows=expected_all_to_allv.shape[0],
                    padding={"elements": 0, "nonzero_count": 0, "max_abs": 0.0},
                )
                additional_reports.append((host_export_report["path"], host_export_report))

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

    case_root = output_root / case["id"] / "rank_{}".format(rank)

    def add_memory_result(report, path):
        if not memory_enabled:
            return
        guard_path = path
        if path.startswith("all_to_allv_export_repeat_"):
            guard_path = path.replace("all_to_allv_export_repeat_", "out_repeat_", 1)
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
    for name, report in additional_reports:
        add_memory_result(report, name)
        write_accuracy_artifacts(case_root / name, report)
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
        report = analyze_accuracy(actual, expected, **analyze_arguments)
        report.update(
            case_id=case["id"], rank=rank, path=name,
            valid_rows=valid_rows, padding=padding,
        )
        add_memory_result(report, name)
        dump = None
        if case["accuracy"]["dump_on_failure"] and not report["passed"]:
            dump = {
                "case": case,
                "rank": rank,
                "x": x,
                "weight": weight,
                "local_table": local_table,
                "global_table": global_table,
                "expected": expected,
                "actual": actual,
            }
        write_accuracy_artifacts(case_root / name, report, dump_payload=dump)
        reports.append(report)

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
    summary = {
        "case_id": case["id"],
        "rank": rank,
        "passed": all(item["passed"] for item in reports),
        "reports": [
            {
                "path": item["path"],
                "passed": item["passed"],
                "matched_ratio": item["matched_ratio"],
                "max_abs_error": item["max_abs_error"],
                "failure_reasons": item["failure_reasons"],
                "actual_nonfinite": item.get("actual_nonfinite", {}).get(
                    "total_nonfinite", 0
                ),
                "memory_safety_passed": item.get("memory_safety", {}).get("passed"),
            }
            for item in reports
        ],
        "repeat_consistency": repeat_consistency,
        "memory_safety": (
            {
                "enabled": True,
                "passed": memory_report["passed"],
                "details": "memory_safety.json",
            }
            if memory_enabled else {"enabled": False}
        ),
    }
    write_json(case_root / "summary.json", summary)
    del candidates
    gc.collect()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-cases", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--memory-safety", action="store_true")
    parser.add_argument("--hbm-reserve-bytes", type=int, default=2 << 30)
    parser.add_argument("--hbm-poison-chunk-bytes", type=int, default=256 << 20)
    parser.add_argument("--guard-bytes", type=int, default=16 << 20)
    parser.add_argument("--fingerprint-elements", type=int, default=1 << 20)
    args = parser.parse_args()
    memory_options = {
        "enabled": args.memory_safety,
        "hbm_reserve_bytes": args.hbm_reserve_bytes,
        "hbm_poison_chunk_bytes": args.hbm_poison_chunk_bytes,
        "guard_bytes": args.guard_bytes,
        "fingerprint_elements": args.fingerprint_elements,
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
                case, rank, world_size, device, output_root, memory_options
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
