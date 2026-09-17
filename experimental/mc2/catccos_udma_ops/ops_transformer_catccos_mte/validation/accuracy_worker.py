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

import ops_transformer_catccos_mte as mte

try:
    from .case_schema import read_resolved_cases
    from .common import mte_init_method_from_env, validation_mte_heap_size, write_json
    from .precision import analyze_accuracy, write_accuracy_artifacts
except ImportError:  # Direct execution keeps the source tree out of package lookup.
    from case_schema import read_resolved_cases
    from common import mte_init_method_from_env, validation_mte_heap_size, write_json
    from precision import analyze_accuracy, write_accuracy_artifacts


def _make_tensor(shape, distribution, device, generator):
    kind = distribution["kind"]
    if kind == "uniform":
        result = torch.empty(shape, dtype=torch.float32)
        result.uniform_(distribution["low"], distribution["high"], generator=generator)
    elif kind == "normal":
        result = torch.empty(shape, dtype=torch.float32)
        result.normal_(distribution["mean"], distribution["std"], generator=generator)
    else:
        raise ValueError("unsupported distribution {}".format(kind))
    return result.to(dtype=torch.bfloat16, device=device)


def _golden(x, weight, route_matrix, rank, world_size, expert_num):
    # Keep the validation control plane on the host.  Initializing an HCCL
    # communicator after ACLSHMEM has claimed the MTE/HDC resources can fail,
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
    gathered = [torch.empty_like(local_output) for _ in range(world_size)]
    dist.all_gather(gathered, local_output)

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


def _run_case(case, rank, world_size, device, output_root):
    shape = case["shape"]
    m, n, k = shape["m"], shape["n"], shape["k"]
    expert_num = case["expert_num"]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(case["seed"] + rank)
    x = _make_tensor((m, k), case["input_distribution"], device, generator)
    weight = _make_tensor(
        (expert_num // world_size, k, n), case["input_distribution"], device, generator
    )
    local_table, global_table, received_rows = mte.build_rank_route(
        m, rank, world_size, expert_num, device
    )
    global_table_2d = global_table.view(world_size, expert_num // world_size)
    route_matrix = mte.build_global_route_matrix(m, world_size, expert_num)
    candidates = []
    segments_override = None
    additional_reports = []
    if case["operator"] == "grouped_mat_mul_all_to_allv":
        expected = _grouped_mat_mul_all_to_allv_golden(
            x, weight, route_matrix, rank, world_size, expert_num
        )
        valid_rows = m
        segments_override = _grouped_mat_mul_all_to_allv_segments(
            route_matrix, rank, world_size, expert_num
        )
        ordinary = mte.grouped_mat_mul_all_to_allv(
            x, weight, local_table, global_table,
            ep_size=world_size, expert_num=expert_num,
        )
        torch_npu.npu.synchronize()
        candidates.append(("ordinary", ordinary.clone(), _padding_stats(ordinary, valid_rows)))
        del ordinary
        gc.collect()

        reusable_out = torch.empty((m, n), dtype=torch.bfloat16, device=device)
        for repeat in range(case["accuracy"]["repeats"]):
            out_actual = mte.grouped_mat_mul_all_to_allv(
                x, weight, local_table, global_table,
                ep_size=world_size, expert_num=expert_num, out=reusable_out,
            )
            torch_npu.npu.synchronize()
            candidates.append((
                "out_repeat_{}".format(repeat), out_actual.clone(),
                _padding_stats(out_actual, valid_rows),
            ))
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
        if export_enabled:
            ordinary, exported_all_to_allv = mte.all_to_allv_grouped_mat_mul(
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
            ordinary = mte.all_to_allv_grouped_mat_mul(
                x, weight, local_table, global_table,
                ep_size=world_size, expert_num=expert_num,
                received_rows=received_rows,
            )
            torch_npu.npu.synchronize()
        candidates.append((
            "export_output" if export_enabled else "ordinary",
            ordinary.clone(), _padding_stats(ordinary, valid_rows),
        ))

        reusable_out = torch.zeros_like(ordinary)
        reusable_all_to_allv_out = (
            torch.empty_like(expected_all_to_allv) if export_enabled else None
        )
        for repeat in range(case["accuracy"]["repeats"]):
            reusable_out.zero_()
            if export_enabled:
                out_actual, _ = mte.all_to_allv_grouped_mat_mul(
                    x, weight, local_table, global_table,
                    ep_size=world_size, expert_num=expert_num, out=reusable_out,
                    received_rows=received_rows,
                    output_all_to_allv=True,
                    all_to_allv_out=reusable_all_to_allv_out,
                )
            else:
                out_actual = mte.all_to_allv_grouped_mat_mul(
                    x, weight, local_table, global_table,
                    ep_size=world_size, expert_num=expert_num, out=reusable_out,
                    received_rows=received_rows,
                )
            torch_npu.npu.synchronize()
            candidates.append((
                "out_repeat_{}".format(repeat), out_actual.clone(),
                _padding_stats(out_actual, valid_rows),
            ))

    case_root = output_root / case["id"] / "rank_{}".format(rank)
    reports = []
    for name, report in additional_reports:
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
            }
            for item in reports
        ],
        "repeat_consistency": repeat_consistency,
    }
    write_json(case_root / "summary.json", summary)
    del candidates
    gc.collect()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-cases", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    cases = read_resolved_cases(args.resolved_cases)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if any(case["world_size"] != world_size for case in cases):
        raise ValueError("all resolved cases must match torchrun world size")

    output_root = Path(args.output_dir).resolve()
    device = "npu:{}".format(local_rank)
    initialized_dist = False
    initialized_mte = False
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
        init_method = mte_init_method_from_env()
        heap_size = validation_mte_heap_size()
        if rank == 0:
            print("MTE bootstrap: {}".format(init_method), flush=True)
            print("MTE symmetric heap: {} bytes".format(heap_size), flush=True)
        mte.init_mte_group(
            rank=rank, world_size=world_size, local_rank=local_rank,
            init_method=init_method, heap_size=heap_size,
        )
        initialized_mte = True
        for case in cases:
            summary = _run_case(case, rank, world_size, device, output_root)
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
        if initialized_mte:
            mte.destroy_mte_group()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
