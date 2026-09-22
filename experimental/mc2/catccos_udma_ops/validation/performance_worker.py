# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Issue phase-aligned operator launches for collection by msprof."""

from __future__ import annotations

import argparse
import gc
import os
import sys
from itertools import accumulate
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu

import ops_transformer_catccos_udma as udma

try:
    from .case_schema import read_resolved_cases
    from .common import udma_init_method_from_env, validation_udma_heap_size, write_json
except ImportError:  # Direct execution keeps the source tree out of package lookup.
    from case_schema import read_resolved_cases
    from common import udma_init_method_from_env, validation_udma_heap_size, write_json


def _make_tensor(shape, distribution, device, generator):
    result = torch.empty(shape, dtype=torch.float32)
    if distribution["kind"] == "uniform":
        result.uniform_(distribution["low"], distribution["high"], generator=generator)
    else:
        result.normal_(distribution["mean"], distribution["std"], generator=generator)
    return result.to(dtype=torch.bfloat16, device=device)


def _gmm_group_sizes(case, rank):
    """Sum source-rank segments for each local expert, in packed-x order."""
    world_size = case["world_size"]
    local_experts = case["expert_num"] // world_size
    _, global_table = udma.build_rank_route_tables(
        case["shape"]["m"], rank, world_size, case["expert_num"],
    )
    sizes = [
        sum(global_table[source * local_experts + expert] for source in range(world_size))
        for expert in range(local_experts)
    ]
    if sum(sizes) != case["shape"]["m"]:
        raise ValueError("GMM group sizes do not cover the packed input rows")
    return sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-case", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    cases = read_resolved_cases(args.resolved_case)
    if len(cases) != 1:
        raise ValueError("performance worker requires exactly one resolved case")
    case = cases[0]
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if case["world_size"] != world_size:
        raise ValueError("resolved case does not match torchrun world size")
    device = "npu:{}".format(local_rank)
    shape = case["shape"]
    performance = case["performance"]
    gmm_only = performance.get("gmm_only", False)
    if gmm_only and not callable(getattr(torch_npu, "npu_grouped_matmul", None)):
        raise RuntimeError("--gmm-only requires torch_npu.npu_grouped_matmul")
    initialized_dist = False
    initialized_udma = False
    x = None
    out = None
    all_to_allv_out = None
    gmm_metadata = {}
    try:
        torch_npu.npu.set_device(local_rank)
        if not dist.is_gloo_available():
            raise RuntimeError(
                "validation requires PyTorch Gloo support for host-side rank coordination"
            )
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        initialized_dist = True
        dist.barrier()
        if not gmm_only:
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
        generator = torch.Generator(device="cpu")
        generator.manual_seed(case["seed"] + rank)
        x = _make_tensor((shape["m"], shape["k"]), case["input_distribution"], device, generator)
        weight = _make_tensor(
            (case["expert_num"] // world_size, shape["k"], shape["n"]),
            case["input_distribution"], device, generator,
        )
        if not gmm_only:
            local_table, global_table, received_rows = udma.build_rank_route(
                shape["m"], rank, world_size, case["expert_num"], device
            )
        if gmm_only:
            group_sizes = _gmm_group_sizes(case, rank)
            # Build cumulative expert boundaries once, outside all timed calls.
            group_list_values = list(accumulate(group_sizes))
            group_list = torch.tensor(group_list_values, dtype=torch.int64, device=device)
            x_list, weight_list = [x], [weight]
            gmm_metadata = {
                "gmm_api": "torch_npu.npu_grouped_matmul",
                "gmm_input_shape": list(x.shape),
                "gmm_weight_shape": list(weight.shape),
                "gmm_output_shape": [shape["m"], shape["n"]],
                "gmm_group_sizes": group_sizes,
                "gmm_group_list": group_list_values,
                "gmm_split_item": 3, "gmm_group_type": 0, "gmm_group_list_type": 0,
                "gmm_reusable_output": False,
            }
            print("GMM-only rank {}: {}".format(rank, gmm_metadata), flush=True)

            def launch():
                # The public torch_npu GMM API has no out= overload. Keep only
                # the latest output; input, weight and group_list are reused.
                nonlocal out
                out = torch_npu.npu_grouped_matmul(
                    x_list, weight_list, group_list=group_list,
                    split_item=3, group_type=0, group_list_type=0,
                )[0]

        elif case["operator"] == "grouped_mat_mul_all_to_allv":
            out = torch.empty(
                (shape["m"], shape["n"]),
                dtype=torch.bfloat16, device=device,
            )

            def launch():
                udma.grouped_mat_mul_all_to_allv(
                    x, weight, local_table, global_table,
                    ep_size=world_size, expert_num=case["expert_num"], out=out,
                )
        else:
            out = torch.empty(
                (received_rows, shape["n"]),
                dtype=torch.bfloat16, device=device,
            )
            output_all_to_allv = case["output_all_to_allv"]
            if output_all_to_allv:
                all_to_allv_out = torch.empty(
                    (received_rows, shape["k"]),
                    dtype=torch.bfloat16, device=device,
                )

            def launch():
                udma.all_to_allv_grouped_mat_mul(
                    x, weight, local_table, global_table,
                    ep_size=world_size, expert_num=case["expert_num"], out=out,
                    received_rows=received_rows,
                    output_all_to_allv=output_all_to_allv,
                    all_to_allv_out=all_to_allv_out,
                )

        # All target launches are deliberately countable. The parser removes
        # warm-up and primer occurrences rather than timing Python-side scopes.
        for _ in range(performance["warmup"]):
            launch()
        torch_npu.npu.synchronize()
        for _ in range(performance["repeats"]):
            dist.barrier()
            for _ in range(performance["align_iterations"]):
                launch()
            for _ in range(performance["iterations"]):
                launch()
            torch_npu.npu.synchronize()
        dist.barrier()
        write_json(Path(args.output_dir) / "worker_rank_{}.json".format(rank), {
            "case_id": case["id"], "rank": rank, "device": local_rank,
            "source_operator": case["operator"],
            "performance_target": "grouped_matmul" if gmm_only else case["operator"],
            **gmm_metadata,
            "warmup": performance["warmup"],
            "align_iterations": performance["align_iterations"],
            "iterations": performance["iterations"],
            "repeats": performance["repeats"],
            "output_all_to_allv": case["output_all_to_allv"],
            "expected_target_launches": performance["warmup"] + performance["repeats"] * (
                performance["align_iterations"] + performance["iterations"]
            ),
        })
    finally:
        del all_to_allv_out, out, x
        gc.collect()
        if initialized_udma:
            udma.destroy_udma_group()
        if initialized_dist:
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
