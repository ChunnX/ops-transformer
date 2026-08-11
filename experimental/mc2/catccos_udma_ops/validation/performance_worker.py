# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Issue phase-aligned operator launches for collection by msprof."""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu

import ops_transformer_catccos_udma as udma

try:
    from .case_schema import read_resolved_cases
    from .common import udma_init_method_from_env, write_json
except ImportError:  # Direct execution keeps the source tree out of package lookup.
    from case_schema import read_resolved_cases
    from common import udma_init_method_from_env, write_json


def _make_tensor(shape, distribution, device, generator):
    result = torch.empty(shape, dtype=torch.float32)
    if distribution["kind"] == "uniform":
        result.uniform_(distribution["low"], distribution["high"], generator=generator)
    else:
        result.normal_(distribution["mean"], distribution["std"], generator=generator)
    return result.to(dtype=torch.bfloat16, device=device)


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
    initialized_dist = False
    initialized_udma = False
    symmetric_x = None
    out = None
    try:
        torch_npu.npu.set_device(local_rank)
        init_method = udma_init_method_from_env()
        if rank == 0:
            print("UDMA bootstrap: {}".format(init_method), flush=True)
        udma.init_udma_group(
            rank=rank, world_size=world_size, local_rank=local_rank,
            init_method=init_method,
        )
        initialized_udma = True
        dist.init_process_group("hccl", rank=rank, world_size=world_size)
        initialized_dist = True
        generator = torch.Generator(device="cpu")
        generator.manual_seed(case["seed"] + rank)
        x = _make_tensor((shape["m"], shape["k"]), case["input_distribution"], device, generator)
        weight = _make_tensor(
            (case["expert_num"] // world_size, shape["k"], shape["n"]),
            case["input_distribution"], device, generator,
        )
        local_table, global_table = udma.build_rank_route_tensors(
            shape["m"], rank, world_size, case["expert_num"], device
        )
        symmetric_x = udma.to_symmetric(x)
        out = torch.empty(
            (world_size * shape["m"], shape["n"]),
            dtype=torch.bfloat16, device=device,
        )

        def launch():
            udma.all_to_allv_grouped_mat_mul(
                symmetric_x, weight, local_table, global_table,
                ep_size=world_size, expert_num=case["expert_num"], out=out,
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
            "warmup": performance["warmup"],
            "align_iterations": performance["align_iterations"],
            "iterations": performance["iterations"],
            "repeats": performance["repeats"],
            "expected_target_launches": performance["warmup"] + performance["repeats"] * (
                performance["align_iterations"] + performance["iterations"]
            ),
        })
    finally:
        if initialized_dist:
            dist.destroy_process_group()
        del out, symmetric_x
        gc.collect()
        if initialized_udma:
            udma.destroy_udma_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
