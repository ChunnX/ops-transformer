# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Multi-NPU correctness and launch-performance validation."""

import argparse
import gc
import json
import os

import torch
import torch.distributed as dist
import torch_npu

import ops_transformer_catccos_udma as udma


def _golden(x, weight, route_matrix, rank, world_size, expert_num):
    gathered = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(gathered, x)
    local_experts = expert_num // world_size
    expert_inputs = []
    for local_expert in range(local_experts):
        pieces = []
        global_expert = rank * local_experts + local_expert
        for src, source_x in enumerate(gathered):
            counts = route_matrix[src]
            offsets = [0]
            for count in counts:
                offsets.append(offsets[-1] + count)
            pieces.append(source_x[offsets[global_expert] : offsets[global_expert + 1]])
        expert_inputs.append(torch.cat(pieces, dim=0))

    valid = torch.cat(
        [part.float().matmul(weight[index].float()) for index, part in enumerate(expert_inputs)],
        dim=0,
    ).to(torch.bfloat16)
    expected = torch.zeros(
        (world_size * x.shape[0], weight.shape[2]), dtype=torch.bfloat16, device=x.device
    )
    expected[: valid.shape[0]].copy_(valid)
    return expected


def _benchmark(call, warmup, iterations):
    for _ in range(warmup):
        call()
    torch_npu.npu.synchronize()
    start = torch_npu.npu.Event(enable_timing=True)
    end = torch_npu.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1_000 / iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--expert-num", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--reference-us", type=float)
    parser.add_argument("--max-regression", type=float, default=0.0)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if args.expert_num % world_size:
        raise ValueError("expert-num must be divisible by world size")

    torch_npu.npu.set_device(local_rank)
    udma.init_udma_group(rank=rank, world_size=world_size, local_rank=local_rank)
    dist.init_process_group("hccl", rank=rank, world_size=world_size)

    torch.manual_seed(2026 + rank)
    device = f"npu:{local_rank}"
    x = torch.randn(args.m, args.k, dtype=torch.bfloat16, device=device)
    weight = torch.randn(
        args.expert_num // world_size, args.k, args.n, dtype=torch.bfloat16, device=device
    )
    local_table, global_table = udma.build_rank_route_tensors(
        args.m, rank, world_size, args.expert_num, device
    )
    route_matrix = udma.build_global_route_matrix(args.m, world_size, args.expert_num)
    expected = _golden(x, weight, route_matrix, rank, world_size, args.expert_num)

    actual = udma.all_to_allv_grouped_mat_mul(
        x,
        weight,
        local_table,
        global_table,
        ep_size=world_size,
        expert_num=args.expert_num,
    )
    torch_npu.npu.synchronize()
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2**-7, atol=2**-7)

    symmetric_x = udma.to_symmetric(x)
    reusable_out = torch.zeros_like(actual)

    def launch():
        return udma.all_to_allv_grouped_mat_mul(
            symmetric_x,
            weight,
            local_table,
            global_table,
            ep_size=world_size,
            expert_num=args.expert_num,
            out=reusable_out,
        )

    symmetric_actual = launch()
    torch_npu.npu.synchronize()
    torch.testing.assert_close(symmetric_actual.float(), expected.float(), rtol=2**-7, atol=2**-7)
    latency_us = _benchmark(launch, args.warmup, args.iterations)
    result = {"rank": rank, "correct": True, "kernel_path_us": latency_us}
    if args.reference_us is not None:
        regression = latency_us / args.reference_us - 1.0
        result["reference_us"] = args.reference_us
        result["regression"] = regression
        if regression > args.max_regression:
            raise AssertionError(
                f"latency regression {regression:.2%} exceeds {args.max_regression:.2%}"
            )
    print(json.dumps(result), flush=True)

    dist.barrier()
    dist.destroy_process_group()
    del symmetric_actual, symmetric_x
    gc.collect()
    udma.destroy_udma_group()


if __name__ == "__main__":
    main()
