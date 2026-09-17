#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Run one HCCL all-gather, then call AllToAllV-GroupedMatMul with UDMA."""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch_npu

import ops_transformer_catccos_udma as udma


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=int(os.environ.get("A2AV_GMM_M", 128)))
    parser.add_argument("--k", type=int, default=int(os.environ.get("A2AV_GMM_K", 256)))
    parser.add_argument("--n", type=int, default=int(os.environ.get("A2AV_GMM_N", 256)))
    parser.add_argument(
        "--expert-num",
        type=int,
        default=int(os.environ.get("A2AV_GMM_EXPERT_NUM", 8)),
    )
    parser.add_argument(
        "--hccl-elements",
        type=int,
        default=int(os.environ.get("HCCL_ALL_GATHER_ELEMENTS", 1024)),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=int(os.environ.get("A2AV_GMM_ITERATIONS", 1)),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--heap-size",
        type=int,
        default=int(os.environ.get("UDMA_HEAP_SIZE", 1 << 30)),
        help="symmetric UDMA heap size in bytes (default: 1 GiB)",
    )
    return parser.parse_args()


def run_hccl_all_gather(
    rank: int, world_size: int, device: torch.device, elements: int
) -> None:
    """Run and validate one device-side HCCL all-gather."""
    value = torch.full((elements,), rank, dtype=torch.int32, device=device)
    gathered = [torch.empty_like(value) for _ in range(world_size)]
    dist.all_gather(gathered, value)
    torch_npu.npu.synchronize()

    for peer_rank, peer_value in enumerate(gathered):
        expected = torch.full_like(peer_value, peer_rank)
        if not torch.equal(peer_value, expected):
            raise RuntimeError(
                "HCCL all_gather validation failed for peer rank {}".format(peer_rank)
            )
    print(
        "rank={} hccl_all_gather_passed=True input_shape={} world_size={}".format(
            rank, tuple(value.shape), world_size
        ),
        flush=True,
    )


def main() -> int:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if min(
        args.m,
        args.k,
        args.n,
        args.expert_num,
        args.hccl_elements,
        args.iterations,
    ) <= 0:
        raise ValueError("all shape, expert, and iteration arguments must be positive")
    if args.expert_num % world_size:
        raise ValueError("expert_num must be divisible by world_size")
    if args.expert_num // world_size > 64:
        raise ValueError("local expert count must not exceed 64")

    device = torch.device("npu", local_rank)
    torch_npu.npu.set_device(local_rank)
    torch.manual_seed(args.seed + rank)

    hccl_initialized = False
    udma_initialized = False
    try:
        # Initialize and exercise HCCL before ACLSHMEM/UDMA initialization.
        # Keep the HCCL process group alive while the UDMA operator runs.
        dist.init_process_group(
            backend="hccl", rank=rank, world_size=world_size
        )
        hccl_initialized = True
        run_hccl_all_gather(rank, world_size, device, args.hccl_elements)
        dist.barrier()
        torch_npu.npu.synchronize()

        config = udma.init_udma_group(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            heap_size=args.heap_size,
        )
        udma_initialized = True

        local_expert_num = args.expert_num // world_size
        x = torch.randn(
            (args.m, args.k), dtype=torch.bfloat16, device=device
        ).contiguous()
        weight = torch.randn(
            (local_expert_num, args.k, args.n),
            dtype=torch.bfloat16,
            device=device,
        ).contiguous()
        local_tokens, global_tokens, received_rows = udma.build_rank_route(
            args.m, rank, world_size, args.expert_num, device
        )
        out = torch.empty(
            (received_rows, args.n),
            dtype=torch.bfloat16,
            device=device,
        )

        result = out
        for _ in range(args.iterations):
            result = udma.all_to_allv_grouped_mat_mul(
                x,
                weight,
                local_tokens,
                global_tokens,
                ep_size=world_size,
                expert_num=args.expert_num,
                received_rows=received_rows,
                out=out,
            )
        torch_npu.npu.synchronize()

        sample = result[: min(received_rows, 8), : min(args.n, 8)]
        print(
            "rank={} udma_operator_passed=True bootstrap={} output_shape={} "
            "valid_rows={} reused_out={} sample_sum={:.6f}".format(
                rank,
                config.init_method,
                tuple(result.shape),
                received_rows,
                result.data_ptr() == out.data_ptr(),
                float(sample.float().sum().item()),
            ),
            flush=True,
        )
        return 0
    finally:
        # UDMA was initialized after HCCL, so release it first.
        if udma_initialized:
            udma.destroy_udma_group()
        if hccl_initialized:
            dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
