#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Call AllToAllV-GroupedMatMul once per NPU through the installed Wheel."""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch_npu

import ops_transformer_catccos_udma as udma


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--expert-num", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--heap-size",
        type=int,
        default=int(os.environ.get("UDMA_HEAP_SIZE", 1 << 30)),
        help="symmetric UDMA heap size in bytes (default: 1 GiB)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if min(args.m, args.k, args.n, args.expert_num, args.iterations) <= 0:
        raise ValueError("m, k, n, expert_num, and iterations must be positive")
    if args.expert_num % world_size:
        raise ValueError("expert_num must be divisible by world_size")
    if args.expert_num // world_size > 64:
        raise ValueError("local expert count must not exceed 64")

    device = torch.device("npu", local_rank)
    torch_npu.npu.set_device(local_rank)
    torch.manual_seed(args.seed + rank)

    initialized = False
    try:
        config = udma.init_udma_group(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            heap_size=args.heap_size,
        )
        initialized = True

        local_expert_num = args.expert_num // world_size
        x = torch.randn(
            (args.m, args.k), dtype=torch.bfloat16, device=device
        ).contiguous()
        weight = torch.randn(
            (local_expert_num, args.k, args.n),
            dtype=torch.bfloat16,
            device=device,
        ).contiguous()

        # This helper creates balanced test routing. In a model, replace these
        # two tensors with the actual router's contiguous int64 count tables.
        local_tokens, global_tokens = udma.build_rank_route_tensors(
            args.m, rank, world_size, args.expert_num, device
        )
        valid_rows = int(global_tokens.sum().item())

        # The operator's capacity output is [world_size * m, n]. Reuse an
        # ordinary NPU tensor to keep allocation outside repeated launches.
        out = torch.zeros(
            (world_size * args.m, args.n),
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
                out=out,
            )
        torch_npu.npu.synchronize()

        sample = result[: min(valid_rows, 8), : min(args.n, 8)]
        sample_sum = float(sample.float().sum().item())
        print(
            "rank={} device={} bootstrap={} output_shape={} valid_rows={} "
            "reused_out={} sample_sum={:.6f}".format(
                rank,
                device,
                config.init_method,
                tuple(result.shape),
                valid_rows,
                result.data_ptr() == out.data_ptr(),
                sample_sum,
            ),
            flush=True,
        )
        return 0
    finally:
        if initialized:
            udma.destroy_udma_group()


if __name__ == "__main__":
    sys.exit(main())
