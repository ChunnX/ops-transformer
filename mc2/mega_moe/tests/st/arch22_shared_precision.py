# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Two-NPU BF16 regression; requires matching custom OPP and Python bindings.

Run with torchrun --standalone --nproc-per-node=2 arch22_shared_precision.py.
Uses an independent CPU FP32 reference, exact expert counts, cosine >= 0.999
and relative L2 error <= 0.01. No threshold is relaxed for shared experts.
"""

import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as functional
import torch_npu

from cann_ops_transformer.ops import get_symm_buffer_for_mega_moe, mega_moe


# (tokens, experts per rank, shared experts, receive budget, active-prefix mask)
CASES = [
    (8, 4, 0, 128, False),
    (8, 4, 2, 128, False),
    (8, 4, 4, 128, False),
    (8, 8, 2, 128, False),
    (512, 8, 0, 128, False),
    (1024, 8, 2, 128, False),
    (1025, 4, 0, 128, False),
    (1025, 4, 2, 128, False),
    (1025, 8, 2, 128, False),
    (1025, 8, 2, 4096, False),
    (1025, 4, 2, 128, True),
]


def ffn(x, w1, w2):
    gate, up = (x.float() @ w1.float()).chunk(2, dim=-1)
    return (functional.silu(gate) * up) @ w2.float()


def run_case(case, seed, rank, group):
    tokens, experts, shared, budget, masked = case
    hidden, intermediate, topk, world = 1024, 512, 2, 2
    generator = torch.Generator().manual_seed(seed)

    def random_bf16(*shape):
        return (torch.randn(*shape, generator=generator) * 0.1).bfloat16()

    # Both ranks generate the same global fixture without device collectives.
    x = random_bf16(world, tokens, hidden)
    ids = torch.randint(world * experts, (world, tokens, topk), generator=generator)
    probs = torch.softmax(torch.randn(world, tokens, topk, generator=generator), dim=-1)
    w1 = random_bf16(world * experts, hidden, 2 * intermediate)
    w2 = random_bf16(world * experts, intermediate, hidden)
    sw1 = random_bf16(shared, hidden, 2 * intermediate)
    sw2 = random_bf16(shared, intermediate, hidden)
    active = torch.ones(world, tokens, dtype=torch.bool)
    if masked:
        # Unequal source counts exercise sender prefixes and an inactive tail chunk.
        active[0, tokens // 2 :] = False
        active[1, tokens - 1 :] = False

    reference = torch.zeros(tokens, hidden)
    for expert in range(world * experts):
        row, slot = torch.where((ids[rank] == expert) & active[rank, :, None])
        if row.numel():
            reference.index_add_(
                0,
                row,
                ffn(x[rank, row], w1[expert], w2[expert])
                * probs[rank, row, slot, None],
            )
    for expert in range(shared):
        # Shared experts are dense, including tokens excluded from routed experts.
        reference += ffn(x[rank], sw1[expert], sw2[expert])
    counts = torch.bincount(ids[active].flatten(), minlength=world * experts)
    expected_counts = counts[rank * experts : (rank + 1) * experts].int()

    def to_device(weights):
        # The BF16 MegaMoe interface requires ND, not quantized weights' NZ format.
        return [weight.npu() for weight in weights]

    local_w1 = to_device(w1[rank * experts : (rank + 1) * experts])
    local_w2 = to_device(w2[rank * experts : (rank + 1) * experts])
    shared_w1, shared_w2 = to_device(sw1), to_device(sw2)
    device_x, device_ids, device_probs = (
        x[rank].npu(),
        ids[rank].int().npu(),
        probs[rank].npu(),
    )
    device_mask = active[rank].to(torch.int8).npu() if masked else None
    buffer = get_symm_buffer_for_mega_moe(
        group=group,
        num_experts=world * experts,
        num_max_tokens_per_rank=tokens,
        num_topk=topk,
        hidden=hidden,
        intermediate_hidden=intermediate,
        max_recv_token_num=budget,
    )
    passed = True
    try:
        for repeat in range(2):
            y, actual_counts = mega_moe(
                device_x,
                device_ids,
                device_probs,
                local_w1,
                local_w2,
                buffer,
                shared_l1_weights=shared_w1 or None,
                shared_l2_weights=shared_w2 or None,
                x_active_mask=device_mask,
            )
            actual = y.cpu().float()
            cosine = functional.cosine_similarity(
                actual.flatten(), reference.flatten(), dim=0
            ).item()
            relative_l2 = ((actual - reference).norm() / reference.norm()).item()
            count_ok = torch.equal(actual_counts.cpu(), expected_counts)
            ok = (
                bool(torch.isfinite(actual).all())
                and cosine >= 0.999
                and relative_l2 <= 0.01
                and count_ok
            )
            passed &= ok
            print(
                f"rank={rank} case={case} repeat={repeat} cosine={cosine:.7f} "
                f"relative_l2={relative_l2:.7f} counts={count_ok} {'PASS' if ok else 'FAIL'}",
                flush=True,
            )
        failed = torch.tensor([not passed], dtype=torch.int32, device="npu")
        dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=group)
        passed = failed.item() == 0
    finally:
        torch.npu.synchronize()
        buffer.destroy()
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--cases", type=int, nargs="+", default=list(range(len(CASES))))
    args = parser.parse_args()
    if any(index < 0 or index >= len(CASES) for index in args.cases):
        parser.error("case index out of range")
    torch.set_num_threads(4)
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    torch.npu.config.allow_internal_format = True
    dist.init_process_group("hccl", timeout=timedelta(minutes=5))
    group = None
    try:
        if dist.get_world_size() != 2:
            raise ValueError("This regression requires exactly two ranks")
        group = dist.new_group(backend="hccl")
        # SymmBuffer requests an existing communicator (init_comm=False).
        group._get_backend(torch.device("npu")).get_hccl_comm_name(
            dist.get_rank(), init_comm=True
        )
        passed = True
        for index in args.cases:
            passed &= run_case(CASES[index], args.seed + index, dist.get_rank(), group)
        if not passed:
            raise AssertionError("BF16 precision or expert-count regression")
    finally:
        if group is not None:
            dist.destroy_process_group(group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
