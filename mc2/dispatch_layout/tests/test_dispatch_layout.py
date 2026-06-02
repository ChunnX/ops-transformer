# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
DispatchLayout standalone smoke test (no HCCL / no notify_dispatch).

Prerequisites on the NPU machine:
  1. Install the custom run package that contains dispatch_layout device kernel.
  2. pip install npu_ops_transformer whl (JIT builds the ACLNN wrapper on first import).

Examples:
  # single card (default)
  python mc2/dispatch_layout/tests/test_dispatch_layout.py

  # multi-rank EP layout (each rank runs layout locally; no collective)
  torchrun --nproc_per_node=2 mc2/dispatch_layout/tests/test_dispatch_layout.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch_npu
import npu_ops_transformer  # noqa: F401
from npu_ops_transformer.ops.dispatch_layout import npu_dispatch_layout
from npu_ops_transformer.ops.dispatch_layout import op_module as dispatch_layout_op_module

MAX_BATCH_SIZE = 4096
EXPERT_DATA_SIZE = 1 + MAX_BATCH_SIZE


def init_npu():
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    torch_npu.npu.set_device(local_rank)
    return local_rank, rank, world_size


def notify_send_data_size(num_experts: int, num_ranks: int, local_ranksize: int, num_topk: int) -> int:
    server_num = num_ranks // local_ranksize
    return (
        num_experts * EXPERT_DATA_SIZE
        + server_num
        + MAX_BATCH_SIZE * (1 + 2 * server_num + 2 * num_topk)
    )


def reference_layout_cpu(
    topk_idx: torch.Tensor,
    num_experts: int,
    num_ranks: int,
    per_round_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mirror dispatch_layout.h counting rules on CPU for golden check."""
    num_tokens, num_topk = topk_idx.shape
    experts_per_rank = num_experts // num_ranks
    round_num = (num_tokens + per_round_tokens - 1) // per_round_tokens

    num_tokens_per_rank = torch.zeros((num_ranks,), dtype=torch.int32)
    num_tokens_per_expert = torch.zeros((round_num * num_experts,), dtype=torch.int32)
    is_token_in_rank = torch.zeros((round_num * per_round_tokens, num_ranks), dtype=torch.int32)

    for r in range(round_num):
        start = r * per_round_tokens
        end = min(start + per_round_tokens, num_tokens)
        for t in range(start, end):
            seen_rank = [False] * num_ranks
            row = r * per_round_tokens + (t - start)
            for j in range(num_topk):
                expert_idx = int(topk_idx[t, j].item())
                if expert_idx < 0 or expert_idx >= num_experts:
                    continue
                flat = r * num_experts + expert_idx
                num_tokens_per_expert[flat] += 1
                rank_id = expert_idx // experts_per_rank
                if not seen_rank[rank_id]:
                    seen_rank[rank_id] = True
                    num_tokens_per_rank[rank_id] += 1
                    is_token_in_rank[row, rank_id] = 1
    return num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank


def assert_layout_outputs(
    topk_idx: torch.Tensor,
    num_experts: int,
    num_ranks: int,
    local_ranksize: int,
    per_round_tokens: int,
    num_tokens_per_rank: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    is_token_in_rank: torch.Tensor,
    notify_send_data: torch.Tensor,
    send_token_idx_small: torch.Tensor,
):
    num_tokens, num_topk = topk_idx.shape
    round_num = (num_tokens + per_round_tokens - 1) // per_round_tokens

    assert num_tokens_per_rank.shape == (num_ranks,)
    assert num_tokens_per_rank.dtype == torch.int32
    assert num_tokens_per_expert.shape == (round_num * num_experts,)
    assert num_tokens_per_expert.dtype == torch.int32
    assert is_token_in_rank.shape == (round_num * per_round_tokens, num_ranks)
    assert is_token_in_rank.dtype == torch.int32
    assert notify_send_data.dtype == torch.int32
    assert send_token_idx_small.shape == (num_tokens, num_topk)
    assert send_token_idx_small.dtype == torch.int32

    expected_notify_size = notify_send_data_size(num_experts, num_ranks, local_ranksize, num_topk)
    assert notify_send_data.numel() == expected_notify_size, (
        f"notify_send_data size {notify_send_data.numel()} != expected {expected_notify_size}"
    )

    ref_rank, ref_expert, ref_is = reference_layout_cpu(
        topk_idx.cpu(), num_experts, num_ranks, per_round_tokens
    )
    torch.testing.assert_close(
        num_tokens_per_rank.cpu(), ref_rank, rtol=0, atol=0, msg="num_tokens_per_rank mismatch"
    )
    torch.testing.assert_close(
        num_tokens_per_expert.cpu(), ref_expert, rtol=0, atol=0, msg="num_tokens_per_expert mismatch"
    )
    torch.testing.assert_close(
        is_token_in_rank.cpu(), ref_is, rtol=0, atol=0, msg="is_token_in_rank mismatch"
    )


def run_op_module(
    topk_idx: torch.Tensor,
    num_experts: int,
    num_ranks: int,
    local_ranksize: int,
    per_round_tokens: int,
    rank_id: int,
):
    (
        num_tokens_per_rank,
        _,
        num_tokens_per_expert,
        is_token_in_rank,
        ext_handle_data,
    ) = dispatch_layout_op_module.npu_dispatch_layout(
        topk_idx,
        num_experts,
        num_ranks,
        local_ranksize,
        per_round_tokens,
        rank_id,
    )
    notify_send_data, send_token_idx_small = ext_handle_data[0], ext_handle_data[1]
    torch.npu.synchronize()
    assert_layout_outputs(
        topk_idx,
        num_experts,
        num_ranks,
        local_ranksize,
        per_round_tokens,
        num_tokens_per_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        notify_send_data,
        send_token_idx_small,
    )
    return num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank


def run_torch_api(
    topk_idx: torch.Tensor,
    num_experts: int,
    num_ranks: int,
    local_ranksize: int,
    per_round_tokens: int,
    rank_id: int,
):
    num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, event = npu_dispatch_layout(
        topk_idx,
        num_experts,
        num_ranks,
        local_ranksize,
        per_round_tokens,
        rank_id,
    )
    assert event is None
    torch.npu.synchronize()
    round_num = (topk_idx.size(0) + per_round_tokens - 1) // per_round_tokens
    assert num_tokens_per_rank.shape == (num_ranks,)
    assert num_tokens_per_expert.shape == (round_num * num_experts,)
    assert is_token_in_rank.shape == (round_num * per_round_tokens, num_ranks)
    return num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank


def main():
    _, rank, world_size = init_npu()

    # EP topology: one node with `world_size` cards is the common smoke setup.
    num_ranks = int(os.getenv("DISPATCH_LAYOUT_NUM_RANKS", str(world_size)))
    local_ranksize = int(os.getenv("DISPATCH_LAYOUT_LOCAL_RANKSIZE", str(world_size)))
    rank_id = int(os.getenv("DISPATCH_LAYOUT_RANK_ID", str(rank)))
    num_experts = int(os.getenv("DISPATCH_LAYOUT_NUM_EXPERTS", "16"))
    per_round_tokens = int(os.getenv("DISPATCH_LAYOUT_PER_ROUND_TOKENS", "1024"))
    num_tokens = int(os.getenv("DISPATCH_LAYOUT_NUM_TOKENS", "8"))
    topk = int(os.getenv("DISPATCH_LAYOUT_TOPK", "4"))

    if num_experts % num_ranks != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by num_ranks={num_ranks}")
    if not (0 <= rank_id < num_ranks):
        raise ValueError(f"rank_id={rank_id} must be in [0, {num_ranks})")
    if num_ranks % local_ranksize != 0:
        raise ValueError(f"num_ranks={num_ranks} must be divisible by local_ranksize={local_ranksize}")

    print(
        f"[rank {rank}/{world_size}] dispatch_layout: "
        f"num_tokens={num_tokens} topk={topk} num_experts={num_experts} "
        f"num_ranks={num_ranks} local_ranksize={local_ranksize} "
        f"per_round_tokens={per_round_tokens} rank_id={rank_id}",
        flush=True,
    )

    # Deterministic small case (golden vs CPU reference).
    topk_det = torch.tensor(
        [[0, 1], [2, 3], [-1, 0], [3, 3]],
        dtype=torch.int64,
        device="npu",
    )
    det_experts = 4
    det_ranks = 2
    det_local = 2
    det_per_round = 2
    run_op_module(topk_det, det_experts, det_ranks, det_local, det_per_round, rank_id=0)

    # Random stress shape (same params as notify_dispatch smoke defaults).
    topk_idx = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int64, device="npu")
    mod_rank, mod_expert, mod_is = run_op_module(
        topk_idx, num_experts, num_ranks, local_ranksize, per_round_tokens, rank_id
    )
    api_rank, api_expert, api_is = run_torch_api(
        topk_idx, num_experts, num_ranks, local_ranksize, per_round_tokens, rank_id
    )
    torch.testing.assert_close(mod_rank.cpu(), api_rank.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(mod_expert.cpu(), api_expert.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(mod_is.cpu(), api_is.cpu(), rtol=0, atol=0)

    print(
        f"[rank {rank}] OK num_tokens_per_rank={mod_rank.tolist()} "
        f"num_tokens_per_expert_sum={int(mod_expert.sum().item())} "
        f"is_token_in_rank_sum={int(mod_is.sum().item())}",
        flush=True,
    )

    if rank == 0:
        print("[INFO] dispatch_layout standalone test finished.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        raise
