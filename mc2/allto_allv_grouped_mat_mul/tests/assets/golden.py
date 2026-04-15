# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
__golden__ = {
        "kernel": {
            "allto_allv_grouped_mat_mul": "allto_allv_grouped_mat_mul_golden"
        }
}

import torch
import torch.distributed as dist


def _permute_from_recv_counts(tokens: torch.Tensor, recv_counts, expert_per_rank: int, ep_world_size: int):
    """
    对齐 MC2_test 的 permute_cpu 语义：
    recv_counts 的布局为 [rank0_exp0..expE-1, rank1_exp0.., ...]
    """
    recv = torch.as_tensor(recv_counts, dtype=torch.int64)
    if recv.numel() != ep_world_size * expert_per_rank:
        raise ValueError("recv_counts size must be ep_world_size * expert_per_rank.")

    flat = recv.reshape(ep_world_size, expert_per_rank).reshape(-1)
    csum = torch.cumsum(flat, dim=0)
    seg_ranges = []
    for i in range(csum.numel()):
        start = 0 if i == 0 else int(csum[i - 1].item())
        end = int(csum[i].item())
        seg_ranges.append((start, end))

    out = []
    for e in range(expert_per_rank):
        idx = []
        for r in range(ep_world_size):
            s, t = seg_ranges[e + r * expert_per_rank]
            idx.extend(range(s, t))
        index = torch.tensor(idx, dtype=torch.long, device=tokens.device)
        out.append(tokens.index_select(0, index))
    return torch.stack(out, dim=0)


def allto_allv_grouped_mat_mul_golden(gmm_x,gmm_weight,send_counts_tensor,
                                      recv_counts_tensor,mm_x,mm_weight,
                                      group,ep_world_size,send_counts,recv_counts,
                                      trans_gmm_weight:bool=False,trans_mm_weight:bool=False,
                                      permute_out_flag:bool=False,**kwargs):
    del send_counts_tensor
    del recv_counts_tensor
    del group

    x = torch.as_tensor(gmm_x)
    w_gmm = torch.as_tensor(gmm_weight)
    mmx = None if mm_x is None else torch.as_tensor(mm_x)
    mmw = None if mm_weight is None else torch.as_tensor(mm_weight)

    if x.dim() != 2 or w_gmm.dim() != 3:
        raise ValueError("gmm_x must be 2-D and gmm_weight must be 3-D.")
    if ep_world_size <= 0:
        raise ValueError("ep_world_size must be positive.")

    expert_per_rank = int(w_gmm.shape[0])
    send_counts_t = torch.as_tensor(send_counts, dtype=torch.int64)
    recv_counts_t = torch.as_tensor(recv_counts, dtype=torch.int64)
    if send_counts_t.numel() != ep_world_size * expert_per_rank:
        raise ValueError("send_counts size must be ep_world_size * expert_per_rank.")
    if recv_counts_t.numel() != ep_world_size * expert_per_rank:
        raise ValueError("recv_counts size must be ep_world_size * expert_per_rank.")

    input_splits = send_counts_t.reshape(ep_world_size, expert_per_rank).sum(dim=1).tolist()
    output_splits = recv_counts_t.reshape(ep_world_size, expert_per_rank).sum(dim=1).tolist()

    alltoall_out = torch.empty((int(sum(output_splits)), x.shape[1]), dtype=x.dtype, device=x.device)
    if dist.is_available() and dist.is_initialized():
        dist.all_to_all_single(
            alltoall_out,
            x,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
        )
    else:
        # 单机回退：仅用于无分布式环境下的可运行参考
        alltoall_out = x[: alltoall_out.shape[0]]

    permute_out = _permute_from_recv_counts(alltoall_out, recv_counts_t, expert_per_rank, ep_world_size)

    w_gmm_use = w_gmm.transpose(1, 2).contiguous() if trans_gmm_weight else w_gmm
    gmm_y = torch.matmul(permute_out.to(torch.float32), w_gmm_use.to(torch.float32)).to(x.dtype)

    mm_y = None
    if mmx is not None and mmw is not None:
        mmw_use = mmw.transpose(0, 1).contiguous() if trans_mm_weight else mmw
        mm_y = torch.mm(mmx.to(torch.float32), mmw_use.to(torch.float32)).to(x.dtype)

    return gmm_y, mm_y, (permute_out if permute_out_flag else None)
