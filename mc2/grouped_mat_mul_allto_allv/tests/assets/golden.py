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
            "grouped_mat_mul_allto_allv": "grouped_mat_mul_allto_allv_golden"
        }
}

import torch
import torch.distributed as dist


def _grouped_matmul_cpu(gmm_x: torch.Tensor, gmm_weight: torch.Tensor, group_list):
    """对齐 MC2_test grouped_matmul_cpu：按 group_list 切分后逐组 matmul。"""
    weights = list(torch.unbind(gmm_weight, dim=0))
    x_groups = torch.split(gmm_x, group_list, dim=0)
    out = [torch.from_numpy((a.cpu().numpy() @ b.cpu().numpy())) for a, b in zip(x_groups, weights)]
    return torch.cat(out, dim=0)


def _unpermute_from_counts(tokens: torch.Tensor, counts_matrix: torch.Tensor):
    """
    对齐 MC2_test 的 unpermute 逻辑：
    counts_matrix 形状 [ep_world_size, expert_per_card]，表示每个source-rank发到本rank各expert的token数。
    """
    ep_world_size, expert_per_card = counts_matrix.shape
    tmp1 = counts_matrix.transpose(0, 1)  # [expert_per_card, ep_world_size]
    sum_list1 = torch.sum(tmp1, dim=1)
    sum_list2 = torch.cumsum(sum_list1, dim=0)
    offsets = [0] + [int(v.item()) for v in sum_list2[:-1]]
    csum = torch.cumsum(tmp1, dim=1)

    pieces = []
    for i in range(ep_world_size):
        for j in range(expert_per_card):
            start = 0 if i == 0 else int(csum[j, i - 1].item())
            end = int(csum[j, i].item())
            if end <= start:
                continue
            idx = torch.arange(start + offsets[j], end + offsets[j], device=tokens.device, dtype=torch.long)
            pieces.append(tokens.index_select(0, idx))
    if not pieces:
        return tokens.new_zeros((0, tokens.shape[1]))
    return torch.cat(pieces, dim=0).to(tokens.dtype)


def grouped_mat_mul_allto_allv_golden(gmm_x,gmm_weight,send_counts_tensor,recv_counts_tensor,
                                      mm_x,mm_weight,group,ep_world_size,send_counts,recv_counts,
                                      trans_gmm_weight:bool=False,trans_mm_weight:bool=False,**kwargs):
    del send_counts_tensor
    del recv_counts_tensor
    del group

    x = torch.as_tensor(gmm_x)
    w_gmm = torch.as_tensor(gmm_weight)
    x_mm = torch.as_tensor(mm_x) if mm_x is not None else None
    w_mm = torch.as_tensor(mm_weight) if mm_weight is not None else None

    if x.dim() != 2 or w_gmm.dim() != 3:
        raise ValueError("gmm_x must be 2-D and gmm_weight must be 3-D.")
    if ep_world_size <= 0:
        raise ValueError("ep_world_size must be positive.")

    expert_per_card = int(w_gmm.shape[0])
    sc = torch.as_tensor(send_counts, dtype=torch.int64)
    rc = torch.as_tensor(recv_counts, dtype=torch.int64)
    if sc.numel() != ep_world_size * expert_per_card or rc.numel() != ep_world_size * expert_per_card:
        raise ValueError("send_counts/recv_counts length must be ep_world_size * expert_per_card.")

    # 与 MC2_test 命名保持语义：group_list 来自“按source聚合到本rank expert”的计数
    counts_matrix = sc.reshape(ep_world_size, expert_per_card)
    group_list = torch.sum(counts_matrix, dim=0).tolist()
    input_splits = [int(v.item()) for v in torch.sum(counts_matrix, dim=1)]  # all_to_all_single 的 input splits
    output_splits = [int(v.item()) for v in torch.sum(rc.reshape(ep_world_size, expert_per_card), dim=1)]  # output splits

    w_gmm_use = w_gmm.transpose(1, 2).contiguous() if trans_gmm_weight else w_gmm
    gmm_out = _grouped_matmul_cpu(x.to(torch.float32), w_gmm_use.to(torch.float32), group_list).to(x.dtype)

    unpermute_out = _unpermute_from_counts(gmm_out, counts_matrix).to(torch.float32)
    y = torch.empty((sum(output_splits), gmm_out.shape[1]), dtype=torch.float32, device=unpermute_out.device)
    if dist.is_available() and dist.is_initialized():
        dist.all_to_all_single(y, input=unpermute_out, output_split_sizes=output_splits, input_split_sizes=input_splits)
    else:
        y = unpermute_out[: y.shape[0]]
    y = y.to(x.dtype)

    mm_y = None
    if x_mm is not None and w_mm is not None:
        w_mm_use = w_mm.transpose(0, 1).contiguous() if trans_mm_weight else w_mm
        mm_y = torch.mm(x_mm.to(torch.float32), w_mm_use.to(torch.float32)).to(x.dtype)

    return y, mm_y