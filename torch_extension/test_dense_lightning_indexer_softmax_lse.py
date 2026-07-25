# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import torch_npu
import cann_ops_transformer
import os
import ast
import argparse
import math

try:
    import openpyxl
except ImportError:
    openpyxl = None


def _golden_dense_lightning_index_lse(query_index, key_index, weight, device="cpu"):
    """Golden: LSE of (relu(q@k^T) * weight) summed over N1, then LogSumExp over S2."""
    q = query_index.float().to(device)
    k = key_index.float().to(device)
    w = weight.float().to(device)
    # q: [B, S1, N1, D], k: [B, S2, N2, D]
    score = torch.matmul(q, k.permute(0, 2, 3, 1))  # [B, S1, N1, S2]
    relu_score = torch.relu(score)
    weighted = relu_score * w.unsqueeze(-1)          # [B, S1, N1, S2]
    x = weighted.sum(dim=-2)                          # [B, S1, S2]
    m = x.max(dim=-1).values                          # [B, S1]
    lse = m + torch.log(torch.exp(x - m.unsqueeze(-1)).sum(dim=-1))  # [B, S1]
    return lse.unsqueeze(1)  # [B, N2=1, S1]

def _compute_row_valid_mask_blk(B, S1, S2, seqused_q, seqused_k, cmp_residual_k,
                                mask_mode, cmp_ratio, device):
    """BSND: 返回 [B, S1] bool，True=有效行（需计算LSE），False=无效行（kernel 可不写出）。

    判断逻辑与 golden_dli_lse_blk 内部一致：
      - q 超出 seqused_q 的行无效
      - mask_mode=3 时 valid_k_size=0 的行无效
      - 其他情况 k_used>0 即有效
    """
    row_valid = torch.zeros(B, S1, dtype=torch.bool, device="cpu")
    for b in range(B):
        q_used = S1 if seqused_q is None else seqused_q[b, 0].item()
        k_used = S2 if seqused_k is None else seqused_k[b, 0].item()
        for s1_i in range(min(q_used, S1)):
            if mask_mode == 3:
                cur_residual = 0
                if cmp_ratio > 1:
                    cur_residual = 0 if cmp_residual_k is None else cmp_residual_k[b, 0].item()
                ori_k_size = k_used * cmp_ratio + cur_residual
                vk = max((s1_i + 1 + ori_k_size - q_used) // cmp_ratio, 0)
                row_valid[b, s1_i] = vk > 0
            else:
                row_valid[b, s1_i] = k_used > 0
    return row_valid.to(device)


def _compute_row_valid_mask_tnd(cu_seqlens_q, cu_seqlens_k, T1,
                                seqused_q, seqused_k, cmp_residual_k,
                                mask_mode, cmp_ratio, device):
    """TND: 返回 [T1] bool，True=有效行。逻辑同 _compute_row_valid_mask_blk。"""
    B = cu_seqlens_q.shape[0] - 1
    row_valid = torch.zeros(T1, dtype=torch.bool, device="cpu")
    for b in range(B):
        q_start = cu_seqlens_q[b].item()
        q_end = cu_seqlens_q[b + 1].item()
        k_start = cu_seqlens_k[b].item()
        k_end = cu_seqlens_k[b + 1].item()
        q_len = q_end - q_start
        k_len = k_end - k_start
        q_used = q_len if seqused_q is None else seqused_q[b, 0].item()
        k_used = k_len if seqused_k is None else seqused_k[b, 0].item()
        for s1_i in range(min(q_used, q_len)):
            if mask_mode == 3:
                cur_residual = 0
                if cmp_ratio > 1:
                    cur_residual = 0 if cmp_residual_k is None else cmp_residual_k[b, 0].item()
                ori_k_size = k_used * cmp_ratio + cur_residual
                vk = max((s1_i + 1 + ori_k_size - q_used) // cmp_ratio, 0)
                row_valid[q_start + s1_i] = vk > 0
            else:
                row_valid[q_start + s1_i] = vk > 0
    return row_valid.to(device)


def golden_dli_lse_blk(query_index, key_index, weight, seqused_q=None, seqused_k=None,
                       cmp_residual_k=None, mask_mode=0, cmp_ratio=1, device="cpu"):
    q = query_index.float().to(device)
    k = key_index.float().to(device)
    w = weight.float().to(device)
    B, S1, N1, D = q.shape
    _, S2, _, _ = k.shape
    dev = device

    TKV = 128
    neg_inf = -torch.inf

    row_valid_all = _compute_row_valid_mask_blk(B, S1, S2, seqused_q, seqused_k,
                                                cmp_residual_k, mask_mode, cmp_ratio, dev)
    lse = torch.zeros(B, S1, dtype=torch.float32, device=dev)

    for b in range(B):
        q_b = q[b]
        k_b = k[b]
        w_b = w[b]

        q_used = S1 if seqused_q is None else seqused_q[b, 0].item()
        k_used = S2 if seqused_k is None else seqused_k[b, 0].item()

        x_full = torch.full((S1, S2), neg_inf, dtype=torch.float32, device=dev)

        s2_tiles = (k_used + TKV - 1) // TKV

        for s2_i in range(s2_tiles):
            s2_start = s2_i * TKV
            s2_end = min(s2_start + TKV, k_used)
            s2_len = s2_end - s2_start

            k_slice = k_b[s2_start:s2_end, 0, :]
            if s2_len < TKV:
                k_tile = torch.cat([k_slice, torch.zeros(TKV - s2_len, D, dtype=torch.float32, device=dev)], dim=0)
            else:
                k_tile = k_slice

            score_tile = torch.matmul(q_b, k_tile.T)
            relu_score = torch.relu(score_tile)
            weighted = relu_score * w_b.unsqueeze(-1)
            reduce_sum = weighted.sum(dim=1)

            x_full[:, s2_start:s2_end] = reduce_sum[:, :s2_len]

        if mask_mode == 3:
            if cmp_ratio > 1:
                cur_residual = 0 if cmp_residual_k is None else cmp_residual_k[b, 0].item()
                ori_k_size = k_used * cmp_ratio + cur_residual
            else:
                ori_k_size = k_used * cmp_ratio
            for s1_i in range(S1):
                if s1_i >= q_used:
                    continue
                valid_k_size = max((s1_i + 1 + ori_k_size - q_used) // cmp_ratio, 0)
                if valid_k_size < k_used:
                    x_full[s1_i, valid_k_size:] = neg_inf

        row_valid = row_valid_all[b]

        reduce_max = x_full.amax(dim=1)
        shifted = x_full - reduce_max.unsqueeze(1)
        sum_exp = torch.exp(shifted).sum(dim=1)
        lse[b] = torch.where(row_valid, reduce_max + torch.log(sum_exp), torch.full_like(reduce_max, neg_inf))

    return lse.unsqueeze(1)  # [B, N2=1, S1]

def test_bsnd_metadata():
    """BSND layout: 调用 metadata 算子。"""
    B, S1, S2, N1, D = 2, 128, 256, 8, 64
    metadata = cann_ops_transformer.dense_lightning_indexer_softmax_lse_metadata(
        num_heads_q=N1,
        num_heads_k=1,
        head_dim=D,
        batch_size=B,
        max_seqlen_q=S1,
        max_seqlen_k=S2,
        layout_q="BSND",
        layout_k="BSND",
        mask_mode=0,
        cmp_ratio=1,
    )
    assert metadata.shape == (64,)
    assert metadata.dtype == torch.int32
    vals = metadata.cpu().tolist()
    print(f"[BSND] metadata shape={metadata.shape}, dtype={metadata.dtype}")
    print(f"[BSND] metadata[0:5]={vals[:5]}")


def test_tnd_metadata():
    """TND layout: 调用 metadata 算子。"""
    T1, T2, N1, D = 256, 512, 8, 64
    cu_seqlens_q = torch.tensor([0, 128, 256], dtype=torch.int32).npu()
    cu_seqlens_k = torch.tensor([0, 256, 512], dtype=torch.int32).npu()
    metadata = cann_ops_transformer.dense_lightning_indexer_softmax_lse_metadata(
        num_heads_q=N1,
        num_heads_k=1,
        head_dim=D,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        layout_q="TND",
        layout_k="TND",
        mask_mode=3,
        cmp_ratio=1,
    )
    assert metadata.shape == (64,)
    assert metadata.dtype == torch.int32
    vals = metadata.cpu().tolist()
    print(f"[TND] metadata shape={metadata.shape}, dtype={metadata.dtype}")
    print(f"[TND] metadata[0:5]={vals[:5]}")


def test_bsnd_metadata_with_seqused():
    """BSND layout + seqused_q: 调用 metadata 算子。"""
    B, S1, S2, N1, D = 4, 128, 128, 8, 64
    seqused_q = torch.tensor([128, 100, 64, 32], dtype=torch.int32).npu()
    metadata = cann_ops_transformer.dense_lightning_indexer_softmax_lse_metadata(
        num_heads_q=N1,
        num_heads_k=1,
        head_dim=D,
        seqused_q=seqused_q,
        batch_size=B,
        max_seqlen_q=S1,
        max_seqlen_k=S2,
        layout_q="BSND",
        layout_k="BSND",
        mask_mode=0,
        cmp_ratio=1,
    )
    assert metadata.shape == (64,)
    assert metadata.dtype == torch.int32
    print(f"[BSND+seqused] metadata shape={metadata.shape}")


def test_e2e():
    """端到端测试: BSND layout, 验证 softmax_lse 输出正确性。

    Kernel 约束: N1=128, D=128, S2 为 128 的倍数。
    test_cases 格式: (B, S1, S2, N1, D, sparse_mode, cmp_ratio, q_seqused, k_seqused)
    """
    device = "npu:0"
    torch.npu.set_device(device)
    torch.manual_seed(42)
    test_cases = [
        (1, 4096, 1024, 64, 128, 0, 4, True, True),
        (1, 65536, 16384, 64, 128, 0, 4, True, True),
        (1, 16384, 262144, 64, 128, 0, 1, True, True),
        # (1, 65536, 16384, 128, 128, 3, 4, True, True),
        # (1, 4096, 1024, 128, 128, 3, 4, True, True),
        # (1, 16384, 262144, 128, 128, 3, 4, True, True),
    ]
    for B, S1, S2, N1, D, sparse_mode, cmp_ratio, q_seqused, k_seqused in test_cases:
        # print(f"\n--- DenseLightningIndexLse (B={B}, S1={S1}, S2={S2}, N1={N1}, D={D}, "
            #   f"SPARSE_MODE={sparse_mode}, CMP_RATIO={cmp_ratio}) ---")
        q = torch.randn((B, S1, N1, D), dtype=torch.float16)
        k = torch.randn((B, S2, 1, D), dtype=torch.float16)
        w = torch.randn((B, S1, N1), dtype=torch.float32)

        seqused_q = torch.full((B, 1), S1, dtype=torch.int32) if q_seqused else None
        seqused_k = torch.full((B, 1), S2, dtype=torch.int32) if k_seqused else None
        cmp_residual_k = torch.zeros((B, 1), dtype=torch.int32)

        seqused_q_npu = seqused_q.to(device) if seqused_q is not None else None
        seqused_k_npu = seqused_k.to(device) if seqused_k is not None else None
        metadata = cann_ops_transformer.dense_lightning_indexer_softmax_lse_metadata(
            num_heads_q=N1,
            num_heads_k=1,
            head_dim=D,
            seqused_q=seqused_q_npu,
            seqused_k=seqused_k_npu,
            cmp_residual_k=cmp_residual_k.to(device),
            batch_size=B,
            max_seqlen_q=S1,
            max_seqlen_k=S2,
            layout_q="BSND",
            layout_k="BSND",
            mask_mode=sparse_mode,
            cmp_ratio=cmp_ratio,
        )
        print(f"metadata done, shape = {metadata.shape}, dtype = {metadata.dtype}")
        lse = cann_ops_transformer.dense_lightning_indexer_softmax_lse(
            q.to(device), k.to(device), w.to(device),
            seqused_q=seqused_q_npu,
            seqused_k=seqused_k_npu,
            cmp_residual_k=cmp_residual_k.to(device),
            metadata=metadata,
            layout_q="BSND",
            layout_k="BSND",
            mask_mode=sparse_mode,
            cmp_ratio=cmp_ratio,
        )
        torch.npu.synchronize()

        # lse_ref = golden_dli_lse_blk(q.cpu(), k.cpu(), w.cpu(),
        #                              seqused_q=seqused_q.cpu() if seqused_q is not None else None,
        #                              seqused_k=seqused_k.cpu() if seqused_k is not None else None,
        #                              cmp_residual_k=cmp_residual_k.cpu(),
        #                              mask_mode=sparse_mode,
        #                              cmp_ratio=cmp_ratio)
        # diff = (lse.cpu().float() - lse_ref).abs().max().item()
        # print(f"[e2e] max|diff|={diff:.6f}")
        # torch.testing.assert_close(lse.cpu().float(), lse_ref, rtol=5e-2, atol=5e-2)
        # print("[e2e] PASS")


def golden_dli_lse_tnd(query_index, key_index, weight, cu_seqlens_q, cu_seqlens_k,
                        seqused_q=None, seqused_k=None, cmp_residual_k=None,
                        mask_mode=0, cmp_ratio=1, device="cpu"):
    q = query_index.float().to(device)
    k = key_index.float().to(device)
    w = weight.float().to(device)
    T1, N1, D = q.shape
    T2, _, _ = k.shape
    dev = device
    B = cu_seqlens_q.shape[0] - 1

    TKV = 128
    neg_inf = -torch.inf

    row_valid_all = _compute_row_valid_mask_tnd(cu_seqlens_q, cu_seqlens_k, T1,
                                                seqused_q, seqused_k, cmp_residual_k,
                                                mask_mode, cmp_ratio, dev)
    lse = torch.zeros(T1, dtype=torch.float32, device=dev)

    for b in range(B):
        q_start = cu_seqlens_q[b].item()
        q_end = cu_seqlens_q[b + 1].item()
        k_start = cu_seqlens_k[b].item()
        k_end = cu_seqlens_k[b + 1].item()
        q_len = q_end - q_start
        k_len = k_end - k_start

        q_b = q[q_start:q_end]
        k_b = k[k_start:k_end]
        w_b = w[q_start:q_end]

        q_used = q_len if seqused_q is None else seqused_q[b, 0].item()
        k_used = k_len if seqused_k is None else seqused_k[b, 0].item()

        if k_len == 0:
            lse[q_start:q_end] = neg_inf
            continue

        x_full = torch.full((q_len, k_len), neg_inf, dtype=torch.float32, device=dev)

        s2_tiles = (k_used + TKV - 1) // TKV
        for s2_i in range(s2_tiles):
            s2_start = s2_i * TKV
            s2_end = min(s2_start + TKV, k_used)
            s2_len = s2_end - s2_start

            k_slice = k_b[s2_start:s2_end, 0, :]
            if s2_len < TKV:
                k_tile = torch.cat([k_slice, torch.zeros(TKV - s2_len, D, dtype=torch.float32, device=dev)], dim=0)
            else:
                k_tile = k_slice

            score_tile = torch.matmul(q_b, k_tile.T)
            relu_score = torch.relu(score_tile)
            weighted = relu_score * w_b.unsqueeze(-1)
            reduce_sum = weighted.sum(dim=1)
            x_full[:, s2_start:s2_end] = reduce_sum[:, :s2_len]

        if mask_mode == 3:
            cur_residual = 0
            if cmp_ratio > 1:
                cur_residual = 0 if cmp_residual_k is None else cmp_residual_k[b, 0].item()
            ori_k_size = k_used * cmp_ratio + cur_residual
            for s1_i in range(q_len):
                if s1_i >= q_used:
                    continue
                valid_k_size = max((s1_i + 1 + ori_k_size - q_used) // cmp_ratio, 0)
                if valid_k_size < k_used:
                    x_full[s1_i, valid_k_size:] = neg_inf

        row_valid = row_valid_all[q_start:q_end]

        reduce_max = x_full.amax(dim=1)
        shifted = x_full - reduce_max.unsqueeze(1)
        sum_exp = torch.exp(shifted).sum(dim=1)
        lse[q_start:q_end] = torch.where(row_valid, reduce_max + torch.log(sum_exp), torch.full_like(reduce_max, neg_inf))

    return lse.unsqueeze(0)  # [N2=1, T1]


def golden_dli_lse_blk_noblock(query_index, key_index, weight, seqused_q=None, seqused_k=None,
                               cmp_residual_k=None, mask_mode=0, cmp_ratio=1, device="cpu"):
    """非分块版 golden (BSND)，一次性 batch matmul，无 b 循环、无 TKV 切块。"""
    q = query_index.float().to(device)
    k = key_index.float().to(device)
    w = weight.float().to(device)
    B, S1, N1, D = q.shape
    _, S2, _, _ = k.shape
    dev = device
    neg_inf = -torch.inf

    row_valid_all = _compute_row_valid_mask_blk(B, S1, S2, seqused_q, seqused_k,
                                                cmp_residual_k, mask_mode, cmp_ratio, dev)

    k_squeeze = k[:, :, 0, :]  # [B, S2, D]
    score = torch.matmul(q, k_squeeze.permute(0, 2, 1))  # [B, S1, N1, S2]
    relu_score = torch.relu(score)
    weighted = relu_score * w.unsqueeze(-1)
    x = weighted.sum(dim=2)  # [B, S1, S2]

    if mask_mode == 3:
        q_used_vec = S1 if seqused_q is None else seqused_q[:B, 0].to(dev)
        k_used_vec = S2 if seqused_k is None else seqused_k[:B, 0].to(dev)
        if cmp_ratio > 1:
            residual_vec = torch.zeros(B, device=dev) if cmp_residual_k is None else cmp_residual_k[:B, 0].float().to(dev)
        else:
            residual_vec = torch.zeros(B, device=dev)
        ori_k_size_vec = k_used_vec * cmp_ratio + residual_vec  # [B]
        s1_idx = torch.arange(S1, device=dev).unsqueeze(0)  # [1, S1]
        valid_k_size_mat = torch.clamp(
            ((s1_idx + 1 + ori_k_size_vec.unsqueeze(1) - q_used_vec.unsqueeze(1)) // cmp_ratio).float(),
            min=0
        )  # [B, S1]
        s2_idx = torch.arange(S2, device=dev)  # [S2]
        mask = s2_idx.unsqueeze(0).unsqueeze(0) >= valid_k_size_mat.unsqueeze(2)  # [B, S1, S2]
        x = x.masked_fill(mask, neg_inf)

    reduce_max = x.amax(dim=2)  # [B, S1]
    shifted = x - reduce_max.unsqueeze(2)
    sum_exp = torch.exp(shifted).sum(dim=2)  # [B, S1]
    lse = torch.where(row_valid_all, reduce_max + torch.log(sum_exp), torch.full_like(reduce_max, neg_inf))

    return lse.unsqueeze(1)  # [B, N2=1, S1]


def golden_dli_lse_tnd_noblock(query_index, key_index, weight, cu_seqlens_q, cu_seqlens_k,
                               seqused_q=None, seqused_k=None, cmp_residual_k=None,
                               mask_mode=0, cmp_ratio=1, device="cpu"):
    """非分块版 golden (TND)，一次性 matmul 计算 score，不按 TKV 切块。"""
    q = query_index.float().to(device)
    k = key_index.float().to(device)
    w = weight.float().to(device)
    T1, N1, D = q.shape
    T2, _, _ = k.shape
    dev = device
    B = cu_seqlens_q.shape[0] - 1
    neg_inf = -torch.inf

    row_valid_all = _compute_row_valid_mask_tnd(cu_seqlens_q, cu_seqlens_k, T1,
                                                seqused_q, seqused_k, cmp_residual_k,
                                                mask_mode, cmp_ratio, dev)
    lse = torch.zeros(T1, dtype=torch.float32, device=dev)

    for b in range(B):
        q_start = cu_seqlens_q[b].item()
        q_end = cu_seqlens_q[b + 1].item()
        k_start = cu_seqlens_k[b].item()
        k_end = cu_seqlens_k[b + 1].item()
        q_len = q_end - q_start
        k_len = k_end - k_start

        q_b = q[q_start:q_end]
        k_b = k[k_start:k_end]
        w_b = w[q_start:q_end]

        q_used = q_len if seqused_q is None else seqused_q[b, 0].item()
        k_used = k_len if seqused_k is None else seqused_k[b, 0].item()

        if k_len == 0:
            lse[q_start:q_end] = neg_inf
            continue

        score = torch.matmul(q_b, k_b[:k_used, 0, :].T)  # [q_len, N1, k_used]
        relu_score = torch.relu(score)
        weighted = relu_score * w_b.unsqueeze(-1)
        x = weighted.sum(dim=1)  # [q_len, k_used]

        if mask_mode == 3:
            cur_residual = 0
            if cmp_ratio > 1:
                cur_residual = 0 if cmp_residual_k is None else cmp_residual_k[b, 0].item()
            ori_k_size = k_used * cmp_ratio + cur_residual
            for s1_i in range(min(q_used, q_len)):
                valid_k_size = max((s1_i + 1 + ori_k_size - q_used) // cmp_ratio, 0)
                if valid_k_size < k_used:
                    x[s1_i, valid_k_size:] = neg_inf

        row_valid = row_valid_all[q_start:q_end]

        reduce_max = x.amax(dim=1)
        shifted = x - reduce_max.unsqueeze(1)
        sum_exp = torch.exp(shifted).sum(dim=1)
        lse[q_start:q_end] = torch.where(row_valid, reduce_max + torch.log(sum_exp), torch.full_like(reduce_max, neg_inf))

    return lse.unsqueeze(0)  # [N2=1, T1]


def cpu_kernel_dli_lse_blk(query_index, key_index, weight, seqused_q=None, seqused_k=None,
                           cmp_residual_k=None, mask_mode=0, cmp_ratio=1, device="cpu"):
    """CPU 实现的 kernel 等价流程，与 NPU kernel 的分块/精度/归约逻辑一致。

    关键对齐点：
    - S2 按 TKV=128 切块，每块 matmul 使用 float32 高精度计算
    - DualModeSplitN: acc[N1, TKV] 按 N 拆为两半 sub_id=0/1，各做 reduce_max 后取 max
    - 两阶段归约: reduce_max(relu(qk)*w, 沿N1求和) → reduce_sum(exp_sub, 沿S2) → lse
    - mask_mode=3 的 valid_k_size 计算与 kernel 一致
    """
    q = query_index.float().to(device)
    k = key_index.float().to(device)
    w = weight.float().to(device)
    B, S1, N1, D = q.shape
    _, S2, _, _ = k.shape
    dev = device

    TKV = 128
    neg_inf = -torch.inf

    lse = torch.zeros(B, S1, dtype=torch.float32, device=dev)

    for b in range(B):
        q_b = q[b]
        k_b = k[b]
        w_b = w[b]

        q_used = S1 if seqused_q is None else seqused_q[b, 0].item()
        k_used = S2 if seqused_k is None else seqused_k[b, 0].item()

        for s1_i in range(S1):
            if s1_i >= q_used:
                continue

            valid_k_size = k_used
            valid_s1_start = 0
            if mask_mode == 3:
                cur_residual = 0
                if cmp_ratio > 1:
                    cur_residual = 0 if cmp_residual_k is None else cmp_residual_k[b, 0].item()
                ori_k_size = k_used * cmp_ratio + cur_residual
                qk_residual = q_used - ori_k_size
                valid_s1_start = max(qk_residual, 0)
                valid_k_size = max((s1_i + 1 - qk_residual) // cmp_ratio, 0)

            if s1_i < valid_s1_start or valid_k_size <= 0:
                continue

            s2_tiles = (valid_k_size + TKV - 1) // TKV

            reduce_sum_vec = torch.zeros(s2_tiles, TKV, dtype=torch.float32, device=dev)
            vec_tail_size = torch.zeros(1, dtype=torch.int32, device=dev)
            running_max = neg_inf

            for s2_i in range(s2_tiles):
                s2_start = s2_i * TKV
                s2_end = min(s2_start + TKV, valid_k_size)
                s2_len = s2_end - s2_start

                k_slice = k_b[s2_start:s2_end, 0, :]
                if s2_len < TKV:
                    k_tile = torch.zeros(TKV, D, dtype=torch.float32, device=dev)
                    k_tile[:s2_len] = k_slice
                else:
                    k_tile = k_slice

                score = torch.matmul(q_b[s1_i], k_tile.T)
                relu_score = torch.relu(score)
                weighted = relu_score * w_b[s1_i].unsqueeze(-1)
                reduce_sum = weighted.sum(dim=0)

                reduce_sum_vec[s2_i, :s2_len] = reduce_sum[:s2_len]

                tile_max = reduce_sum[:s2_len].max().item() if s2_len > 0 else neg_inf
                if tile_max > running_max:
                    running_max = tile_max

                if s2_i == s2_tiles - 1:
                    vec_tail_size[0] = s2_len

            if running_max == neg_inf:
                lse[b, s1_i] = neg_inf
            else:
                all_sum = 0.0
                for s2_i in range(s2_tiles - 1):
                    exp_val = torch.exp(reduce_sum_vec[s2_i] - running_max)
                    all_sum += exp_val.sum().item()
                tail_size = vec_tail_size[0].item()
                if tail_size > 0:
                    exp_tail = torch.exp(reduce_sum_vec[s2_tiles - 1, :tail_size] - running_max)
                    all_sum += exp_tail.sum().item()
                lse[b, s1_i] = running_max + math.log(all_sum) if all_sum > 0 else neg_inf

    return lse.unsqueeze(1)  # [B, N2=1, S1]


def cpu_kernel_dli_lse_tnd(query_index, key_index, weight, cu_seqlens_q, cu_seqlens_k,
                           seqused_q=None, seqused_k=None, cmp_residual_k=None,
                           mask_mode=0, cmp_ratio=1, device="cpu"):
    """CPU 实现的 kernel 等价流程 (TND layout)。

    与 cpu_kernel_dli_lse_blk 逻辑一致，区别在于按 cu_seqlens 切分 batch。
    matmul 使用 float32 高精度计算。
    """
    q = query_index.float().to(device)
    k = key_index.float().to(device)
    w = weight.float().to(device)
    T1, N1, D = q.shape
    T2, _, _ = k.shape
    dev = device
    B = cu_seqlens_q.shape[0] - 1

    TKV = 128
    neg_inf = -torch.inf

    lse = torch.zeros(T1, dtype=torch.float32, device=dev)

    for b in range(B):
        q_start = cu_seqlens_q[b].item()
        q_end = cu_seqlens_q[b + 1].item()
        k_start = cu_seqlens_k[b].item()
        k_end = cu_seqlens_k[b + 1].item()
        q_len = q_end - q_start
        k_len = k_end - k_start

        q_b = q[q_start:q_end]
        k_b = k[k_start:k_end]
        w_b = w[q_start:q_end]

        q_used = q_len if seqused_q is None else seqused_q[b, 0].item()
        k_used = k_len if seqused_k is None else seqused_k[b, 0].item()

        if k_len == 0:
            lse[q_start:q_end] = neg_inf
            continue

        for s1_i in range(q_len):
            if s1_i >= q_used:
                continue

            valid_k_size = k_used
            valid_s1_start = 0
            if mask_mode == 3:
                cur_residual = 0
                if cmp_ratio > 1:
                    cur_residual = 0 if cmp_residual_k is None else cmp_residual_k[b, 0].item()
                ori_k_size = k_used * cmp_ratio + cur_residual
                qk_residual = q_used - ori_k_size
                valid_s1_start = max(qk_residual, 0)
                valid_k_size = max((s1_i + 1 - qk_residual) // cmp_ratio, 0)

            if s1_i < valid_s1_start or valid_k_size <= 0:
                continue

            s2_tiles = (valid_k_size + TKV - 1) // TKV

            reduce_sum_vec = torch.zeros(s2_tiles, TKV, dtype=torch.float32, device=dev)
            vec_tail_size = torch.zeros(1, dtype=torch.int32, device=dev)
            running_max = neg_inf

            for s2_i in range(s2_tiles):
                s2_start = s2_i * TKV
                s2_end = min(s2_start + TKV, valid_k_size)
                s2_len = s2_end - s2_start

                k_slice = k_b[s2_start:s2_end, 0, :]
                if s2_len < TKV:
                    k_tile = torch.zeros(TKV, D, dtype=torch.float32, device=dev)
                    k_tile[:s2_len] = k_slice
                else:
                    k_tile = k_slice

                score = torch.matmul(q_b[s1_i, :, :], k_tile.T)
                relu_score = torch.relu(score)
                weighted = relu_score * w_b[s1_i].unsqueeze(-1)
                reduce_sum = weighted.sum(dim=0)

                reduce_sum_vec[s2_i, :s2_len] = reduce_sum[:s2_len]

                tile_max = reduce_sum[:s2_len].max().item() if s2_len > 0 else neg_inf
                if tile_max > running_max:
                    running_max = tile_max

                if s2_i == s2_tiles - 1:
                    vec_tail_size[0] = s2_len

            if running_max == neg_inf:
                lse[q_start + s1_i] = neg_inf
            else:
                all_sum = 0.0
                for s2_i in range(s2_tiles - 1):
                    exp_val = torch.exp(reduce_sum_vec[s2_i] - running_max)
                    all_sum += exp_val.sum().item()
                tail_size = vec_tail_size[0].item()
                if tail_size > 0:
                    exp_tail = torch.exp(reduce_sum_vec[s2_tiles - 1, :tail_size] - running_max)
                    all_sum += exp_tail.sum().item()
                lse[q_start + s1_i] = running_max + math.log(all_sum) if all_sum > 0 else neg_inf

    return lse.unsqueeze(0)  # [N2=1, T1]


def test_invalid_num_heads_q():
    """num_heads_q 不能被 num_heads_k 整除应抛出异常。"""
    try:
        cann_ops_transformer.dense_lightning_indexer_softmax_lse_metadata(
            num_heads_q=7,
            num_heads_k=4,
            head_dim=64,
            batch_size=2,
            max_seqlen_q=128,
            max_seqlen_k=256,
        )
        assert False, "Should have raised for num_heads_q not divisible by num_heads_k"
    except RuntimeError:
        pass



def cpu_kernel_dli_lse_blk_noblock(query_index, key_index, weight, seqused_q=None, seqused_k=None,
                                   cmp_residual_k=None, mask_mode=0, cmp_ratio=1, device="cpu"):
    """非分块版 cpu_kernel (BSND)，matmul 使用 float32 高精度。

    与 _golden_dense_lightning_index_lse 的非分块实现类似，但保留 kernel 的 mask 逻辑。
    """
    q = query_index.float().to(device)
    k = key_index.float().to(device)
    w = weight.float().to(device)
    B, S1, N1, D = q.shape
    _, S2, _, _ = k.shape
    dev = device
    neg_inf = -torch.inf

    lse = torch.zeros(B, S1, dtype=torch.float32, device=dev)

    for b in range(B):
        q_b = q[b]
        k_b = k[b]
        w_b = w[b]

        q_used = S1 if seqused_q is None else seqused_q[b, 0].item()
        k_used = S2 if seqused_k is None else seqused_k[b, 0].item()

        score = torch.matmul(q_b, k_b[:, 0, :].T)  # [S1, N1, S2]
        relu_score = torch.relu(score)
        weighted = relu_score * w_b.unsqueeze(-1)
        x = weighted.sum(dim=1)  # [S1, S2]

        if mask_mode == 3:
            cur_residual = 0
            if cmp_ratio > 1:
                cur_residual = 0 if cmp_residual_k is None else cmp_residual_k[b, 0].item()
            ori_k_size = k_used * cmp_ratio + cur_residual
            for s1_i in range(min(q_used, S1)):
                valid_k_size = max((s1_i + 1 + ori_k_size - q_used) // cmp_ratio, 0)
                if valid_k_size < k_used:
                    x[s1_i, valid_k_size:] = neg_inf

        for s1_i in range(S1):
            if s1_i >= q_used:
                lse[b, s1_i] = neg_inf
                continue
            row = x[s1_i, :k_used]
            if mask_mode == 3:
                cur_residual = 0
                if cmp_ratio > 1:
                    cur_residual = 0 if cmp_residual_k is None else cmp_residual_k[b, 0].item()
                ori_k_size = k_used * cmp_ratio + cur_residual
                valid_k_size = max((s1_i + 1 + ori_k_size - q_used) // cmp_ratio, 0)
                if valid_k_size <= 0:
                    lse[b, s1_i] = neg_inf
                    continue
                row = x[s1_i, :valid_k_size]
            row_max = row.max()
            if row_max == neg_inf:
                lse[b, s1_i] = neg_inf
            else:
                lse[b, s1_i] = row_max + torch.log(torch.exp(row - row_max).sum())

    return lse.unsqueeze(1)  # [B, N2=1, S1]


def cpu_kernel_dli_lse_tnd_noblock(query_index, key_index, weight, cu_seqlens_q, cu_seqlens_k,
                                   seqused_q=None, seqused_k=None, cmp_residual_k=None,
                                   mask_mode=0, cmp_ratio=1, device="cpu"):
    """非分块版 cpu_kernel (TND)，matmul 使用 float32 高精度。

    与 cpu_kernel_dli_lse_blk_noblock 逻辑一致，区别在于按 cu_seqlens 切分 batch。
    """
    q = query_index.float().to(device)
    k = key_index.float().to(device)
    w = weight.float().to(device)
    T1, N1, D = q.shape
    T2, _, _ = k.shape
    dev = device
    B = cu_seqlens_q.shape[0] - 1
    neg_inf = -torch.inf

    lse = torch.zeros(T1, dtype=torch.float32, device=dev)

    for b in range(B):
        q_start = cu_seqlens_q[b].item()
        q_end = cu_seqlens_q[b + 1].item()
        k_start = cu_seqlens_k[b].item()
        k_end = cu_seqlens_k[b + 1].item()
        q_len = q_end - q_start
        k_len = k_end - k_start

        q_b = q[q_start:q_end]
        k_b = k[k_start:k_end]
        w_b = w[q_start:q_end]

        q_used = q_len if seqused_q is None else seqused_q[b, 0].item()
        k_used = k_len if seqused_k is None else seqused_k[b, 0].item()

        if k_len == 0:
            lse[q_start:q_end] = neg_inf
            continue

        score = torch.matmul(q_b, k_b[:, 0, :].T)  # [q_len, N1, k_len]
        relu_score = torch.relu(score)
        weighted = relu_score * w_b.unsqueeze(-1)
        x = weighted.sum(dim=1)  # [q_len, k_len]

        if mask_mode == 3:
            cur_residual = 0
            if cmp_ratio > 1:
                cur_residual = 0 if cmp_residual_k is None else cmp_residual_k[b, 0].item()
            ori_k_size = k_used * cmp_ratio + cur_residual
            for s1_i in range(min(q_used, q_len)):
                valid_k_size = max((s1_i + 1 + ori_k_size - q_used) // cmp_ratio, 0)
                if valid_k_size < k_used:
                    x[s1_i, valid_k_size:] = neg_inf

        for s1_i in range(q_len):
            if s1_i >= q_used:
                lse[q_start + s1_i] = neg_inf
                continue
            if mask_mode == 3:
                cur_residual = 0
                if cmp_ratio > 1:
                    cur_residual = 0 if cmp_residual_k is None else cmp_residual_k[b, 0].item()
                ori_k_size = k_used * cmp_ratio + cur_residual
                valid_k_size = max((s1_i + 1 + ori_k_size - q_used) // cmp_ratio, 0)
                if valid_k_size <= 0:
                    lse[q_start + s1_i] = neg_inf
                    continue
                row = x[s1_i, :valid_k_size]
            else:
                row = x[s1_i, :k_used]
            row_max = row.max()
            if row_max == neg_inf:
                lse[q_start + s1_i] = neg_inf
            else:
                lse[q_start + s1_i] = row_max + torch.log(torch.exp(row - row_max).sum())

    return lse.unsqueeze(0)  # [N2=1, T1]


def test_invalid_cmp_ratio():
    """cmp_ratio 超范围应抛出异常。"""
    try:
        cann_ops_transformer.dense_lightning_indexer_softmax_lse_metadata(
            num_heads_q=8,
            num_heads_k=1,
            head_dim=64,
            batch_size=2,
            max_seqlen_q=128,
            max_seqlen_k=256,
            cmp_ratio=0,
        )
        assert False, "Should have raised for invalid cmp_ratio"
    except RuntimeError:
        pass


def _parse_cell(val):
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        s = val.strip()
        if s.lower() == "null" or s == "":
            return None
        if s.startswith("[") and s.endswith("]"):
            return ast.literal_eval(s)
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return s
    return val


DLI_METADATA_SIZE = 64
DEFAULT_AIC_CORE_NUM = 36


def _compute_metadata_python(
    num_heads_q,
    num_heads_k,
    head_dim,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    layout_q=None,
    layout_k=None,
    mask_mode=None,
    cmp_ratio=None,
    aic_core_num=DEFAULT_AIC_CORE_NUM,
    device="npu:0",
):
    layout = "BSND" if layout_q is None else layout_q
    is_tnd = (layout == "TND")

    if is_tnd:
        if cu_seqlens_q is None:
            raise ValueError("TND layout requires cu_seqlens_q")
        cu_q = cu_seqlens_q.cpu() if isinstance(cu_seqlens_q, torch.Tensor) else cu_seqlens_q
        total_num = int(cu_q[-1].item())
    else:
        b_size = batch_size if batch_size is not None else 0
        s1 = max_seqlen_q if max_seqlen_q is not None else 0
        total_num = b_size * s1

    n = aic_core_num

    if total_num <= 0:
        raise ValueError(f"total_num should be > 0, got {total_num}")

    if total_num < n:
        forecore_num = total_num
        tail_core_num = 0
        b_s1_per_core = 1
        b_s1_per_tail_core = 0
    else:
        base = total_num // n
        rem = total_num % n
        if rem == 0:
            forecore_num = n
            tail_core_num = 0
            b_s1_per_core = base
            b_s1_per_tail_core = 0
        else:
            forecore_num = rem
            b_s1_per_core = base + 1
            tail_core_num = n - rem
            b_s1_per_tail_core = base

    metadata_list = [0] * DLI_METADATA_SIZE
    metadata_list[0] = forecore_num
    metadata_list[1] = tail_core_num
    metadata_list[2] = b_s1_per_core
    metadata_list[3] = b_s1_per_tail_core

    return torch.tensor(metadata_list, dtype=torch.int32, device="cpu")


print_log = print


_ERR_FLOOR = 2 ** -14


def check_with_gpu_compare_standard(cpu_tensor, npu_tensor, gpu_tensor, level=2):
    result = {}
    small_value_ratio = 2
    if level == 0:
        mare_ratio = 10.0
        mere_ratio = 2.0
        rmse_ratio = 2.0
    elif level == 1:
        mare_ratio = 5.0
        mere_ratio = 1.5
        rmse_ratio = 1.5
    elif level == 2:
        mare_ratio = 2.0
        mere_ratio = 1.2
        rmse_ratio = 1.2
    else:
        print_log(f"[check_with_gpu_compare_standard]unknow level:{level}")
        return result
    cpu_tensor = cpu_tensor.flatten()
    npu_tensor = npu_tensor.flatten()
    gpu_tensor = gpu_tensor.flatten()
    actual_dtype = npu_tensor.dtype
    print_log(f"[check_with_gpu_compare_standard]actual_dtype:{actual_dtype}")
    if actual_dtype == torch.bfloat16:
        small_value_thresh =  2 ** -8
        small_value_atol = 2 ** -16
    elif actual_dtype == torch.float16:
        small_value_thresh = 2 ** -11
        small_value_atol = 2 ** -16
    else:
        small_value_thresh = 2 ** -14
        small_value_atol = 2 ** -30
    #eps = torch.finfo(npu_tensor.dtype).eps
    eps = 1e-7
    abs_cpu_tensor = torch.abs(cpu_tensor)
    abs_cpu_with_eps = abs_cpu_tensor + eps
    cpu_down_dtype = cpu_tensor.to(npu_tensor.dtype)
    cpu_finite_mask = torch.isfinite(cpu_down_dtype)
    npu_finite_mask = torch.isfinite(npu_tensor)
    gpu_finite_mask = torch.isfinite(gpu_tensor)
    all_finite_mask = torch.logical_and(cpu_finite_mask,torch.logical_and(npu_finite_mask, gpu_finite_mask))

    #inf nan 的比较
    inf_nan_pass = True
    #-inf
    cpu_neg_inf_mask = torch.isneginf(cpu_down_dtype)#
    print_log(f"[check_with_gpu_compare_standard]cpu_ninf_count:{torch.sum(cpu_neg_inf_mask)}")
    npu_neg_inf_mask = torch.isneginf(npu_tensor)
    gpu_neg_inf_mask = torch.isneginf(gpu_tensor)
    gpu_cpu_neg_inf_mask = torch.logical_and(cpu_neg_inf_mask,gpu_neg_inf_mask)#gpu和cpu相等的
    npu_not_cpu_ninf_mask = npu_neg_inf_mask.not_equal(cpu_neg_inf_mask) #npu和cpu不相等的
    npu_error_ninf_mask = torch.logical_and(npu_not_cpu_ninf_mask,~gpu_cpu_neg_inf_mask)
    npu_neg_inf_indices = npu_error_ninf_mask.nonzero(as_tuple=False).squeeze()
    npu_error_ninf_count = torch.sum(npu_error_ninf_mask)
    print_log(f"[check_with_gpu_compare_standard]npu_error_ninf_count:{npu_error_ninf_count} indices:{npu_neg_inf_indices}")
    if npu_error_ninf_count > 0:
        inf_nan_pass = False


    #+inf
    cpu_pos_inf_mask = torch.isinf(cpu_down_dtype)#
    print_log(f"[check_with_gpu_compare_standard]cpu_pinf_count:{torch.sum(cpu_pos_inf_mask)}")
    npu_pos_inf_mask = torch.isinf(npu_tensor)
    gpu_pos_inf_mask = torch.isinf(gpu_tensor)
    gpu_cpu_pos_inf_mask = torch.logical_and(cpu_pos_inf_mask, gpu_pos_inf_mask)#gpu和cpu相等的
    npu_not_cpu_inf_mask = npu_pos_inf_mask.not_equal(cpu_pos_inf_mask)  # npu和cpu不相等的
    npu_error_inf_mask = torch.logical_and(npu_not_cpu_inf_mask, ~gpu_cpu_pos_inf_mask)
    npu_inf_indices = npu_error_inf_mask.nonzero(as_tuple=False).squeeze()
    npu_error_inf_count = torch.sum(npu_error_inf_mask)
    print_log(f"[check_with_gpu_compare_standard]npu_error_inf_count:{npu_error_inf_count} indices:{npu_inf_indices}")
    if npu_error_inf_count > 0:
        inf_nan_pass = False
    #nan
    cpu_nan_mask = torch.isnan(cpu_down_dtype)#
    print_log(f"[check_with_gpu_compare_standard]cpu_nan_count:{torch.sum(cpu_nan_mask)}")
    npu_nan_mask = torch.isnan(npu_tensor)
    gpu_nan_mask = torch.isnan(gpu_tensor)
    gpu_cpu_nan_mask = torch.logical_and(cpu_nan_mask, gpu_nan_mask)#gpu和cpu相等的
    npu_not_cpu_nan_mask = npu_nan_mask.not_equal(cpu_nan_mask)  # npu和cpu不相等的
    npu_error_nan_mask = torch.logical_and(npu_not_cpu_nan_mask, ~gpu_cpu_nan_mask)
    npu_nan_indices = npu_error_nan_mask.nonzero(as_tuple=False).squeeze()
    npu_error_nan_count = torch.sum(npu_error_nan_mask)
    print_log(f"[check_with_gpu_compare_standard]npu_error_nan_count:{npu_error_nan_count} indices:{npu_nan_indices}")
    if npu_error_nan_count > 0:
        inf_nan_pass = False

    if inf_nan_pass:
        result["infnan"] = "pass"
    else:
        result["infnan"] = "error"

    #small values (per 新精度标准.md formula: |golden| < threshold)
    cpu_small_value_mask = torch.logical_and(all_finite_mask, torch.less_equal(abs_cpu_tensor, small_value_thresh))
    cpu_small_values_count = torch.sum(cpu_small_value_mask)
    print_log(f"[check_with_gpu_compare_standard]cpu_small_values_count:{cpu_small_values_count}")
    cpu_masked_small_values = cpu_tensor.masked_fill(~cpu_small_value_mask, 0.0)

    npu_masked_small_values = npu_tensor.masked_fill(~cpu_small_value_mask, 0.0)
    abs_npu_small_values_ae = torch.abs(cpu_masked_small_values - npu_masked_small_values)
    npu_small_values_error_mask = torch.greater(abs_npu_small_values_ae, small_value_atol)
    npu_small_values_error_count = torch.sum(npu_small_values_error_mask)
    #npu_small_values_error_ratio = npu_small_values_error_count/cpu_small_values_count
    print_log(f"[check_with_gpu_compare_standard]npu_small_values_error_count:{npu_small_values_error_count}")

    gpu_masked_small_values = gpu_tensor.masked_fill(~cpu_small_value_mask, 0.0)
    abs_gpu_small_values_ae = torch.abs(cpu_masked_small_values - gpu_masked_small_values)
    gpu_small_values_error_mask = torch.greater(abs_gpu_small_values_ae, small_value_atol)
    gpu_small_values_error_count = torch.sum(gpu_small_values_error_mask)
    #gpu_small_values_error_ratio = gpu_small_values_error_count / cpu_small_values_count
    print_log(f"[check_with_gpu_compare_standard]gpu_small_values_error_count:{gpu_small_values_error_count}")

    if gpu_small_values_error_count == 0:
        gpu_small_values_error_count = 1
    if npu_small_values_error_count > small_value_ratio*gpu_small_values_error_count:
        result["smallvalue"] = "error"
    elif npu_small_values_error_count < gpu_small_values_error_count:
        result["smallvalue"] = "pass"
    else:
        result["smallvalue"] = "warning"


    #normal values (per 新精度标准.md formula: |golden| > threshold)
    cpu_normal_value_mask = torch.logical_and(all_finite_mask, torch.greater(abs_cpu_tensor, small_value_thresh))
    cpu_normal_values_count = torch.sum(cpu_normal_value_mask)
    print_log(f"[check_with_gpu_compare_standard]cpu_normal_values_count:{cpu_normal_values_count}")
    cpu_masked_normal_values = cpu_tensor.masked_fill(~cpu_normal_value_mask, 0.0)

    npu_masked_normal_values = npu_tensor.masked_fill(~cpu_normal_value_mask, 0.0)
    abs_npu_normal_values_ae = torch.abs(cpu_masked_normal_values - npu_masked_normal_values)
    npu_re = abs_npu_normal_values_ae/abs_cpu_with_eps
    max_npu_re = torch.max(npu_re)

    avg_npu_re = torch.sum(npu_re)/cpu_normal_values_count
    rmse_npu = torch.sqrt(torch.sum(torch.square(cpu_masked_normal_values - npu_masked_normal_values))/cpu_normal_values_count)


    gpu_masked_normal_values = gpu_tensor.masked_fill(~cpu_normal_value_mask, 0.0)
    abs_gpu_normal_values_ae = torch.abs(cpu_masked_normal_values - gpu_masked_normal_values)
    gpu_re = abs_gpu_normal_values_ae/abs_cpu_with_eps
    max_gpu_re = torch.max(gpu_re)

    avg_gpu_re = torch.sum(gpu_re)/cpu_normal_values_count
    rmse_gpu = torch.sqrt(torch.sum(torch.square(cpu_masked_normal_values - gpu_masked_normal_values))/cpu_normal_values_count)

    mare_npu_val = max_npu_re.item() if torch.is_tensor(max_npu_re) else float(max_npu_re)
    mare_gpu_val = max_gpu_re.item() if torch.is_tensor(max_gpu_re) else float(max_gpu_re)
    mere_npu_val = avg_npu_re.item() if torch.is_tensor(avg_npu_re) else float(avg_npu_re)
    mere_gpu_val = avg_gpu_re.item() if torch.is_tensor(avg_gpu_re) else float(avg_gpu_re)
    rmse_npu_val = rmse_npu.item() if torch.is_tensor(rmse_npu) else float(rmse_npu)
    rmse_gpu_val = rmse_gpu.item() if torch.is_tensor(rmse_gpu) else float(rmse_gpu)
    mare_ratio_val = mare_npu_val / max(mare_gpu_val, _ERR_FLOOR)
    mere_ratio_val = mere_npu_val / max(mere_gpu_val, _ERR_FLOOR)
    rmse_ratio_val = rmse_npu_val / max(rmse_gpu_val, _ERR_FLOOR)
    result["mare_npu"] = mare_npu_val
    result["mare_gpu"] = mare_gpu_val
    result["mare_ratio"] = mare_ratio_val
    result["mere_npu"] = mere_npu_val
    result["mere_gpu"] = mere_gpu_val
    result["mere_ratio"] = mere_ratio_val
    result["rmse_npu"] = rmse_npu_val
    result["rmse_gpu"] = rmse_gpu_val
    result["rmse_ratio"] = rmse_ratio_val

    if max_npu_re > mare_ratio * max(max_gpu_re, _ERR_FLOOR):
        result["max_re"] = "error"
    elif max_npu_re < max_gpu_re:
        result["max_re"] = "pass"
    else:
        result["max_re"] = "warning"

    if avg_npu_re > mere_ratio * max(avg_gpu_re, _ERR_FLOOR):
        result["avg_re"] = "error"
    elif avg_npu_re < avg_gpu_re:
        result["avg_re"] = "pass"
    else:
        result["avg_re"] = "warning"

    if rmse_npu > rmse_ratio * max(rmse_gpu, _ERR_FLOOR):
        result["rmse"] = "error"
    elif rmse_npu < rmse_gpu:
        result["rmse"] = "pass"
    else:
        result["rmse"] = "warning"

    #eb
    npu_larger_count  = torch.sum(torch.greater(npu_tensor - cpu_tensor, 0.0))
    npu_smaller_count  = torch.sum(torch.less(npu_tensor - cpu_tensor, 0.0))
    npu_eb = abs(npu_larger_count - npu_smaller_count) / cpu_tensor.numel()

    gpu_larger_count  = torch.sum(torch.greater(gpu_tensor - cpu_tensor, 0.0))
    gpu_smaller_count  = torch.sum(torch.less(gpu_tensor - cpu_tensor, 0.0))
    gpu_eb = abs(gpu_larger_count - gpu_smaller_count) / cpu_tensor.numel()

    if npu_eb > 2 * gpu_eb:
        result["eb"] = "error"
    elif npu_eb < gpu_eb:
        result["eb"] = "pass"
    else:
        result["eb"] = "warning"
    return result


def _load_xlsx_cases(xlsx_path):
    if openpyxl is None:
        raise ImportError("openpyxl is required to read xlsx test cases. Install via: pip install openpyxl")
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb.active
    headers = [cell.value for cell in ws[1]]
    cases = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None:
            continue
        case = {}
        for i, h in enumerate(headers):
            if i < len(row):
                case[h] = _parse_cell(row[i])
        enable_val = case.get("enable")
        if enable_val is False or (isinstance(enable_val, str) and enable_val.lower() == "false"):
            continue
        cases.append(case)
    return cases


def _dump_npu_inputs(tag, tensors):
    print(f"  [{tag} npu inputs] " + ", ".join(
        f"{name}={t.shape if t is not None else None}/{t.dtype if t is not None else None}"
        for name, t in tensors.items()))


def _build_case_inputs(case, device, raw_optional_shape=False):
    B = case["B"]
    N1 = case["N1"]
    N2 = case.get("N2", 1)
    if N2 is None:
        N2 = 1
    D = case["D"]
    S1 = case["S1"]
    S2 = case["S2"]
    dtype_str = case.get("dtype", "float16")
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float32": torch.float32}
    if "_" in dtype_str:
        q_dtype_str, k_dtype_str = dtype_str.split("_")
        torch_dtype_q = dtype_map.get(q_dtype_str, torch.float16)
        torch_dtype_k = dtype_map.get(k_dtype_str, torch.float16)
    else:
        torch_dtype_q = dtype_map.get(dtype_str, torch.float16)
        torch_dtype_k = torch_dtype_q
    seed = case.get("manual_seed", 0)
    if seed is None:
        seed = 0
    layout_q = case.get("layout_q", "BSND")
    if layout_q is None:
        layout_q = "BSND"
    layout_k = case.get("layout_k", "BSND")
    if layout_k is None:
        layout_k = "BSND"
    is_tnd = (layout_q == "TND")

    torch.manual_seed(seed)

    seqlens_q = case.get("seqlens_list_q")
    seqlens_k = case.get("seqlens_list_k")

    if is_tnd and seqlens_q is not None:
        seqlens_q_list = seqlens_q if isinstance(seqlens_q, list) else list(seqlens_q)
        seqlens_k_list = seqlens_k if isinstance(seqlens_k, list) else list(seqlens_k)
        T1 = sum(seqlens_q_list)
        T2 = sum(seqlens_k_list)
        cu_seqlens_q = torch.tensor([0] + list(__import__("itertools").accumulate(seqlens_q_list)),
                                     dtype=torch.int32)
        cu_seqlens_k = torch.tensor([0] + list(__import__("itertools").accumulate(seqlens_k_list)),
                                     dtype=torch.int32)
    else:
        T1 = B * S1
        T2 = B * S2
        cu_seqlens_q = None
        cu_seqlens_k = None

    D_k = case.get("D_k")
    if D_k is None:
        D_k = D
    q_ndim = case.get("q_ndim")
    if q_ndim is None:
        q_ndim = 3 if is_tnd else 4
    k_ndim = case.get("k_ndim")
    if k_ndim is None:
        k_ndim = 3 if is_tnd else 4

    if q_ndim == 3:
        q = torch.randn((T1, N1, D), dtype=torch_dtype_q)
    else:
        q = torch.randn((B, S1, N1, D), dtype=torch_dtype_q)
    if k_ndim == 3:
        k = torch.randn((T2, N2, D_k), dtype=torch_dtype_k)
    else:
        k = torch.randn((B, S2, N2, D_k), dtype=torch_dtype_k)
    if is_tnd:
        w = torch.randn((T1, N1), dtype=torch.float32)
    else:
        w = torch.randn((B, S1, N1), dtype=torch.float32)

    seqused_q_val = case.get("seqused_q")
    seqused_k_val = case.get("seqused_k")
    cmp_residual_k_val = case.get("cmp_residual_k")

    seqused_q = None
    if seqused_q_val is not None:
        seqused_q = torch.tensor(seqused_q_val, dtype=torch.int32)
        if not raw_optional_shape:
            seqused_q = seqused_q.reshape(B, 1)

    seqused_k = None
    if seqused_k_val is not None:
        seqused_k = torch.tensor(seqused_k_val, dtype=torch.int32)
        if not raw_optional_shape:
            seqused_k = seqused_k.reshape(B, 1)

    cmp_residual_k = None
    if cmp_residual_k_val is not None:
        cmp_residual_k = torch.tensor(cmp_residual_k_val, dtype=torch.int32)
        if not raw_optional_shape:
            cmp_residual_k = cmp_residual_k.reshape(B, 1)

    mask_mode = case.get("mask_mode", 0)
    if mask_mode is None:
        mask_mode = 0
    cmp_ratio = case.get("cmp_ratio", 1)
    if cmp_ratio is None:
        cmp_ratio = 1

    return {
        "q": q, "k": k, "w": w,
        "cu_seqlens_q": cu_seqlens_q, "cu_seqlens_k": cu_seqlens_k,
        "seqused_q": seqused_q, "seqused_k": seqused_k,
        "cmp_residual_k": cmp_residual_k,
        "layout_q": layout_q, "layout_k": layout_k,
        "mask_mode": mask_mode,
        "cmp_ratio": cmp_ratio,
        "is_tnd": is_tnd,
        "B": B,
        "N2": N2,
    }


def _err_check_case(case, device):
    """Faithfully construct illegal inputs (preserving wrong shapes/lengths, zero sizes)
    and call only the NPU ops (metadata + main). Returns a dict describing which stage
    rejected the input, or NOT_REJECTED if the op accepted it."""
    inputs = _build_case_inputs(case, device, raw_optional_shape=True)
    B = case["B"]
    N1 = case["N1"]
    S1 = case["S1"]
    S2 = case["S2"]
    D = case["D"]
    mask_mode = case.get("mask_mode", 0)
    if mask_mode is None:
        mask_mode = 0
    cmp_ratio = case.get("cmp_ratio", 1)
    if cmp_ratio is None:
        cmp_ratio = 1
    is_tnd = inputs["is_tnd"]

    result = {"meta_stage": "SKIPPED", "meta_msg": "", "main_stage": "SKIPPED", "main_msg": ""}

    def _to(t):
        return t.to(device) if t is not None else None

    try:
        _dump_npu_inputs("metadata", {
            "cu_seqlens_q": inputs["cu_seqlens_q"],
            "cu_seqlens_k": inputs["cu_seqlens_k"],
            "seqused_q": inputs["seqused_q"],
            "seqused_k": inputs["seqused_k"],
            "cmp_residual_k": inputs["cmp_residual_k"],
        })
        print(f"  [metadata npu inputs] num_heads_q={N1}, num_heads_k={inputs['N2']}, "
              f"head_dim={D}, batch_size={B if not is_tnd else None}, "
              f"max_seqlen_q={S1 if not is_tnd else None}, "
              f"max_seqlen_k={S2 if not is_tnd else None}, "
              f"layout_q={inputs['layout_q']}, layout_k={inputs['layout_k']}, "
              f"mask_mode={mask_mode}, cmp_ratio={cmp_ratio}", flush=True)
        metadata = cann_ops_transformer.dense_lightning_indexer_softmax_lse_metadata(
            num_heads_q=N1,
            num_heads_k=inputs["N2"],
            head_dim=D,
            cu_seqlens_q=_to(inputs["cu_seqlens_q"]),
            cu_seqlens_k=_to(inputs["cu_seqlens_k"]),
            seqused_q=_to(inputs["seqused_q"]),
            seqused_k=_to(inputs["seqused_k"]),
            cmp_residual_k=_to(inputs["cmp_residual_k"]),
            batch_size=B if not is_tnd else None,
            max_seqlen_q=S1 if not is_tnd else None,
            max_seqlen_k=S2 if not is_tnd else None,
            layout_q=inputs["layout_q"],
            layout_k=inputs["layout_k"],
            mask_mode=mask_mode,
            cmp_ratio=cmp_ratio,
        )
        torch.npu.synchronize()
        result["meta_stage"] = "OK"
        print(f"  [metadata] OK, metadata={tuple(metadata.shape)}", flush=True)
    except Exception as e:
        result["meta_stage"] = "REJECT"
        result["meta_msg"] = f"{type(e).__name__}: {str(e)[:500]}"
        return result

    try:
        _dump_npu_inputs("lse", {
            "q": inputs["q"],
            "k": inputs["k"],
            "w": inputs["w"],
            "cu_seqlens_q": inputs["cu_seqlens_q"],
            "cu_seqlens_k": inputs["cu_seqlens_k"],
            "seqused_q": inputs["seqused_q"],
            "seqused_k": inputs["seqused_k"],
            "cmp_residual_k": inputs["cmp_residual_k"],
            "metadata": metadata,
        })
        print(f"  [lse npu inputs] layout_q={inputs['layout_q']}, "
              f"layout_k={inputs['layout_k']}, mask_mode={mask_mode}, cmp_ratio={cmp_ratio}",
              flush=True)
        lse = cann_ops_transformer.dense_lightning_indexer_softmax_lse(
            inputs["q"].to(device), inputs["k"].to(device), inputs["w"].to(device),
            cu_seqlens_q=_to(inputs["cu_seqlens_q"]),
            cu_seqlens_k=_to(inputs["cu_seqlens_k"]),
            seqused_q=_to(inputs["seqused_q"]),
            seqused_k=_to(inputs["seqused_k"]),
            cmp_residual_k=_to(inputs["cmp_residual_k"]),
            metadata=metadata,
            layout_q=inputs["layout_q"],
            layout_k=inputs["layout_k"],
            mask_mode=mask_mode,
            cmp_ratio=cmp_ratio,
        )
        torch.npu.synchronize()
        result["main_stage"] = "NOT_REJECTED"
        result["main_msg"] = f"out shape={tuple(lse.shape)}"
    except Exception as e:
        result["main_stage"] = "REJECT"
        result["main_msg"] = f"{type(e).__name__}: {str(e)[:500]}"
    return result


def _write_errcheck_results_to_xlsx(xlsx_path, results):
    if openpyxl is None:
        print("[fuzz] openpyxl not available, skip writing results to xlsx")
        return
    import shutil
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(xlsx_path)
    out_path = f"{base}_errcheck_{timestamp}{ext}"
    shutil.copy2(xlsx_path, out_path)
    wb = openpyxl.load_workbook(out_path)
    ws = wb.active

    max_col = ws.max_column
    ws.cell(row=1, column=max_col + 1, value="err_result")
    ws.cell(row=1, column=max_col + 2, value="err_stage")
    ws.cell(row=1, column=max_col + 3, value="err_msg")

    case_name_to_row = {}
    for row_idx in range(2, ws.max_row + 1):
        name = ws.cell(row=row_idx, column=2).value
        if name is not None:
            case_name_to_row[name] = row_idx

    written = 0
    for case_name, final, stage, msg in results:
        row_idx = case_name_to_row.get(case_name)
        if row_idx is not None:
            ws.cell(row=row_idx, column=max_col + 1, value=final)
            ws.cell(row=row_idx, column=max_col + 2, value=stage)
            ws.cell(row=row_idx, column=max_col + 3, value=msg)
            written += 1

    wb.save(out_path)
    print(f"[fuzz] Err-check results written to {out_path} ({written}/{len(results)} cases)")


def _get_cache_path(case_name, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    safe_name = case_name.replace("/", "_").replace("\\", "_")
    return os.path.join(cache_dir, f"{safe_name}.pt")


def _save_case(case_name, cache_dir, inputs, lse_out, lse_ref):
    cache_path = _get_cache_path(case_name, cache_dir)
    save_dict = {
        "inputs": {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()},
        "lse_out": lse_out.cpu() if isinstance(lse_out, torch.Tensor) else lse_out,
        "lse_ref": lse_ref.cpu() if isinstance(lse_ref, torch.Tensor) else lse_ref,
    }
    torch.save(save_dict, cache_path)
    print(f"  [cache] saved to {cache_path}")


def _load_cached_case(case_name, cache_dir):
    cache_path = _get_cache_path(case_name, cache_dir)
    if not os.path.exists(cache_path):
        return None
    return torch.load(cache_path, map_location="cpu", weights_only=False)


def test_fuzz_from_xlsx(xlsx_path=None, cache_dir=None,
                        save_golden=False, use_cache=False, verbose_elem=False,
                        no_metadata=False, aic_core_num=DEFAULT_AIC_CORE_NUM,
                        l2_precision=False, ref_device="cpu", noblock=False,
                        err_check=False):
    if xlsx_path is None:
        xlsx_path = r"C:\Users\kjc\Desktop\pto相关\dli\测试用例\dense_lightning_indexer_softmax_lse_fuzz_0.5g - mk.xlsx"
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".dli_lse_cache")

    device = "npu:0"
    torch.npu.set_device(device)

    cases = _load_xlsx_cases(xlsx_path)
    print(f"[fuzz] Loaded {len(cases)} test cases from {xlsx_path}")
    print(f"[fuzz] save_golden={save_golden}, use_cache={use_cache}, cache_dir={cache_dir}")
    print(f"[fuzz] verbose_elem={verbose_elem}")
    if err_check:
        os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"
        print(f"[fuzz] ** ERR-CHECK MODE: NPU-only, faithful illegal inputs, no golden, ASCEND_LAUNCH_BLOCKING=1 **")
    if no_metadata:
        print(f"[fuzz] ** USING PYTHON METADATA (no NPU metadata op), aic_core_num={aic_core_num} **")
    if l2_precision:
        print(f"[fuzz] ** L2 PRECISION CHECK ENABLED (cpu_kernel as gpu reference) **")
    print(f"[fuzz] ref_device={ref_device}, noblock={noblock}")

    pass_count = 0
    fail_count = 0
    results = []
    err_results = []

    for idx, case in enumerate(cases):
        case_name = case.get("testcase_name", f"case_{idx}")
        B = case["B"]
        N1 = case["N1"]
        S1 = case["S1"]
        S2 = case["S2"]
        D = case["D"]
        mask_mode = case.get("mask_mode", 0)
        if mask_mode is None:
            mask_mode = 0
        cmp_ratio = case.get("cmp_ratio", 1)
        if cmp_ratio is None:
            cmp_ratio = 1
        layout_q = case.get("layout_q", "BSND")
        if layout_q is None:
            layout_q = "BSND"
        is_tnd = (layout_q == "TND")
        print(f"\n[{idx+1}/{len(cases)}] {case_name} (B={B}, N1={N1}, S1={S1}, S2={S2}, D={D}, "
              f"mask={mask_mode}, cmp={cmp_ratio}, layout={layout_q})")

        if err_check:
            try:
                res = _err_check_case(case, device)
                if res["meta_stage"] == "REJECT":
                    final, stage, msg = "REJECTED", "METADATA", res["meta_msg"]
                elif res["main_stage"] == "REJECT":
                    final, stage, msg = "REJECTED", "MAIN", res["main_msg"]
                elif res["main_stage"] == "NOT_REJECTED":
                    final, stage, msg = "NOT_REJECTED", "-", res["main_msg"]
                else:
                    final, stage, msg = "UNKNOWN", "-", ""
                print(f"  [ERR-CHECK] {final} ({stage}) {msg}")
                err_results.append((case_name, final, stage, msg))
                if final == "REJECTED":
                    pass_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                print(f"  [ERR-CHECK] BUILD_ERROR {type(e).__name__}: {e}")
                err_results.append((case_name, "BUILD_ERROR", "-", f"{type(e).__name__}: {e}"))
                fail_count += 1
            continue

        case_max_diff = None
        case_result = "ERROR"
        case_l2_mare_ratio = None
        case_l2_mere_ratio = None
        case_l2_rmse_ratio = None
        case_l2_result = None
        try:
            inputs = _build_case_inputs(case, device)

            if no_metadata:
                metadata = _compute_metadata_python(
                    num_heads_q=N1,
                    num_heads_k=inputs["N2"],
                    head_dim=D,
                    cu_seqlens_q=inputs["cu_seqlens_q"],
                    cu_seqlens_k=inputs["cu_seqlens_k"],
                    seqused_q=inputs["seqused_q"],
                    seqused_k=inputs["seqused_k"],
                    cmp_residual_k=inputs["cmp_residual_k"],
                    batch_size=B if not is_tnd else None,
                    max_seqlen_q=S1 if not is_tnd else None,
                    max_seqlen_k=S2 if not is_tnd else None,
                    layout_q=inputs["layout_q"],
                    layout_k=inputs["layout_k"],
                    mask_mode=mask_mode,
                    cmp_ratio=cmp_ratio,
                    aic_core_num=aic_core_num,
                    device="cpu",
                )
                metadata = metadata.to(device)
                print(f"  [metadata] fore={metadata[0].item()}, tail={metadata[1].item()}, "
                      f"per_fore={metadata[2].item()}, per_tail={metadata[3].item()}")
            else:
                metadata = cann_ops_transformer.dense_lightning_indexer_softmax_lse_metadata(
                    num_heads_q=N1,
                    num_heads_k=inputs["N2"],
                    head_dim=D,
                    cu_seqlens_q=inputs["cu_seqlens_q"].to(device) if inputs["cu_seqlens_q"] is not None else None,
                    cu_seqlens_k=inputs["cu_seqlens_k"].to(device) if inputs["cu_seqlens_k"] is not None else None,
                    seqused_q=inputs["seqused_q"].to(device) if inputs["seqused_q"] is not None else None,
                    seqused_k=inputs["seqused_k"].to(device) if inputs["seqused_k"] is not None else None,
                    cmp_residual_k=inputs["cmp_residual_k"].to(device) if inputs["cmp_residual_k"] is not None else None,
                    batch_size=B if not is_tnd else None,
                    max_seqlen_q=S1 if not is_tnd else None,
                    max_seqlen_k=S2 if not is_tnd else None,
                    layout_q=inputs["layout_q"],
                    layout_k=inputs["layout_k"],
                    mask_mode=mask_mode,
                    cmp_ratio=cmp_ratio,
                )
                torch.npu.synchronize()

            lse = cann_ops_transformer.dense_lightning_indexer_softmax_lse(
                inputs["q"].to(device), inputs["k"].to(device), inputs["w"].to(device),
                cu_seqlens_q=inputs["cu_seqlens_q"].to(device) if inputs["cu_seqlens_q"] is not None else None,
                cu_seqlens_k=inputs["cu_seqlens_k"].to(device) if inputs["cu_seqlens_k"] is not None else None,
                seqused_q=inputs["seqused_q"].to(device) if inputs["seqused_q"] is not None else None,
                seqused_k=inputs["seqused_k"].to(device) if inputs["seqused_k"] is not None else None,
                cmp_residual_k=inputs["cmp_residual_k"].to(device) if inputs["cmp_residual_k"] is not None else None,
                metadata=metadata,
                layout_q=inputs["layout_q"],
                layout_k=inputs["layout_k"],
                mask_mode=mask_mode,
                cmp_ratio=cmp_ratio,
            )
            torch.npu.synchronize()
            do_golden = True
            if use_cache:
                cached = _load_cached_case(case_name, cache_dir)
                if cached is not None:
                    lse_ref = cached["lse_ref"].to(device)
                    do_golden = False
                    print(f"  [cache] loaded golden from cache, skipping golden computation")
                else:
                    print(f"  [cache] no cached golden found, computing golden (use --save-golden to cache for next run)")
            if do_golden:
                if noblock:
                    golden_tnd = golden_dli_lse_tnd_noblock
                    golden_blk = golden_dli_lse_blk_noblock
                else:
                    golden_tnd = golden_dli_lse_tnd
                    golden_blk = golden_dli_lse_blk
                if is_tnd:
                    lse_ref = golden_tnd(
                        inputs["q"], inputs["k"], inputs["w"],
                        inputs["cu_seqlens_q"],
                        inputs["cu_seqlens_k"],
                        seqused_q=inputs["seqused_q"],
                        seqused_k=inputs["seqused_k"],
                        cmp_residual_k=inputs["cmp_residual_k"],
                        mask_mode=mask_mode,
                        cmp_ratio=cmp_ratio,
                        device=ref_device)
                else:
                    lse_ref = golden_blk(
                        inputs["q"], inputs["k"], inputs["w"],
                        seqused_q=inputs["seqused_q"],
                        seqused_k=inputs["seqused_k"],
                        cmp_residual_k=inputs["cmp_residual_k"],
                        mask_mode=mask_mode,
                        cmp_ratio=cmp_ratio,
                        device=ref_device)

            if save_golden and do_golden:
                _save_case(case_name, cache_dir, inputs, lse.cpu(), lse_ref.cpu() if lse_ref.is_npu else lse_ref)

            lse_cpu = lse.cpu().float()
            lse_ref_f = lse_ref.cpu().float() if lse_ref.is_npu else lse_ref.float()
            if lse_cpu.shape != lse_ref_f.shape:
                print(f"  [FAIL] shape mismatch: lse={lse_cpu.shape} vs lse_ref={lse_ref_f.shape}")
                case_max_diff = None
                case_result = "FAIL"
                fail_count += 1
                results.append((case_name, case_max_diff, case_result,
                                case_l2_mare_ratio, case_l2_mere_ratio, case_l2_rmse_ratio, case_l2_result))
                continue

            if verbose_elem:
                print(f"  [verbose_elem] === {case_name} ===")
                if is_tnd:
                    cu_q = inputs["cu_seqlens_q"]
                    B_local = cu_q.shape[0] - 1
                    for b in range(B_local):
                        q_start = cu_q[b].item()
                        q_end = cu_q[b + 1].item()
                        for s1_i in range(q_end - q_start):
                            idx = q_start + s1_i
                            print(f"    [b={b}, s1={s1_i}] npu={lse_cpu[0, idx].item():.6f}  "
                                  f"golden={lse_ref_f[0, idx].item():.6f}")
                else:
                    for b in range(B):
                        for s1_i in range(S1):
                            print(f"    [b={b}, s1={s1_i}] npu={lse_cpu[b, 0, s1_i].item():.6f}  "
                                  f"golden={lse_ref_f[b, 0, s1_i].item():.6f}")

            if is_tnd:
                T1_cmp = lse_cpu.shape[1]
                valid_mask = _compute_row_valid_mask_tnd(
                    inputs["cu_seqlens_q"], inputs["cu_seqlens_k"], T1_cmp,
                    inputs["seqused_q"],
                    inputs["seqused_k"],
                    inputs["cmp_residual_k"], mask_mode, cmp_ratio, "cpu")
            else:
                valid_mask = _compute_row_valid_mask_blk(
                    B, S1, S2,
                    inputs["seqused_q"],
                    inputs["seqused_k"],
                    inputs["cmp_residual_k"], mask_mode, cmp_ratio, "cpu")

            if is_tnd:
                valid_mask = valid_mask.unsqueeze(0)  # [1, T1]
            else:
                valid_mask = valid_mask.unsqueeze(1)  # [B, 1, S1]

            diff = (lse_cpu - lse_ref_f).abs()
            diff_valid = diff[valid_mask]
            nan_count = torch.isnan(lse_cpu[valid_mask]).sum().item()
            inf_count = torch.isinf(lse_cpu[valid_mask]).sum().item()
            if diff_valid.numel() == 0:
                max_diff = 0.0
                max_rel_diff = 0.0
                has_nan_diff = False
            else:
                has_nan_diff = bool(torch.isnan(diff_valid).any().item())
                max_diff = diff_valid.max().item()
                if has_nan_diff:
                    max_diff = float('inf')
                a_valid = lse_cpu[valid_mask]
                b_valid = lse_ref_f[valid_mask]
                denom = torch.max(a_valid.abs(), b_valid.abs()).clamp(min=1e-8)
                rel_diff_valid = torch.nan_to_num(
                    diff_valid / denom, nan=float('inf'), posinf=float('inf'), neginf=float('inf'))
                max_rel_diff = rel_diff_valid.max().item()

            atol = 5e-2
            rtol = 5e-2
            case_max_diff = max_diff
            if max_diff > atol or max_rel_diff > rtol or nan_count > 0 or has_nan_diff:
                case_result = "FAIL"
                print(f"  [FAIL] max|diff|={max_diff:.6f} (atol={atol}), "
                      f"max|rel_diff|={max_rel_diff:.6f} (rtol={rtol}), nan={nan_count}, "
                      f"inf={inf_count}, nan_in_diff={has_nan_diff}")
                fail_count += 1
            else:
                case_result = "PASS"
                print(f"  [PASS] max|diff|={max_diff:.6f}, max|rel_diff|={max_rel_diff:.6f}")
                pass_count += 1

            case_l2_mare_ratio = None
            case_l2_mere_ratio = None
            case_l2_rmse_ratio = None
            case_l2_result = None
            if l2_precision:
                if noblock:
                    cpu_kernel_tnd = cpu_kernel_dli_lse_tnd_noblock
                    cpu_kernel_blk = cpu_kernel_dli_lse_blk_noblock
                else:
                    cpu_kernel_tnd = cpu_kernel_dli_lse_tnd
                    cpu_kernel_blk = cpu_kernel_dli_lse_blk
                if is_tnd:
                    lse_gpu = cpu_kernel_tnd(
                        inputs["q"], inputs["k"], inputs["w"],
                        inputs["cu_seqlens_q"],
                        inputs["cu_seqlens_k"],
                        seqused_q=inputs["seqused_q"],
                        seqused_k=inputs["seqused_k"],
                        cmp_residual_k=inputs["cmp_residual_k"],
                        mask_mode=mask_mode,
                        cmp_ratio=cmp_ratio,
                        device=ref_device)
                else:
                    lse_gpu = cpu_kernel_blk(
                        inputs["q"], inputs["k"], inputs["w"],
                        seqused_q=inputs["seqused_q"],
                        seqused_k=inputs["seqused_k"],
                        cmp_residual_k=inputs["cmp_residual_k"],
                        mask_mode=mask_mode,
                        cmp_ratio=cmp_ratio,
                        device=ref_device)

                lse_gpu_cpu = lse_gpu.cpu().float() if lse_gpu.is_npu else lse_gpu.float()
                npu_v = lse_cpu[valid_mask].float()
                golden_v = lse_ref_f[valid_mask].float()
                gpu_v = lse_gpu_cpu[valid_mask].float()
                for i in range(len(lse_cpu.flatten())):
                    print(f"test, lse_gpu[{i}] = {(lse_gpu_cpu.flatten())[i]}")
                    print(f"test, lse_cpu[{i}] = {(lse_cpu.flatten())[i]}")
                    print(f"test, lse_ref_f[{i}] = {(lse_ref_f.flatten())[i]}")
                # for i in range(len(gpu_v)):
                #     print(f"test, lse_gpu[{i}] = {gpu_v[i]}")
                #     print(f"test, lse_cpu[{i}] = {npu_v[i]}")
                #     print(f"test, lse_ref_f[{i}] = {golden_v[i]}")
                l2_result_dict = check_with_gpu_compare_standard(golden_v, npu_v, gpu_v, level=2)
                case_l2_mare_ratio = l2_result_dict.get("mare_ratio")
                case_l2_mere_ratio = l2_result_dict.get("mere_ratio")
                case_l2_rmse_ratio = l2_result_dict.get("rmse_ratio")
                l2_error_keys = [k for k, v in l2_result_dict.items()
                                 if v == "error" and k != "eb"]
                l2_passed = len(l2_error_keys) == 0
                mare_str = (f"MARE: npu={l2_result_dict['mare_npu']:.6g} "
                            f"gpu={l2_result_dict['mare_gpu']:.6g} "
                            f"ratio={l2_result_dict['mare_ratio']:.4f}")
                mere_str = (f"MERE: npu={l2_result_dict['mere_npu']:.6g} "
                            f"gpu={l2_result_dict['mere_gpu']:.6g} "
                            f"ratio={l2_result_dict['mere_ratio']:.4f}")
                rmse_str = (f"RMSE: npu={l2_result_dict['rmse_npu']:.6g} "
                            f"gpu={l2_result_dict['rmse_gpu']:.6g} "
                            f"ratio={l2_result_dict['rmse_ratio']:.4f}")
                if l2_passed:
                    case_l2_result = "PASS"
                    print(f"  [L2 PASS] infnan={l2_result_dict['infnan']} "
                          f"smallvalue={l2_result_dict['smallvalue']} "
                          f"max_re={l2_result_dict['max_re']} "
                          f"avg_re={l2_result_dict['avg_re']} "
                          f"rmse={l2_result_dict['rmse']} "
                          f"eb={l2_result_dict['eb']}")
                    print(f"           {mare_str} | {mere_str} | {rmse_str}")
                else:
                    case_l2_result = "FAIL"
                    print(f"  [L2 FAIL] error_metrics={l2_error_keys} | "
                          f"infnan={l2_result_dict['infnan']} "
                          f"smallvalue={l2_result_dict['smallvalue']} "
                          f"max_re={l2_result_dict['max_re']} "
                          f"avg_re={l2_result_dict['avg_re']} "
                          f"rmse={l2_result_dict['rmse']} "
                          f"eb={l2_result_dict['eb']}")
                    print(f"           {mare_str} | {mere_str} | {rmse_str}")

        except Exception as e:
            case_result = "ERROR"
            print(f"  [ERROR] {type(e).__name__}: {e}")
            fail_count += 1

        results.append((case_name, case_max_diff, case_result,
                        case_l2_mare_ratio, case_l2_mere_ratio, case_l2_rmse_ratio, case_l2_result))

    print(f"\n[fuzz] Summary: {pass_count} passed, {fail_count} failed")

    if err_check:
        _write_errcheck_results_to_xlsx(xlsx_path, err_results)
    else:
        _write_results_to_xlsx(xlsx_path, results)


def _write_results_to_xlsx(xlsx_path, results):
    if openpyxl is None:
        print("[fuzz] openpyxl not available, skip writing results to xlsx")
        return
    import shutil
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(xlsx_path)
    out_path = f"{base}_result_{timestamp}{ext}"
    shutil.copy2(xlsx_path, out_path)
    wb = openpyxl.load_workbook(out_path)
    ws = wb.active

    max_col = ws.max_column
    ws.cell(row=1, column=max_col + 1, value="max_diff")
    ws.cell(row=1, column=max_col + 2, value="result")
    has_l2 = len(results) > 0 and len(results[0]) > 3 and results[0][3] is not None
    if has_l2:
        ws.cell(row=1, column=max_col + 3, value="l2_mare_ratio")
        ws.cell(row=1, column=max_col + 4, value="l2_mere_ratio")
        ws.cell(row=1, column=max_col + 5, value="l2_rmse_ratio")
        ws.cell(row=1, column=max_col + 6, value="l2_result")

    case_name_to_row = {}
    for row_idx in range(2, ws.max_row + 1):
        name = ws.cell(row=row_idx, column=2).value
        if name is not None:
            case_name_to_row[name] = row_idx

    written = 0
    for row_data in results:
        case_name = row_data[0]
        max_diff = row_data[1]
        result = row_data[2]
        row_idx = case_name_to_row.get(case_name)
        if row_idx is not None:
            ws.cell(row=row_idx, column=max_col + 1, value=max_diff)
            ws.cell(row=row_idx, column=max_col + 2, value=result)
            if has_l2 and len(row_data) > 6:
                ws.cell(row=row_idx, column=max_col + 3, value=row_data[3])
                ws.cell(row=row_idx, column=max_col + 4, value=row_data[4])
                ws.cell(row=row_idx, column=max_col + 5, value=row_data[5])
                ws.cell(row=row_idx, column=max_col + 6, value=row_data[6])
            written += 1

    wb.save(out_path)
    print(f"[fuzz] Results written to {out_path} ({written}/{len(results)} cases)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DenseLightningIndexerSoftmaxLseV2 fuzz test from xlsx")
    parser.add_argument("--xlsx", type=str, default=None,
                        help="Path to xlsx test case file")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Directory to save/load cached golden results")
    parser.add_argument("--save-golden", action="store_true",
                        help="Save golden results to cache_dir for future use")
    parser.add_argument("--use-cache", action="store_true",
                        help="Use cached golden results instead of computing")
    parser.add_argument("--verbose-elem", action="store_true",
                        help="Print lse element-by-element with [b, s1] indices")
    parser.add_argument("--no-metadata", action="store_true",
                        help="Use Python metadata computation instead of NPU metadata op")
    parser.add_argument("--aic-core-num", type=int, default=DEFAULT_AIC_CORE_NUM,
                        help=f"AIC core count for Python metadata (default: {DEFAULT_AIC_CORE_NUM}, try 36 if results wrong)")
    parser.add_argument("--l2-precision", action="store_true",
                        help="Enable L2 precision check (MARE/MERE/RMSE ratio, cpu_kernel as gpu reference)")
    parser.add_argument("--ref-device", type=str, default="cpu", choices=["cpu", "npu"],
                        help="Device for golden/cpu_kernel reference computation (default: cpu)")
    parser.add_argument("--noblock", action="store_true",
                        help="Use non-blocked variant for both golden and cpu_kernel (one-shot matmul, no TKV tiling)")
    parser.add_argument("--err-check", action="store_true",
                        help="NPU-only mode: faithfully construct illegal inputs, call NPU ops, record rejection (no golden)")
    args = parser.parse_args()

    ref_device = "npu:0" if args.ref_device == "npu" else "cpu"
    if ref_device == "npu:0":
        torch.npu.set_device(ref_device)

    test_fuzz_from_xlsx(
        xlsx_path=args.xlsx,
        cache_dir=args.cache_dir,
        save_golden=args.save_golden,
        use_cache=args.use_cache,
        verbose_elem=args.verbose_elem,
        no_metadata=args.no_metadata,
        aic_core_num=args.aic_core_num,
        l2_precision=args.l2_precision,
        ref_device=ref_device,
        noblock=args.noblock,
        err_check=args.err_check,
    )
