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


def _golden_dense_lightning_index_lse(query_index, key_index, weight):
    """Golden: LSE of (relu(q@k^T) * weight) summed over N1, then LogSumExp over S2."""
    q = query_index.float()
    k = key_index.float()
    w = weight.float()
    # q: [B, S1, N1, D], k: [B, S2, 1, D]
    score = torch.matmul(q, k.permute(0, 2, 3, 1))  # [B, S1, N1, S2]
    relu_score = torch.relu(score)
    weighted = relu_score * w.unsqueeze(-1)          # [B, S1, N1, S2]
    x = weighted.sum(dim=-2)                          # [B, S1, S2]
    m = x.max(dim=-1).values                          # [B, S1]
    lse = m + torch.log(torch.exp(x - m.unsqueeze(-1)).sum(dim=-1))  # [B, S1]
    return lse

def golden_dli_lse_blk(query_index, key_index, weight, seqused_q=None, seqused_k=None,
                       cmp_residual_k=None, mask_mode=0, cmp_ratio=1):
    q = query_index.float()
    k = key_index.float()
    w = weight.float()
    B, S1, N1, D = q.shape
    _, S2, _, _ = k.shape
    dev = q.device

    TKV = 128
    neg_inf = -3.4028234663852886e+38

    lse = torch.zeros(B, S1, dtype=torch.float32, device=dev)

    for b in range(B):
        q_b = q[b]
        k_b = k[b]
        w_b = w[b]

        q_used = S1 if seqused_q is None else seqused_q[b, 0].item()
        k_used = S2 if seqused_k is None else seqused_k[b, 0].item()

        x_full = torch.zeros(S1, S2, dtype=torch.float32, device=dev)

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
                valid_k_size = max((s1_i + 1 + ori_k_size - q_used) // cmp_ratio, 0)
                if valid_k_size < k_used:
                    x_full[s1_i, valid_k_size:] = 0

        reduce_max = x_full.amax(dim=1)
        valid = x_full.abs().sum(dim=1) > 0
        shifted = x_full - reduce_max.unsqueeze(1)
        sum_exp = torch.exp(shifted).sum(dim=1)
        lse[b] = torch.where(valid, reduce_max + torch.log(sum_exp), torch.zeros_like(reduce_max))

    return lse

def test_bsnd_metadata():
    """BSND layout: 调用 metadata 算子。"""
    B, S1, S2, N1, D = 2, 128, 256, 8, 64
    metadata = cann_ops_transformer.dense_lightning_indexer_softmax_lse_v2_metadata(
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
    metadata = cann_ops_transformer.dense_lightning_indexer_softmax_lse_v2_metadata(
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
    metadata = cann_ops_transformer.dense_lightning_indexer_softmax_lse_v2_metadata(
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
        (1, 4096, 1024, 128, 128, 0, 4, True, True),
        (1, 65536, 16384, 128, 128, 0, 4, True, True),
        (1, 65536, 16384, 128, 128, 3, 4, True, True),
        (1, 4096, 1024, 128, 128, 3, 4, True, True),
    ]
    for B, S1, S2, N1, D, sparse_mode, cmp_ratio, q_seqused, k_seqused in test_cases:
        print(f"\n--- DenseLightningIndexLse (B={B}, S1={S1}, S2={S2}, N1={N1}, D={D}, "
              f"SPARSE_MODE={sparse_mode}, CMP_RATIO={cmp_ratio}) ---")
        q = (torch.rand((B, S1, N1, D), device=device, dtype=torch.float16) * 2 - 1)
        k = (torch.rand((B, S2, 1, D), device=device, dtype=torch.float16) * 2 - 1)
        w = (torch.rand((B, S1, N1), device=device, dtype=torch.float32) * 2 - 1)

        seqused_q = torch.full((B, 1), S1, device=device, dtype=torch.int32) if q_seqused else None
        seqused_k = torch.full((B, 1), S2, device=device, dtype=torch.int32) if k_seqused else None
        cmp_residual_k = torch.zeros((B, 1), device=device, dtype=torch.int32)

        metadata = cann_ops_transformer.dense_lightning_indexer_softmax_lse_v2_metadata(
            num_heads_q=N1,
            num_heads_k=1,
            head_dim=D,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            cmp_residual_k=cmp_residual_k,
            batch_size=B,
            max_seqlen_q=S1,
            max_seqlen_k=S2,
            layout_q="BSND",
            layout_k="BSND",
            mask_mode=sparse_mode,
            cmp_ratio=cmp_ratio,
        )

        lse = cann_ops_transformer.dense_lightning_indexer_softmax_lse_v2(
            q, k, w,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            cmp_residual_k=cmp_residual_k,
            metadata=metadata,
            layout_q="BSND",
            layout_k="BSND",
            mask_mode=sparse_mode,
            cmp_ratio=cmp_ratio,
        )
        torch.npu.synchronize()

        lse_ref = golden_dli_lse_blk(q.cpu(), k.cpu(), w.cpu(),
                                     seqused_q=seqused_q.cpu() if seqused_q is not None else None,
                                     seqused_k=seqused_k.cpu() if seqused_k is not None else None,
                                     cmp_residual_k=cmp_residual_k.cpu(),
                                     mask_mode=sparse_mode,
                                     cmp_ratio=cmp_ratio)
        diff = (lse.cpu().float() - lse_ref).abs().max().item()
        print(f"[e2e] max|diff|={diff:.6f}")
        torch.testing.assert_close(lse.cpu().float(), lse_ref, rtol=5e-2, atol=5e-2)
        print("[e2e] PASS")


def test_invalid_num_heads_q():
    """num_heads_q 不能被 num_heads_k 整除应抛出异常。"""
    try:
        cann_ops_transformer.dense_lightning_indexer_softmax_lse_v2_metadata(
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


def test_invalid_cmp_ratio():
    """cmp_ratio 超范围应抛出异常。"""
    try:
        cann_ops_transformer.dense_lightning_indexer_softmax_lse_v2_metadata(
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


if __name__ == "__main__":
    # test_bsnd_metadata()
    # test_tnd_metadata()
    # test_bsnd_metadata_with_seqused()
    test_e2e()
    # test_invalid_num_heads_q()
    # test_invalid_cmp_ratio()
    print("All tests passed!")
