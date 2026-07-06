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


def test_main_op_tiling_trigger():
    """调用主算子触发 tiling，kernel 未实现预期会失败。
    通过日志确认 tiling 被触发（在 RuntimeError 之前输出 tiling completed 日志）。
    """
    B, S1, S2, N1, D = 2, 128, 256, 8, 64
    q = torch.randn(B, S1, N1, D, dtype=torch.float16).npu()
    k = torch.randn(B, S2, 1, D, dtype=torch.float16).npu()
    w = torch.randn(B, S1, N1, dtype=torch.float16).npu()

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

    try:
        lse = cann_ops_transformer.dense_lightning_indexer_softmax_lse_v2(
            q, k, w,
            metadata=metadata,
            layout_q="BSND",
            layout_k="BSND",
            mask_mode=0,
            cmp_ratio=1,
        )
        print(f"[main op] lse shape={lse.shape} (unexpected success, kernel may be available)")
    except RuntimeError as e:
        print(f"[main op] Expected RuntimeError (kernel not implemented):")
        print(f"  {e}")
        print("[main op] Check log for 'tiling completed' to verify tiling was triggered.")


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
    test_bsnd_metadata()
    test_tnd_metadata()
    test_bsnd_metadata_with_seqused()
    test_main_op_tiling_trigger()
    test_invalid_num_heads_q()
    test_invalid_cmp_ratio()
    print("All tests passed!")
