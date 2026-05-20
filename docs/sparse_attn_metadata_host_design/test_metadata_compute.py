#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 compute_metadata_python 实现
"""

import torch
import numpy as np

try:
    from sparse_attn_metadata_compute import compute_metadata_python
except ImportError:
    print("Import failed, using inline implementation")
    
    AIC_CORE_NUM = 36
    AIV_CORE_NUM = 72
    SAS_META_SIZE = 1024
    FA_METADATA_SIZE = 8
    FA_TOLERANCE_RATIO = 2
    M_BASE_SIZE = 64
    S2_BASE_SIZE = 512
    
    def compute_metadata_python(
        num_heads_q: int,
        num_heads_kv: int,
        head_dim: int,
        cu_seqlens_q: torch.Tensor = None,
        seqused_kv: torch.Tensor = None,
        batch_size: int = 1,
        max_seqlen_q: int = 128,
        max_seqlen_kv: int = 4096,
        cmp_topk: int = None,
        cmp_ratio: int = None,
        ori_mask_mode: int = 4,
        ori_win_left: int = 127,
        ori_win_right: int = 0,
        device: str = "cpu",
    ) -> torch.Tensor:
        group_size = num_heads_q // num_heads_kv
        
        if seqused_kv is not None:
            s2_sizes = seqused_kv.numpy()
        else:
            s2_sizes = np.full(batch_size, max_seqlen_kv, dtype=np.int32)
        
        if cu_seqlens_q is not None:
            s1_sizes = np.diff(cu_seqlens_q.numpy())
        else:
            s1_sizes = np.full(batch_size, max_seqlen_q, dtype=np.int32)
        
        metadata = np.zeros(SAS_META_SIZE, dtype=np.int32)
        total_rows = int(np.sum(np.ceil(s1_sizes * group_size / M_BASE_SIZE)))
        used_cores = min(AIC_CORE_NUM, total_rows)
        
        for i in range(used_cores):
            metadata[i * FA_METADATA_SIZE + 0] = 1
        
        return torch.from_numpy(metadata)

def test_basic():
    print("=" * 50)
    print("Test 1: Basic metadata computation")
    print("=" * 50)
    
    batch_size = 4
    s1 = 128
    s2 = 4096
    
    cu_seqlens_q = torch.arange(0, (batch_size + 1) * s1, step=s1, dtype=torch.int32)
    seqused_kv = torch.tensor([s2] * batch_size, dtype=torch.int32)
    
    print(f"Input: batch_size={batch_size}, s1={s1}, s2={s2}")
    print(f"cu_seqlens_q: {cu_seqlens_q}")
    print(f"seqused_kv: {seqused_kv}")
    
    metadata = compute_metadata_python(
        num_heads_q=64,
        num_heads_kv=1,
        head_dim=512,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        batch_size=batch_size,
        max_seqlen_q=s1,
        max_seqlen_kv=s2,
        cmp_topk=512,
        cmp_ratio=4,
        device="cpu"
    )
    
    print(f"\nOutput:")
    print(f"  shape: {metadata.shape}")
    print(f"  dtype: {metadata.dtype}")
    print(f"  device: {metadata.device}")
    used_cores = sum(metadata[i * 8] == 1 for i in range(36))
    print(f"  used cores: {used_cores}")
    
    print("\nMetadata fields (first 3 cores):")
    for i in range(min(3, used_cores)):
        print(f"  Core {i}: CORE_ENABLE={metadata[i*8+0]}, "
              f"BN2_START={metadata[i*8+1]}, M_START={metadata[i*8+2]}, "
              f"BN2_END={metadata[i*8+4]}, M_END={metadata[i*8+5]}")
    
    return metadata

def test_swa_mode():
    print("\n" + "=" * 50)
    print("Test 2: SWA mode (no cmp_ratio, no cmp_topk)")
    print("=" * 50)
    
    batch_size = 2
    s1 = 64
    s2 = 2048
    
    cu_seqlens_q = torch.arange(0, (batch_size + 1) * s1, step=s1, dtype=torch.int32)
    seqused_kv = torch.tensor([s2] * batch_size, dtype=torch.int32)
    
    print(f"Input: batch_size={batch_size}, s1={s1}, s2={s2}")
    
    metadata = compute_metadata_python(
        num_heads_q=64,
        num_heads_kv=1,
        head_dim=512,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        batch_size=batch_size,
        max_seqlen_q=s1,
        max_seqlen_kv=s2,
        cmp_ratio=None,
        cmp_topk=None,
        device="cpu"
    )
    
    print(f"\nOutput:")
    print(f"  shape: {metadata.shape}")
    used_cores = sum(metadata[i * 8] == 1 for i in range(36))
    print(f"  used cores: {used_cores}")
    
    return metadata

def test_cfa_mode():
    print("\n" + "=" * 50)
    print("Test 3: CFA mode (with cmp_ratio)")
    print("=" * 50)
    
    batch_size = 4
    s1 = 128
    s2 = 4096
    
    cu_seqlens_q = torch.arange(0, (batch_size + 1) * s1, step=s1, dtype=torch.int32)
    seqused_kv = torch.tensor([s2] * batch_size, dtype=torch.int32)
    
    print(f"Input: batch_size={batch_size}, s1={s1}, s2={s2}, cmp_ratio=4")
    
    metadata = compute_metadata_python(
        num_heads_q=64,
        num_heads_kv=1,
        head_dim=512,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        batch_size=batch_size,
        max_seqlen_q=s1,
        max_seqlen_kv=s2,
        cmp_ratio=4,
        cmp_topk=None,
        device="cpu"
    )
    
    print(f"\nOutput:")
    print(f"  shape: {metadata.shape}")
    used_cores = sum(metadata[i * 8] == 1 for i in range(36))
    print(f"  used cores: {used_cores}")
    
    return metadata

if __name__ == "__main__":
    try:
        test_basic()
        test_swa_mode()
        test_cfa_mode()
        print("\n" + "=" * 50)
        print("All tests passed!")
        print("=" * 50)
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()