#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
FIA MLA D512 non-quant tiling template verification.

Case 1 (new template, isReconstructTemp=true):
  B=1, N1=2, N2=1, S1=1, S2=512, qk_d=512, rope_d=64, v_d=512
  no mask, sparse_mode=0, BSH layout
  → should route to FiaTilingNonQuantMlaArch35 (priority 20)
  → tilingKey with config=9, isReconstructTemp=true

Case 2 (old template, isReconstructTemp=false):
  Same shape but WITH mask (attenMaskFlag=true)
  → FiaTilingNonQuantMlaArch35.IsCapable rejects mask
  → falls through to old impl (priority 28 GQA also rejects ROPE_SPLIT_D512)
  → old impl sets isReconstructTemp=false, routes to FAKernelNoquantMla
"""

import torch
import torch_npu
import numpy as np
import sys
import os

torch.npu.config.allow_internal_format = True
DEVICE_ID = 0


def rand_range(shape, data_range=(-10, 10), dtype=torch.float32):
    return (torch.rand(shape, dtype=dtype) * (data_range[1] - data_range[0]) + data_range[0])


def run_case(b, n1, n2, s1, s2, qk_d, v_d, rope_d, enable_mask=False, sparse_mode=0,
             input_layout="BNSD", dtype=torch.bfloat16):
    g = n1 // n2
    scale = 1.0 / (qk_d ** 0.5)

    if input_layout == "BNSD":
        query = rand_range((b, n1, s1, qk_d), dtype=dtype).npu()
        key = rand_range((b, n2, s2, qk_d), dtype=dtype).npu()
        value = rand_range((b, n2, s2, v_d), dtype=dtype).npu()
        if rope_d > 0:
            query_rope = rand_range((b, n1, s1, rope_d), dtype=dtype).npu()
            key_rope = rand_range((b, n2, s2, rope_d), dtype=dtype).npu()
        else:
            query_rope = None
            key_rope = None
    else:  # BSH
        query = rand_range((b, s1, n1 * qk_d), dtype=dtype).npu()
        key = rand_range((b, s2, n2 * qk_d), dtype=dtype).npu()
        value = rand_range((b, s2, n2 * v_d), dtype=dtype).npu()
        if rope_d > 0:
            query_rope = rand_range((b, s1, n1 * rope_d), dtype=dtype).npu()
            key_rope = rand_range((b, s2, n2 * rope_d), dtype=dtype).npu()
        else:
            query_rope = None
            key_rope = None

    atten_mask = None
    if enable_mask:
        if sparse_mode == 0:
            atten_mask = torch.randint(0, 2, (b, s1, s2), dtype=torch.bool).npu()
        else:
            atten_mask = torch.randint(0, 2, (s1, s2), dtype=torch.bool).npu()

    act_seq_q = [s1]
    act_seq_kv = [s2]

    print(f"\n{'='*80}")
    print(f"Case: B={b}, N1={n1}, N2={n2}, G={g}, S1={s1}, S2={s2}")
    print(f"  qk_d={qk_d}, v_d={v_d}, rope_d={rope_d}")
    print(f"  mask={enable_mask}, sparse_mode={sparse_mode}, layout={input_layout}, dtype={dtype}")
    print(f"{'='*80}")

    try:
        attn_out, softmax_lse = torch_npu.npu_fused_infer_attention_score(
            query, key, value,
            query_rope=query_rope,
            key_rope=key_rope,
            atten_mask=atten_mask,
            actual_seq_lengths=act_seq_q,
            actual_seq_lengths_kv=act_seq_kv,
            num_heads=n1,
            scale=scale,
            pre_tokens=2147483647,
            next_tokens=2147483647,
            input_layout=input_layout,
            num_key_value_heads=n2,
            sparse_mode=sparse_mode,
            inner_precise=0,
            block_size=0,
            antiquant_mode=0,
            softmax_lse_flag=False,
            key_antiquant_mode=0,
            value_antiquant_mode=0,
        )
        print(f"[OK] attn_out shape={attn_out.shape}, dtype={attn_out.dtype}")
        print(f"     attn_out sample (first 5): {attn_out.flatten()[:5].cpu()}")
        if softmax_lse is not None:
            print(f"     softmax_lse shape={softmax_lse.shape}")
        return True
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    torch.npu.set_device(DEVICE_ID)

    print(">>> Case 1: MLA D512 BNSD no mask (should hit NEW template, isReconstructTemp=true)")
    print("    Expected: tilingKey with config=9, isReconstructTemp=true")
    print("    If custom pkg not compiled for this tilingKey → kernel not found error")
    ok1 = run_case(
        b=1, n1=2, n2=1, s1=1, s2=512,
        qk_d=512, v_d=512, rope_d=64,
        enable_mask=False, sparse_mode=0,
        input_layout="BNSD", dtype=torch.bfloat16,
    )

    print("\n>>> Case 2: MLA D512 BSH layout (should hit OLD template, isReconstructTemp=false)")
    print("    New template IsCapable only accepts BNSD → falls to old impl")
    ok2 = run_case(
        b=1, n1=2, n2=1, s1=1, s2=512,
        qk_d=512, v_d=512, rope_d=64,
        enable_mask=False, sparse_mode=0,
        input_layout="BSH", dtype=torch.bfloat16,
    )

    print(f"\n{'='*80}")
    print(f"Summary: Case1(new template)={'PASS' if ok1 else 'FAIL'}, "
          f"Case2(old template)={'PASS' if ok2 else 'FAIL'}")
    print(f"{'='*80}")
