# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to the License for details.

import math
import os

import torch
import torch_npu

from npu_ops_transformer.ops import npu_sparse_flash_mla, npu_sparse_flash_mla_metadata


def main():
    device_id = int(os.environ.get("DEVICE_ID", "2"))
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"
    torch.manual_seed(0)

    batch = 1
    s1 = 16
    ori_len = 10
    cmp_ratio = 4
    cmp_len = 2
    residual = ori_len - cmp_len * cmp_ratio
    nq = 4
    nkv = 1
    head_dim = 512

    q = torch.randn((s1, nq, head_dim), device=device, dtype=torch.float16)
    ori_kv = torch.randn((ori_len, nkv, head_dim), device=device, dtype=torch.float16)
    cmp_kv = torch.randn((cmp_len, nkv, head_dim), device=device, dtype=torch.float16)

    cu_q = torch.tensor([0, s1], device=device, dtype=torch.int32)
    cu_ori = torch.tensor([0, ori_len], device=device, dtype=torch.int32)
    cu_cmp = torch.tensor([0, cmp_len], device=device, dtype=torch.int32)
    seq_q = torch.tensor([s1], device=device, dtype=torch.int32)
    seq_ori = torch.tensor([ori_len], device=device, dtype=torch.int32)
    cmp_residual = torch.tensor([residual], device=device, dtype=torch.int32)
    sinks = torch.zeros((nq,), device=device, dtype=torch.float32)

    metadata = npu_sparse_flash_mla_metadata(
        nq,
        nkv,
        head_dim,
        cu_seqlens_q=cu_q,
        cu_seqlens_ori_kv=cu_ori,
        cu_seqlens_cmp_kv=cu_cmp,
        seqused_q=seq_q,
        seqused_ori_kv=seq_ori,
        cmp_residual_kv=cmp_residual,
        batch_size=batch,
        max_seqlen_q=s1,
        max_seqlen_kv=ori_len,
        ori_topk=0,
        cmp_topk=0,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="TND",
        has_ori_kv=True,
        has_cmp_kv=True,
    )

    # Current arch22 tiling rejects seqused_q when layout_q is TND; cu_seqlens_q is used by the kernel.
    attn_out, softmax_lse = npu_sparse_flash_mla(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        cu_seqlens_q=cu_q,
        cu_seqlens_ori_kv=cu_ori,
        cu_seqlens_cmp_kv=cu_cmp,
        seqused_ori_kv=seq_ori,
        cmp_residual_kv=cmp_residual,
        sinks=sinks,
        metadata=metadata,
        softmax_scale=1.0 / math.sqrt(head_dim),
        cmp_ratio=cmp_ratio,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="TND",
        topk_value_mode=1,
        return_softmax_lse=False,
    )
    torch.npu.synchronize()

    finite = bool(torch.isfinite(attn_out.cpu()).all().item())
    result = {
        "device": device,
        "residual": residual,
        "metadata_shape": tuple(metadata.shape),
        "attn_out_shape": tuple(attn_out.shape),
        "softmax_lse_shape": tuple(softmax_lse.shape),
        "attn_out_finite": finite,
        "attn_out_abs_max": float(attn_out.abs().max().cpu().item()),
    }
    print(result)
    if not finite:
        raise RuntimeError("SparseFlashMla output contains non-finite values.")


if __name__ == "__main__":
    main()
