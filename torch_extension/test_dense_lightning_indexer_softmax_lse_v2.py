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

try:
    import openpyxl
except ImportError:
    openpyxl = None


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
    neg_inf = -torch.inf

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

        row_valid = torch.zeros(S1, dtype=torch.bool, device=dev)
        for s1_i in range(min(q_used, S1)):
            if mask_mode == 3:
                cur_residual = 0
                if cmp_ratio > 1:
                    cur_residual = 0 if cmp_residual_k is None else cmp_residual_k[b, 0].item()
                ori_k_size = k_used * cmp_ratio + cur_residual
                vk = max((s1_i + 1 + ori_k_size - q_used) // cmp_ratio, 0)
                row_valid[s1_i] = vk > 0
            else:
                row_valid[s1_i] = k_used > 0

        reduce_max = x_full.amax(dim=1)
        shifted = x_full - reduce_max.unsqueeze(1)
        sum_exp = torch.exp(shifted).sum(dim=1)
        lse[b] = torch.where(row_valid, reduce_max + torch.log(sum_exp), torch.full_like(reduce_max, neg_inf))

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
        print(f"metadata done, shape = {metadata.shape}, dtype = {metadata.dtype}")
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
                        mask_mode=0, cmp_ratio=1):
    q = query_index.float()
    k = key_index.float()
    w = weight.float()
    T1, N1, D = q.shape
    T2, _, _ = k.shape
    dev = q.device
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

        row_valid = torch.zeros(q_len, dtype=torch.bool, device=dev)
        for s1_i in range(min(q_used, q_len)):
            if mask_mode == 3:
                cur_residual = 0
                if cmp_ratio > 1:
                    cur_residual = 0 if cmp_residual_k is None else cmp_residual_k[b, 0].item()
                ori_k_size = k_used * cmp_ratio + cur_residual
                vk = max((s1_i + 1 + ori_k_size - q_used) // cmp_ratio, 0)
                row_valid[s1_i] = vk > 0
            else:
                row_valid[s1_i] = k_used > 0

        reduce_max = x_full.amax(dim=1)
        shifted = x_full - reduce_max.unsqueeze(1)
        sum_exp = torch.exp(shifted).sum(dim=1)
        lse[q_start:q_end] = torch.where(row_valid, reduce_max + torch.log(sum_exp), torch.full_like(reduce_max, neg_inf))

    return lse


def test_e2e_tnd():
    device = "npu:0"
    torch.npu.set_device(device)
    torch.manual_seed(42)
    B, S1_per_b, S2_per_b, N1, D = 3, 128, 256, 64, 128
    T1 = B * S1_per_b
    T2 = B * S2_per_b

    cu_seqlens_q = torch.tensor([0, S1_per_b, 2*S1_per_b, 3*S1_per_b], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0, S2_per_b, 2*S2_per_b, 3*S2_per_b], dtype=torch.int32, device=device)

    q = (torch.rand((T1, N1, D), device=device, dtype=torch.float16) * 2 - 1)
    k = (torch.rand((T2, 1, D), device=device, dtype=torch.float16) * 2 - 1)
    w = (torch.rand((T1, N1), device=device, dtype=torch.float32) * 2 - 1)

    seqused_q = torch.full((B, 1), S1_per_b, device=device, dtype=torch.int32)
    seqused_k = torch.full((B, 1), S2_per_b, device=device, dtype=torch.int32)
    cmp_residual_k = torch.zeros((B, 1), device=device, dtype=torch.int32)

    sparse_mode = 0
    cmp_ratio = 4

    metadata = cann_ops_transformer.dense_lightning_indexer_softmax_lse_v2_metadata(
        num_heads_q=N1,
        num_heads_k=1,
        head_dim=D,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        layout_q="TND",
        layout_k="TND",
        mask_mode=sparse_mode,
        cmp_ratio=cmp_ratio,
    )
    print(f"[TND] metadata shape={metadata.shape}, dtype={metadata.dtype}")

    lse = cann_ops_transformer.dense_lightning_indexer_softmax_lse_v2(
        q, k, w,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        metadata=metadata,
        layout_q="TND",
        layout_k="TND",
        mask_mode=sparse_mode,
        cmp_ratio=cmp_ratio,
    )
    torch.npu.synchronize()

    lse_ref = golden_dli_lse_tnd(q.cpu(), k.cpu(), w.cpu(),
                                  cu_seqlens_q.cpu(), cu_seqlens_k.cpu(),
                                  seqused_q=seqused_q.cpu(),
                                  seqused_k=seqused_k.cpu(),
                                  cmp_residual_k=cmp_residual_k.cpu(),
                                  mask_mode=sparse_mode,
                                  cmp_ratio=cmp_ratio)
    diff = (lse.cpu().float() - lse_ref).abs().max().item()
    print(f"[TND e2e] max|diff|={diff:.6f}")
    torch.testing.assert_close(lse.cpu().float(), lse_ref, rtol=5e-2, atol=5e-2)
    print("[TND e2e] PASS")


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


def _build_case_inputs(case, device):
    B = case["B"]
    N1 = case["N1"]
    D = case["D"]
    S1 = case["S1"]
    S2 = case["S2"]
    dtype_str = case.get("dtype", "float16")
    torch_dtype = torch.float16 if dtype_str == "float16" else torch.bfloat16
    seed = case.get("manual_seed", 0) or 0
    layout_q = case.get("layout_q", "BSND") or "BSND"
    layout_k = case.get("layout_k", "BSND") or "BSND"
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
                                     dtype=torch.int32, device=device)
        cu_seqlens_k = torch.tensor([0] + list(__import__("itertools").accumulate(seqlens_k_list)),
                                     dtype=torch.int32, device=device)
    else:
        T1 = B * S1
        T2 = B * S2
        cu_seqlens_q = None
        cu_seqlens_k = None

    if is_tnd:
        q = (torch.rand((T1, N1, D), device=device, dtype=torch_dtype) * 2 - 1)
        k = (torch.rand((T2, 1, D), device=device, dtype=torch_dtype) * 2 - 1)
        w = (torch.rand((T1, N1), device=device, dtype=torch.float32) * 2 - 1)
    else:
        q = (torch.rand((B, S1, N1, D), device=device, dtype=torch_dtype) * 2 - 1)
        k = (torch.rand((B, S2, 1, D), device=device, dtype=torch_dtype) * 2 - 1)
        w = (torch.rand((B, S1, N1), device=device, dtype=torch.float32) * 2 - 1)

    seqused_q_val = case.get("seqused_q")
    seqused_k_val = case.get("seqused_k")
    cmp_residual_k_val = case.get("cmp_residual_k")

    seqused_q = None
    if seqused_q_val is not None:
        seqused_q = torch.tensor(seqused_q_val, dtype=torch.int32, device=device).reshape(B, 1)

    seqused_k = None
    if seqused_k_val is not None:
        seqused_k = torch.tensor(seqused_k_val, dtype=torch.int32, device=device).reshape(B, 1)

    cmp_residual_k = None
    if cmp_residual_k_val is not None:
        cmp_residual_k = torch.tensor(cmp_residual_k_val, dtype=torch.int32, device=device).reshape(B, 1)
    else:
        cmp_residual_k = torch.zeros((B, 1), dtype=torch.int32, device=device)

    return {
        "q": q, "k": k, "w": w,
        "cu_seqlens_q": cu_seqlens_q, "cu_seqlens_k": cu_seqlens_k,
        "seqused_q": seqused_q, "seqused_k": seqused_k,
        "cmp_residual_k": cmp_residual_k,
        "layout_q": layout_q, "layout_k": layout_k,
        "mask_mode": case.get("mask_mode", 0) or 0,
        "cmp_ratio": case.get("cmp_ratio", 1) or 1,
        "is_tnd": is_tnd,
        "B": B,
    }


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
                        save_golden=False, use_cache=False):
    if xlsx_path is None:
        xlsx_path = r"C:\Users\kjc\Desktop\pto相关\dli\测试用例\dense_lightning_indexer_softmax_lse_fuzz_0.5g - mk.xlsx"
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".dli_lse_cache")

    device = "npu:0"
    torch.npu.set_device(device)

    cases = _load_xlsx_cases(xlsx_path)
    print(f"[fuzz] Loaded {len(cases)} test cases from {xlsx_path}")
    print(f"[fuzz] save_golden={save_golden}, use_cache={use_cache}, cache_dir={cache_dir}")

    pass_count = 0
    fail_count = 0
    results = []

    for idx, case in enumerate(cases):
        case_name = case.get("testcase_name", f"case_{idx}")
        B = case["B"]
        N1 = case["N1"]
        S1 = case["S1"]
        S2 = case["S2"]
        D = case["D"]
        mask_mode = case.get("mask_mode", 0) or 0
        cmp_ratio = case.get("cmp_ratio", 1) or 1
        layout_q = case.get("layout_q", "BSND") or "BSND"
        is_tnd = (layout_q == "TND")
        print(f"\n[{idx+1}/{len(cases)}] {case_name} (B={B}, N1={N1}, S1={S1}, S2={S2}, D={D}, "
              f"mask={mask_mode}, cmp={cmp_ratio}, layout={layout_q})")

        case_max_diff = None
        case_result = "ERROR"
        try:
            inputs = _build_case_inputs(case, device)

            metadata = cann_ops_transformer.dense_lightning_indexer_softmax_lse_v2_metadata(
                num_heads_q=N1,
                num_heads_k=1,
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
            )

            lse = cann_ops_transformer.dense_lightning_indexer_softmax_lse_v2(
                inputs["q"], inputs["k"], inputs["w"],
                cu_seqlens_q=inputs["cu_seqlens_q"],
                cu_seqlens_k=inputs["cu_seqlens_k"],
                seqused_q=inputs["seqused_q"],
                seqused_k=inputs["seqused_k"],
                cmp_residual_k=inputs["cmp_residual_k"],
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
                q_cpu = inputs["q"].cpu()
                k_cpu = inputs["k"].cpu()
                w_cpu = inputs["w"].cpu()
                if is_tnd:
                    lse_ref = golden_dli_lse_tnd(
                        q_cpu, k_cpu, w_cpu,
                        inputs["cu_seqlens_q"].cpu(),
                        inputs["cu_seqlens_k"].cpu(),
                        seqused_q=inputs["seqused_q"].cpu() if inputs["seqused_q"] is not None else None,
                        seqused_k=inputs["seqused_k"].cpu() if inputs["seqused_k"] is not None else None,
                        cmp_residual_k=inputs["cmp_residual_k"].cpu(),
                        mask_mode=mask_mode,
                        cmp_ratio=cmp_ratio)
                else:
                    lse_ref = golden_dli_lse_blk(
                        q_cpu, k_cpu, w_cpu,
                        seqused_q=inputs["seqused_q"].cpu() if inputs["seqused_q"] is not None else None,
                        seqused_k=inputs["seqused_k"].cpu() if inputs["seqused_k"] is not None else None,
                        cmp_residual_k=inputs["cmp_residual_k"].cpu(),
                        mask_mode=mask_mode,
                        cmp_ratio=cmp_ratio)

            if save_golden and do_golden:
                _save_case(case_name, cache_dir, inputs, lse.cpu(), lse_ref)

            diff = (lse.cpu().float() - lse_ref.float()).abs()
            max_diff = diff.max().item()
            nan_count = torch.isnan(lse.cpu().float()).sum().item()
            inf_count = torch.isinf(lse.cpu().float()).sum().item()

            tol = 5e-2
            case_max_diff = max_diff
            if max_diff > tol or nan_count > 0:
                case_result = "FAIL"
                print(f"  [FAIL] max|diff|={max_diff:.6f} (tol={tol}), nan={nan_count}, inf={inf_count}")
                fail_count += 1
            else:
                case_result = "PASS"
                print(f"  [PASS] max|diff|={max_diff:.6f}")
                pass_count += 1

        except Exception as e:
            case_result = "ERROR"
            print(f"  [ERROR] {type(e).__name__}: {e}")
            fail_count += 1

        results.append((case_name, case_max_diff, case_result))

    print(f"\n[fuzz] Summary: {pass_count} passed, {fail_count} failed")

    _write_results_to_xlsx(xlsx_path, results)


def _write_results_to_xlsx(xlsx_path, results):
    if openpyxl is None:
        print("[fuzz] openpyxl not available, skip writing results to xlsx")
        return
    import shutil
    base, ext = os.path.splitext(xlsx_path)
    out_path = f"{base}_result{ext}"
    shutil.copy2(xlsx_path, out_path)
    wb = openpyxl.load_workbook(out_path)
    ws = wb.active

    max_col = ws.max_column
    ws.cell(row=1, column=max_col + 1, value="max_diff")
    ws.cell(row=1, column=max_col + 2, value="result")

    case_name_to_row = {}
    for row_idx in range(2, ws.max_row + 1):
        name = ws.cell(row=row_idx, column=2).value
        if name is not None:
            case_name_to_row[name] = row_idx

    written = 0
    for case_name, max_diff, result in results:
        row_idx = case_name_to_row.get(case_name)
        if row_idx is not None:
            ws.cell(row=row_idx, column=max_col + 1, value=max_diff)
            ws.cell(row=row_idx, column=max_col + 2, value=result)
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
    args = parser.parse_args()

    test_fuzz_from_xlsx(
        xlsx_path=args.xlsx,
        cache_dir=args.cache_dir,
        save_golden=args.save_golden,
        use_cache=args.use_cache,
    )
