# Copyright (c) PyPTO Contributors.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for full text of the License.
# -----------------------------------------------------------------------------------------------------------

import ast
import hashlib
import logging
import math
import os
import numpy as np
import openpyxl
import torch
import torch_npu
import cann_ops_transformer
from datetime import datetime

D_TOTAL = 512
SCALE = 1.0 / math.sqrt(D_TOTAL)


def _parse_list(val):
    if val is None or val == "" or str(val).lower() == "no":
        return None
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except:
            return None
    return val


def _to_torch_dtype(dtype_str):
    if str(dtype_str).lower() == "bfloat16":
        return torch.bfloat16
    return torch.float16


def build_cumsum_offsets(a):
    cumsum_a = torch.cumsum(a, dim=0)
    c = torch.cat([torch.zeros(1, dtype=a.dtype, device=a.device), cumsum_a])
    return c.to(a.dtype)


def compute_metadata(seqused_q, seq_q_cpu, core_num):
    if seqused_q is not None:
        total_t1 = int(seqused_q.sum().item())
    else:
        total_t1 = int(seq_q_cpu.sum().item())
    
    if total_t1 <= core_num:
        return [total_t1, 1, 0, 0, total_t1]
    
    former = math.ceil(total_t1 / core_num)
    remain = former - 1
    former_num = total_t1 - core_num * remain
    return [total_t1, former, remain, core_num - former_num, core_num]


def generate_sparse_indices_bsnd(b, gen_s1, gen_s2, k, mask_mode, cmp_ratio, seq_q, seq_k, sparse_mode, has_topk, cmp_residual_k):
    sparse_indices = torch.zeros(b, gen_s1, 1, k, dtype=torch.int32)
    topk_length = torch.zeros(b, gen_s1, 1, dtype=torch.int32) if has_topk else None

    for b_id in range(b):
        cur_sq = seq_q[b_id].item() if seq_q is not None else gen_s1
        cur_skv = seq_k[b_id].item() if seq_k is not None else gen_s2
        for s1_idx in range(cur_sq):
            if mask_mode == 0:
                actual_sk = cur_skv
            else:
                cmp_val = cmp_residual_k[b_id].item() if cmp_residual_k is not None else 0
                if cmp_ratio > 1:
                    ori_sk = cur_skv * cmp_ratio + cmp_val
                else:
                    ori_sk = cur_skv * cmp_ratio
                actual_sk = max((ori_sk - cur_sq + s1_idx + 1) // cmp_ratio, 0)
            
            limit = min(actual_sk, k)

            if has_topk:
                if sparse_mode == "random":
                    if limit > 0:
                        topk_length[b_id, s1_idx, 0] = torch.randint(0, limit, (1,), dtype=torch.int32)
                    else:
                        topk_length[b_id, s1_idx, 0] = 0
                else:
                    topk_length[b_id, s1_idx, 0] = limit
                limit = int(topk_length[b_id, s1_idx, 0])

            if limit > 0 and actual_sk > 0:
                sparse_indices[b_id, s1_idx, 0, :limit] = torch.randint(0, actual_sk, (limit,), dtype=torch.int32)

    return sparse_indices, topk_length


def generate_sparse_indices_tnd(seq_q_cpu, seq_k_cpu, k, mask_mode, cmp_ratio, seqused_q, seqused_k, sparse_mode, has_topk, cmp_residual_k):
    B = seq_q_cpu.shape[0]
    T1 = seq_q_cpu.sum().item()
    cu_q = build_cumsum_offsets(seq_q_cpu).int()

    sparse_indices = torch.zeros(T1, 1, k, dtype=torch.int32)
    topk_length = torch.zeros(T1, 1, dtype=torch.int32) if has_topk else None

    for b_id in range(B):
        sq_len = seqused_q[b_id].item() if seqused_q is not None else seq_q_cpu[b_id].item()
        sk_len = seqused_k[b_id].item() if seqused_k is not None else seq_k_cpu[b_id].item()
        for s1_idx in range(sq_len):
            global_idx = cu_q[b_id].item() + s1_idx
            if mask_mode == 0:
                actual_sk = sk_len
            else:
                cmp_val = cmp_residual_k[b_id].item() if cmp_residual_k is not None else 0
                if cmp_ratio > 1:
                    ori_sk = sk_len * cmp_ratio + cmp_val
                else:
                    ori_sk = sk_len * cmp_ratio
                actual_sk = max((ori_sk - sq_len + s1_idx + 1) // cmp_ratio, 0)

            limit = min(actual_sk, k)

            if has_topk:
                if sparse_mode == "random":
                    if limit > 0:
                        topk_length[global_idx, 0] = torch.randint(0, limit, (1,), dtype=torch.int32).item()
                    else:
                        topk_length[global_idx, 0] = 0
                else:
                    topk_length[global_idx, 0] = limit
                limit = int(topk_length[global_idx, 0])

            if limit > 0 and actual_sk > 0:
                sparse_indices[global_idx, 0, :limit] = torch.randint(0, actual_sk, (limit,), dtype=torch.int32)

    return sparse_indices, topk_length

def tensor_md5(tensor):
    # 确保 float32, 分离, 移到 CPU, 转为 numpy, 再转为 bytes
    arr = tensor.float().detach().cpu().numpy()
    # 确保是连续内存
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    bytes_data = arr.tobytes()
    return hashlib.md5(bytes_data).hexdigest()

def md5_compare(input_md5, output):
    print()
    print("= " * 33 + output + " md5 compare " + "= " * 37)
    print("md5 of " + output + " are :", input_md5)
    if all(x == input_md5[0] for x in input_md5):
        print("= " * 33 + output + " md5 compare pass " + "= " * 33)
        print()
        return "pass", input_md5[0]
    else:
        print("= " * 33 + output + " md5 compare failed " + "= " * 33)
        print()
        return "failed", str(input_md5)

def compute_golden_bsnd(q, k, lse, seqused_q, seqused_k, cmp_residual_k, sparse_indices, topk_length, mask_mode, cmp_ratio, core_num, softmax_scale):
    B, Sq, G, _ = q.shape
    Sk = k.shape[1]
    device = q.device
    
    metadata = torch.zeros(1024, dtype=torch.int32, device=device)
    meta_vals = compute_metadata(seqused_q, torch.full((B,), Sq, dtype=torch.int32), core_num)
    metadata[:5] = torch.tensor(meta_vals, dtype=torch.int32)

    k_length = sparse_indices.shape[-1] if sparse_indices is not None else Sk
    out = torch.zeros(B, Sq, 1, k_length, dtype=torch.float32, device=device)

    for b_id in range(B):
        sq_len = seqused_q[b_id].item() if seqused_q is not None else Sq
        sk_len = seqused_k[b_id].item() if seqused_k is not None else Sk
        for s1_id in range(sq_len):
            query = q[b_id, s1_id, :, :].float()
            if mask_mode == 0:
                actual_sk = sk_len
            else:
                cmp_val = cmp_residual_k[b_id].item() if cmp_residual_k is not None else 0
                if cmp_ratio > 1:
                    ori_sk = sk_len * cmp_ratio + cmp_val
                else:
                    ori_sk = sk_len * cmp_ratio
                actual_sk = max((ori_sk - sq_len + s1_id + 1) // cmp_ratio, 0)

            if sparse_indices is not None:
                K = sparse_indices.shape[-1]
                if topk_length is not None:
                    K = min(int(topk_length[b_id, s1_id, 0]), K)
                actual_sk = min(actual_sk, K)
                if actual_sk > 0:
                    key = k[b_id, sparse_indices[b_id, s1_id, 0, :actual_sk], 0, :].float()
                else:
                    key = k[b_id, :0, 0, :].float()
            else:
                key = k[b_id, :actual_sk, 0, :].float()

            qk = query @ key.T
            lse_vec = lse[b_id, 0, s1_id, :]
            p = torch.exp(qk * softmax_scale - lse_vec.unsqueeze(1))
            if actual_sk > 0:
                out[b_id, s1_id, 0, :actual_sk] = (p.sum(dim=0) / G)[:actual_sk]

    return out, metadata


def compute_golden_tnd(q, k, lse, seq_q_cpu, seq_k_cpu, seqused_q, seqused_k, cmp_residual_k, sparse_indices, topk_length, mask_mode, cmp_ratio, core_num, softmax_scale):
    B = seq_q_cpu.shape[0]
    G = q.shape[1]
    device = q.device
    cu_q = build_cumsum_offsets(seq_q_cpu).int()
    cu_k = build_cumsum_offsets(seq_k_cpu).int()
    T1 = seq_q_cpu.sum().item()
    K = sparse_indices.shape[-1] if sparse_indices is not None else seq_k_cpu.sum().item()
    out = torch.zeros(T1, 1, K, dtype=torch.float32, device=device)

    metadata = torch.zeros(1024, dtype=torch.int32, device=device)
    meta_vals = compute_metadata(seqused_q, seq_q_cpu, core_num)
    metadata[:5] = torch.tensor(meta_vals, dtype=torch.int32)

    for b_id in range(B):
        sq_len = seqused_q[b_id].item() if seqused_q is not None else seq_q_cpu[b_id].item()
        sk_len = seqused_k[b_id].item() if seqused_k is not None else seq_k_cpu[b_id].item()
        q_start = cu_q[b_id].item()
        k_start = cu_k[b_id].item()

        for s1_idx in range(sq_len):
            query = q[q_start + s1_idx].float()
            if mask_mode == 0:
                actual_sk = sk_len
            else:
                cmp_val = cmp_residual_k[b_id].item() if cmp_residual_k is not None else 0
                if cmp_ratio > 1:
                    ori_sk = sk_len * cmp_ratio + cmp_val
                else:
                    ori_sk = sk_len * cmp_ratio
                actual_sk = max((ori_sk - sq_len + s1_idx + 1) // cmp_ratio, 0)

            global_idx = q_start + s1_idx
            if sparse_indices is not None:
                KK = sparse_indices.shape[-1]
                if topk_length is not None:
                    KK = min(int(topk_length[global_idx, 0].item()), KK)
                actual_sk = min(actual_sk, KK)
                if actual_sk > 0:
                    indices = sparse_indices[global_idx, 0, :actual_sk].long()
                    key = k[k_start + indices, 0, :].float()
                else:
                    key = k[k_start:k_start, 0, :].float()
            else:
                key = k[k_start:k_start + actual_sk, 0, :].float()

            qk = query @ key.T
            lse_vec = lse[0, global_idx, :]
            p = torch.exp(qk * softmax_scale - lse_vec.unsqueeze(1))
            if actual_sk > 0:
                out[global_idx, 0, :actual_sk] = (p.sum(dim=0) / G)[:actual_sk]

    return out, metadata


def find_mismatched_elements(actual, expected, rtol=5e-3, atol=5e-3):
    actual_flat = actual.cpu().flatten().to(torch.float32)
    expected_flat = expected.cpu().flatten().to(torch.float32)
    abs_diff = torch.abs(actual_flat - expected_flat)
    tolerance = atol + rtol * torch.abs(expected_flat)
    mismatch_mask = abs_diff > tolerance
    nan_mask = torch.isnan(actual_flat) | torch.isnan(expected_flat)
    inf_mask = torch.isinf(actual_flat) | torch.isinf(expected_flat)
    mismatch_mask = mismatch_mask | nan_mask | inf_mask
    mismatch_indices = torch.nonzero(mismatch_mask, as_tuple=False).flatten()
    total = actual_flat.numel()
    mismatch_count = mismatch_indices.numel()
    print(f"总元素数: {total}")
    print(f"异常点数: {mismatch_count} ({mismatch_count / total * 100:.4f}%)")
    print("-" * 80)
    if mismatch_count > 0:
        print(f"{'Index':<15} {'Actual':<20} {'Expected':<20} {'AbsDiff':<20} {'Tolerance':<20}")
        print("-" * 80)
        print_count = 50
        for idx in mismatch_indices:
            i = idx.item()
            a = actual_flat[i].item()
            e = expected_flat[i].item()
            d = abs_diff[i].item()
            t = tolerance[i].item()
            print(f"{i:<15} {a:<20.8f} {e:<20.8f} {d:<20.8f} {t:<20.8f}")
            print_count -= 1
            if print_count <= 0:
                break
        print("-" * 80)
    return mismatch_count == 0


def run_test_case(params):
    device = "npu:0"
    torch.npu.set_device(device)
    torch.manual_seed(int(params.get("manual_seed", 42)))

    B = int(params["b"])
    S1 = int(params["s1"])
    S2 = int(params["s2"])
    G = int(params["n1"])
    D = int(params["d"])
    dtype = _to_torch_dtype(params.get("dtype", "float16"))
    layout = params.get("layout_q", "BSND")
    mask_mode = int(params.get("mask_mode", 0))
    cmp_ratio = int(params.get("cmp_ratio", 1))
    softmax_scale = float(params.get("softmax_scale", SCALE))
    core_num = 32

    K = params.get("sparse_size")
    if K is not None: K = int(K)
    is_sparse = K is not None and K > 0

    sparse_mode = params.get("sparse_mode", "")
    topk_mode = params.get("topk_mode", "")
    has_topk = str(topk_mode).lower() != "no" and str(topk_mode) != ""

    seq_q_list = _parse_list(params.get("seqlens_list_q"))
    seq_k_list = _parse_list(params.get("seqlens_list_kv"))
    seqused_q_list = _parse_list(params.get("seqused_q"))
    seqused_k_list = _parse_list(params.get("seqused_kv"))
    cmp_residual_k_list = _parse_list(params.get("cmp_residual_kv"))

    logging.info(f"\nRunning test: {params.get('testcase_name', 'N/A')} (layout={layout}, sparse={is_sparse}, topk={has_topk})")

    if layout == "TND":
        if seq_q_list is None or len(seq_q_list) < B:
            seq_q_list = [S1 // B] * B
            seq_q_list[-1] += S1 - sum(seq_q_list)
        if seq_k_list is None or len(seq_k_list) < B:
            seq_k_list = [S2 // B] * B
            seq_k_list[-1] += S2 - sum(seq_k_list)

        seq_q_cpu = torch.tensor(seq_q_list[:B], dtype=torch.int32)
        seq_k_cpu = torch.tensor(seq_k_list[:B], dtype=torch.int32)
        T1 = seq_q_cpu.sum().item()
        T2 = seq_k_cpu.sum().item()

        seqused_q_cpu = torch.tensor(seqused_q_list[:B], dtype=torch.int32) if seqused_q_list else None
        seqused_k_cpu = torch.tensor(seqused_k_list[:B], dtype=torch.int32) if seqused_k_list else None
        cmp_residual_k_cpu = torch.tensor(cmp_residual_k_list[:B], dtype=torch.int32) if cmp_residual_k_list else None

        query = torch.rand((T1, G, D), dtype=dtype)
        key = torch.rand((T2, 1, D), dtype=dtype)
        lse = torch.rand((1, T1, G), dtype=torch.float32)

        if is_sparse:
            sparse_indices, topk_length = generate_sparse_indices_tnd(seq_q_cpu, seq_k_cpu, K, mask_mode, cmp_ratio,
                                                                      seqused_q_cpu, seqused_k_cpu, sparse_mode, has_topk, cmp_residual_k_cpu)
        else:
            sparse_indices, topk_length = None, None

        golden_out, metadata = compute_golden_tnd(query, key, lse, seq_q_cpu, seq_k_cpu,
                                                   seqused_q_cpu, seqused_k_cpu, cmp_residual_k_cpu,
                                                   sparse_indices, topk_length,
                                                   mask_mode, cmp_ratio, core_num, softmax_scale)

        query_dev = query.to(device)
        key_dev = key.to(device)
        lse_dev = lse.to(device)
        sparse_indices_dev = sparse_indices.to(device) if sparse_indices is not None else None
        topk_length_dev = topk_length.to(device) if topk_length is not None else None
        seqused_q_dev = seqused_q_cpu.to(device) if seqused_q_cpu is not None else None
        seqused_k_dev = seqused_k_cpu.to(device) if seqused_k_cpu is not None else None
        cmp_residual_k_dev = cmp_residual_k_cpu.to(device) if cmp_residual_k_cpu is not None else None
        metadata_dev = metadata.to(device)
        seq_q_dev = build_cumsum_offsets(seq_q_cpu).to(device)
        seq_k_dev = build_cumsum_offsets(seq_k_cpu).to(device)

        if topk_length_dev is not None:
            print(f"topk_length_dev shape: {topk_length_dev.shape}")
        input_md5 = []
        for i in range(10):
            npu_out = cann_ops_transformer.sparse_flash_mla_softmax_l1_norm(
                query_dev, key_dev, lse_dev,
                sparse_indices=sparse_indices_dev,
                cu_seqlens_q=seq_q_dev,
                cu_seqlens_k=seq_k_dev,
                seqused_q=seqused_q_dev,
                seqused_k=seqused_k_dev,
                cmp_residual_k=cmp_residual_k_dev,
                topk_length=topk_length_dev,
                metadata=metadata_dev,
                scale_value=softmax_scale,
                max_seqlen_k=S2,
                cmp_ratio=cmp_ratio,
                mask_mode=mask_mode,
                layout_q="TND",
                layout_k="TND",
            )
            torch.npu.synchronize()
            md5_sum_v = tensor_md5(npu_out.cpu().flatten())
            input_md5.append(md5_sum_v)
        md5_status, md5_val = md5_compare(input_md5, params.get('testcase_name', 'N/A'))
        golden_out_final = golden_out

    else:
        gen_s1 = S1
        gen_s2 = S2
        seqused_q_cpu = torch.tensor(seqused_q_list[:B], dtype=torch.int32) if seqused_q_list else None
        seqused_k_cpu = torch.tensor(seqused_k_list[:B], dtype=torch.int32) if seqused_k_list else None
        cmp_residual_k_cpu = torch.tensor(cmp_residual_k_list[:B], dtype=torch.int32) if cmp_residual_k_list else None

        query = torch.rand((B, gen_s1, G, D), dtype=dtype)
        key = torch.rand((B, gen_s2, 1, D), dtype=dtype)
        lse = torch.rand((B, 1, gen_s1, G), dtype=torch.float32)

        if is_sparse:
            sparse_indices, topk_length = generate_sparse_indices_bsnd(B, gen_s1, gen_s2, K, mask_mode, cmp_ratio,
                                                                       seqused_q_cpu if seqused_q_cpu is not None else torch.full((B,), gen_s1, dtype=torch.int32),
                                                                       seqused_k_cpu if seqused_k_cpu is not None else torch.full((B,), gen_s2, dtype=torch.int32),
                                                                       sparse_mode, has_topk, cmp_residual_k_cpu)
        else:
            sparse_indices, topk_length = None, None

        golden_out, metadata = compute_golden_bsnd(query, key, lse, seqused_q_cpu, seqused_k_cpu, cmp_residual_k_cpu,
                                                    sparse_indices, topk_length, mask_mode, cmp_ratio, core_num, softmax_scale)

        query_dev = query.to(device)
        key_dev = key.to(device)
        lse_dev = lse.to(device)
        sparse_indices_dev = sparse_indices.to(device) if sparse_indices is not None else None
        topk_length_dev = topk_length.to(device) if topk_length is not None else None
        seqused_q_dev = seqused_q_cpu.to(device) if seqused_q_cpu is not None else None
        seqused_k_dev = seqused_k_cpu.to(device) if seqused_k_cpu is not None else None
        cmp_residual_k_dev = cmp_residual_k_cpu.to(device) if cmp_residual_k_cpu is not None else None
        metadata_dev = metadata.to(device)

        input_md5 = []
        for i in range(10):
            npu_out = cann_ops_transformer.sparse_flash_mla_softmax_l1_norm(
                query_dev, key_dev, lse_dev,
                sparse_indices=sparse_indices_dev,
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                seqused_q=seqused_q_dev,
                seqused_k=seqused_k_dev,
                cmp_residual_k=cmp_residual_k_dev,
                topk_length=topk_length_dev,
                metadata=metadata_dev,
                scale_value=softmax_scale,
                max_seqlen_k=gen_s2 if not is_sparse else 0,
                cmp_ratio=cmp_ratio,
                mask_mode=mask_mode,
                layout_q="BSND",
                layout_k="BSND",
            )
            torch.npu.synchronize()
            md5_sum_v = tensor_md5(npu_out.cpu().flatten())
            input_md5.append(md5_sum_v)
        md5_status, md5_val = md5_compare(input_md5, params.get('testcase_name', 'N/A'))
        golden_out_final = golden_out

    torch.npu.synchronize()

    diff = (npu_out.cpu().flatten().to(torch.float32) - golden_out_final.cpu().flatten().to(torch.float32)).abs().max().item()
    logging.info(f"Max diff: {diff}")

    is_match = find_mismatched_elements(npu_out, golden_out_final, rtol=5e-3, atol=5e-3)
    result = "pass" if is_match else "failed"
    if is_match:
        logging.info(f"casename: {params.get('testcase_name', 'N/A')}, 精度校验通过，所有点均在容差范围内。")
    else:
        logging.info(f"casename: {params.get('testcase_name', 'N/A')}, 精度校验失败，存在异常点。")

    logging.info(f"  PASS: {params.get('testcase_name', 'N/A')}")

    return {"result": result, "max_diff": diff, "md5_result": md5_status}


def load_excel_params(filepath):
    wb = openpyxl.load_workbook(filepath)
    ws = wb["Sheet1"]
    headers = [cell.value for cell in ws[1]]
    test_cases = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None:
            continue
        params = {}
        for i, h in enumerate(headers):
            if h is None:
                continue
            h_clean = h.strip().lower().replace(" ", "_")
            val = row[i]
            if h_clean in ["seqlens_list_q", "seqlens_list_kv", "seqused_q", "seqused_kv", "cmp_residual_kv"]:
                params[h_clean] = _parse_list(val)
            else:
                params[h_clean] = val
        test_cases.append(params)
    return test_cases


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("SparseFlashMlaSoftmaxL1Norm Merged Tests V2")

    excel_path = r"/npu/d00483431/code/ops-transformer/torch_extension/sparse_flash_mla_softmax_l1_norm_fuzz_gate.xlsx"
    test_cases = load_excel_params(excel_path)

    results = []
    for idx, case in enumerate(test_cases):
        if not case.get('enable'):
            # logging.info(f"  SKIPPED: {case.get('testcase_name', 'N/A')} (enable=False)")
            results.append({"result": "skipped", "max_diff": "", "md5_result": ""})
            continue
        try:
            logging.info("%s", '=' * 60)
            logging.info(f"Test Case {idx+1}: {case.get('testcase_name', 'N/A')}")
            res = run_test_case(case)
            results.append(res)
        except Exception as e:
            logging.error(f"Test Case {idx+1} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append({"result": "error", "max_diff": "", "md5_result": str(e)})

    logging.info("\nAll tests completed!")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_name = os.path.splitext(os.path.basename(excel_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{base_name}_Result_{timestamp}.xlsx"
    output_path = os.path.join(script_dir, output_filename)

    wb = openpyxl.load_workbook(excel_path)
    ws = wb["Sheet1"]

    result_col_idx = ws.max_column + 1
    max_diff_col_idx = result_col_idx + 1
    md5_result_col_idx = result_col_idx + 1

    ws.cell(row=1, column=result_col_idx, value="result")
    ws.cell(row=1, column=max_diff_col_idx, value="max_diff")
    ws.cell(row=1, column=md5_result_col_idx, value="md5_result")

    for i, res in enumerate(results):
        row = i + 2
        ws.cell(row=row, column=result_col_idx, value=res["result"])
        ws.cell(row=row, column=max_diff_col_idx, value=res["max_diff"])
        ws.cell(row=row, column=md5_result_col_idx, value=res["md5_result"])

    wb.save(output_path)
    logging.info(f"Results saved to: {output_path}")
