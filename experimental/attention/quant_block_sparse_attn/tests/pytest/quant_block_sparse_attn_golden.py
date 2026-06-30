#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import math
import os
import random

import torch

import check_valid_param
import combined_kv_cache


DATA_RANGE_LEFT = -1.0
DATA_RANGE_RIGHT = 1.0
EMPTY_LSE = -3.4028234663852886e38
MASK_VALUE = -10000.0


def _fp8_dtype():
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("torch.float8_e4m3fn is required for quant_block_sparse_attn FP8 test data")
    return torch.float8_e4m3fn


def _torch_dtype(value):
    if value in (torch.bfloat16, "torch.bfloat16", "bfloat16", "bf16"):
        return torch.bfloat16
    return value


def _normalize_params(params):
    if not isinstance(params, dict):
        raise ValueError("quant_block_sparse_attn params should be a dict")

    normalized = dict(params)
    normalized.setdefault("Testcase_Name", None)
    normalized.setdefault("layout_q", "BSND")
    normalized.setdefault("layout_kv", "PA_BNSD")
    normalized.setdefault("output_dtype", torch.bfloat16)
    normalized.setdefault("sparse_q_block_size", 128)
    normalized.setdefault("sparse_kv_block_size", 128)
    normalized.setdefault("layout_sparse_indices", "B_N_Qb_Kb")
    normalized.setdefault("layout_out", "TND")
    normalized.setdefault("quant_mode", 1)
    normalized.setdefault("mask_mode", 3)
    normalized.setdefault("return_softmax_lse", False)
    normalized.setdefault("actlen_mode", "full")
    normalized.setdefault("S1EQS2", False)
    normalized.setdefault("seed", 0)
    normalized.setdefault("sparse_pattern", "sequential")
    normalized.setdefault("scale_pattern", "ones")
    normalized.setdefault("block_table_pattern", "sequential")
    normalized.setdefault("pa_block_padding_bytes", 0)
    normalized.setdefault("p_scale_value", 1.0)
    normalized["output_dtype"] = _torch_dtype(normalized["output_dtype"])
    for key in (
        "B", "S1", "S2", "N1", "N2", "D", "sparse_q_block_size", "sparse_kv_block_size",
        "sparse_count", "seed", "pa_block_padding_bytes"
    ):
        if isinstance(normalized.get(key), float) and normalized[key].is_integer():
            normalized[key] = int(normalized[key])
    normalized["sparse_count"] = normalized.get("sparse_count") or math.ceil(
        normalized["S2"] / normalized["sparse_kv_block_size"])
    normalized["softmax_scale"] = normalized.get("softmax_scale") or (1.0 / math.sqrt(normalized["D"]))
    return normalized


def _make_lengths(batch, max_len, mode, rng):
    if mode == "full":
        return [max_len] * batch, max_len
    if mode != "random":
        raise ValueError(f"unsupported actlen_mode: {mode}")
    low = max(1, max_len // 2)
    res_lens = [rng.randint(low, max_len) for _ in range(batch)]
    return res_lens, max(res_lens)


def _prefix(lengths):
    values = [0]
    for length in lengths:
        values.append(values[-1] + int(length))
    return torch.tensor(values, dtype=torch.int32)


def _rand_float(shape, generator):
    return torch.empty(shape, dtype=torch.float32).uniform_(DATA_RANGE_LEFT, DATA_RANGE_RIGHT, generator=generator)


def _rand_fp8(shape, generator):
    return _rand_float(shape, generator).to(_fp8_dtype())


def _physical_ids(start, count, pattern, rng):
    ids = list(range(start, start + count))
    if pattern == "reverse":
        ids.reverse()
    elif pattern == "random":
        rng.shuffle(ids)
    elif pattern != "sequential":
        raise ValueError(f"unsupported block_table_pattern: {pattern}")
    return ids


def _make_block_table(batch, seq_len, block_size, pattern, rng):
    block_num_per_batch = math.ceil(seq_len / block_size)
    block_table = torch.empty((batch, block_num_per_batch), dtype=torch.int32)
    for batch_idx in range(batch):
        physical_ids = _physical_ids(batch_idx * block_num_per_batch, block_num_per_batch, pattern, rng)
        for block_idx, physical_block in enumerate(physical_ids):
            block_table[batch_idx, block_idx] = int(physical_block)
    return block_table


def _allowed_blocks(mask_mode, qb_idx, sparse_q_block_size, sparse_kv_block_size, q_len, kv_len):
    block_num = math.ceil(kv_len / sparse_kv_block_size)
    if block_num <= 0:
        return []
    if mask_mode == 0:
        return list(range(block_num))
    if mask_mode != 3:
        raise ValueError(f"unsupported mask_mode: {mask_mode}")

    max_token = (qb_idx + 1) * sparse_q_block_size - 1
    max_token += kv_len - q_len
    if max_token < 0:
        return []
    max_block = min(block_num - 1, max_token // sparse_kv_block_size)
    return list(range(max_block + 1))


def _select_blocks(blocks, sparse_count, pattern, rng):
    if sparse_count <= 0 or not blocks or pattern == "empty":
        return []
    if pattern in ("sequential", "dense", "causal"):
        return blocks[:min(sparse_count, len(blocks))]
    if pattern == "reverse":
        return list(reversed(blocks[-min(sparse_count, len(blocks)):]))
    if pattern == "tail":
        selected = blocks[:max(0, min(sparse_count, len(blocks)) - 1)]
        if blocks[-1] not in selected:
            selected.append(blocks[-1])
        return selected[:sparse_count]
    if pattern == "random":
        selected = blocks[:]
        rng.shuffle(selected)
        return selected[:min(sparse_count, len(selected))]
    if pattern == "empty_tail":
        return blocks[:min(sparse_count, len(blocks))]
    raise ValueError(f"unsupported sparse_pattern: {pattern}")


def _make_sparse_indices(case, q_lengths, kv_lengths, rng):
    batch = case["B"]
    n1 = case["N1"]
    n2 = case["N2"]
    group = n1 // n2
    qb_max = math.ceil(case["S1"] / case["sparse_q_block_size"])
    kv_max = math.ceil(case["S2"] / case["sparse_kv_block_size"])
    sparse_count = case["sparse_count"]
    sparse_indices = torch.full((batch, n1, qb_max, kv_max), fill_value=-1, dtype=torch.int32)
    sparse_seq_len = torch.zeros((batch, n1, qb_max), dtype=torch.int32)

    for batch_idx in range(batch):
        qb_batch = math.ceil(q_lengths[batch_idx] / case["sparse_q_block_size"])
        for qb_idx in range(qb_max):
            allowed = _allowed_blocks(
                case["mask_mode"], qb_idx, case["sparse_q_block_size"], case["sparse_kv_block_size"],
                q_lengths[batch_idx], kv_lengths[batch_idx])
            for n2_idx in range(n2):
                for group_idx in range(group):
                    head_idx = n2_idx * group + group_idx
                    if (case["sparse_pattern"] == "empty_tail" and qb_idx == qb_batch - 1) or qb_idx >= qb_batch:
                        selected = []
                    else:
                        selected = _select_blocks(allowed, sparse_count, case["sparse_pattern"], rng)
                    sparse_seq_len[batch_idx, head_idx, qb_idx] = len(selected)
                    if selected:
                        sparse_indices[batch_idx, head_idx, qb_idx, :len(selected)] = torch.tensor(
                            selected, dtype=torch.int32)
    return sparse_indices, sparse_seq_len


def _query_index(layout_q, cu_seqlens_q, batch_idx, q_idx):
    if layout_q == "BSND":
        return batch_idx, q_idx
    return int(cu_seqlens_q[batch_idx].item()) + q_idx


def _kv_index(layout_q, cu_seqlens_kv, batch_idx, kv_idx):
    if layout_q == "BSND":
        return batch_idx, kv_idx
    return int(cu_seqlens_kv[batch_idx].item()) + kv_idx


def _get_query(query, layout_q, cu_seqlens_q, batch_idx, q_idx, head_idx):
    if layout_q == "BSND":
        return query[batch_idx, q_idx, head_idx]
    if layout_q == "NTD":
        return query[head_idx, int(cu_seqlens_q[batch_idx].item()) + q_idx]
    return query[_query_index(layout_q, cu_seqlens_q, batch_idx, q_idx), head_idx]


def _get_q_scale(q_scale, layout_q, cu_seqlens_q, batch_idx, q_idx, head_idx):
    if layout_q == "BSND":
        return q_scale[batch_idx, q_idx, head_idx]
    if layout_q == "NTD":
        return q_scale[head_idx, int(cu_seqlens_q[batch_idx].item()) + q_idx]
    return q_scale[_query_index(layout_q, cu_seqlens_q, batch_idx, q_idx), head_idx]


def _get_k_scale(k_scale, layout_q, cu_seqlens_kv, batch_idx, positions, n2_idx):
    if layout_q == "BSND":
        return k_scale[batch_idx, positions, n2_idx]
    indices = [int(cu_seqlens_kv[batch_idx].item()) + pos for pos in positions]
    return k_scale[indices, n2_idx]


def _set_output(output, layout_q, cu_seqlens_q, batch_idx, q_idx, head_idx, value):
    if layout_q == "BSND":
        output[batch_idx, q_idx, head_idx] = value
    else:
        output[_query_index(layout_q, cu_seqlens_q, batch_idx, q_idx), head_idx] = value


def _set_lse(softmax_lse, layout_q, cu_seqlens_q, batch_idx, q_idx, head_idx, value):
    if layout_q == "BSND":
        softmax_lse[batch_idx, head_idx, q_idx] = value
    else:
        softmax_lse[head_idx, _query_index(layout_q, cu_seqlens_q, batch_idx, q_idx)] = value


def _make_scale_tensor(shape, pattern):
    if pattern == "ones":
        return torch.ones(shape, dtype=torch.float32)
    numel = math.prod(shape)
    values = torch.linspace(0.25, 2.0, steps=numel, dtype=torch.float32)
    return values.reshape(shape)


def _make_scales(case, cu_seqlens_q, cu_seqlens_kv, kv_lengths):
    layout_q = case["layout_q"]
    if layout_q == "BSND":
        q_shape = (case["B"], case["S1"], case["N1"])
        k_shape = (case["B"], case["S2"], case["N2"])
    elif layout_q == "NTD":
        q_shape = (case["N1"], int(cu_seqlens_q[-1].item()))
        k_shape = (int(cu_seqlens_kv[-1].item()), case["N2"])
    else:
        q_shape = (int(cu_seqlens_q[-1].item()), case["N1"])
        k_shape = (int(cu_seqlens_kv[-1].item()), case["N2"])

    q_scale = _make_scale_tensor(q_shape, case["scale_pattern"])
    k_scale = _make_scale_tensor(k_shape, case["scale_pattern"])
    if layout_q == "BSND":
        dense_k_scale = k_scale
    else:
        dense_k_scale = torch.zeros((case["B"], case["S2"], case["N2"]), dtype=torch.float32)
        for batch_idx, kv_len in enumerate(kv_lengths):
            start = int(cu_seqlens_kv[batch_idx].item())
            dense_k_scale[batch_idx, :kv_len] = k_scale[start:start + kv_len]
    if case["scale_pattern"] == "ones":
        v_scale = torch.ones((case["N2"],), dtype=torch.float32)
    else:
        v_scale = torch.linspace(0.5, 1.75, steps=case["N2"], dtype=torch.float32)
    p_scale = torch.tensor([float(case["p_scale_value"])], dtype=torch.float32)
    return q_scale, k_scale, dense_k_scale, v_scale, p_scale


def _mask_positions(case, q_idx, positions, q_len, kv_len):
    if case["mask_mode"] == 0:
        return torch.ones((len(positions),), dtype=torch.bool)
    if case["mask_mode"] != 3:
        raise ValueError(f"unsupported mask_mode: {case['mask_mode']}")
    limit = q_idx + kv_len - q_len
    return torch.tensor([pos <= limit for pos in positions], dtype=torch.bool)


def _positions_from_sparse(case, sparse_indices, sparse_seq_len, batch_idx, head_idx, qb_idx, kv_len):
    block_count = int(sparse_seq_len[batch_idx, head_idx, qb_idx].item())
    chunk_positions = []
    i = 0
    while i < block_count:
        pair_positions = []
        for j in range(2):
            if i + j < block_count:
                block_idx = int(sparse_indices[batch_idx, head_idx, qb_idx, i + j].item())
                if block_idx >= 0:
                    start = block_idx * case["sparse_kv_block_size"]
                    end = min(start + case["sparse_kv_block_size"], kv_len)
                    if start < kv_len:
                        pair_positions.extend(range(start, end))
        if pair_positions:
            chunk_positions.append(pair_positions)
        i += 2
    return chunk_positions


def _reference_attention(case, query, dense_key, dense_value, q_scale, k_scale, v_scale, p_scale,
                         sparse_indices, sparse_seq_len, cu_seqlens_q, cu_seqlens_kv, q_lengths, kv_lengths):
    # 向量化版本（采纳 MR!40 的 einops 思路把 query token 维度批量进 matmul），
    # 但严格保留参考实现的数据流以保证逐位一致：
    #  - 标度在 QK matmul 之后乘到 score 上（q_scale*k_scale*softmax_scale），不在 matmul 前缩放 q/k；
    #  - 沿 KV 轴按「2 个 sparse block」(s2LoopCount*2) 分块的 online/flash FP8 量化（运行中局部 max）；
    #  - 按 sparse_indices 中 block 的出现顺序拼接 positions（random/reverse/tail 非物理升序）来分块；
    #  - 实际长度掩码（q>=q_len、kv>=kv_len 无效；causal 对角线为 kv_len - q_len）；
    #  - 全掩码空行：输出 0、lse = EMPTY_LSE。
    query = query.to(torch.float32)
    dense_key = dense_key.to(torch.float32)
    dense_value = dense_value.to(torch.float32)

    layout_q = case["layout_q"]
    batch = case["B"]
    n1 = case["N1"]
    n2 = case["N2"]
    group = n1 // n2
    head_dim = case["D"]
    output_dtype = case["output_dtype"]
    fp8_dtype = _fp8_dtype()
    softmax_scale = float(case["softmax_scale"])
    p_scale_value = float(p_scale[0].item())
    sparse_q_block_size = case["sparse_q_block_size"]
    neg_inf = float("-inf")

    if layout_q == "BSND":
        attention_out = torch.zeros((batch, case["S1"], n1, head_dim), dtype=output_dtype)
        softmax_lse = torch.full((batch, n1, case["S1"]), EMPTY_LSE, dtype=torch.float32)
    else:
        total_q = int(cu_seqlens_q[-1].item())
        attention_out = torch.zeros((total_q, n1, head_dim), dtype=output_dtype)
        softmax_lse = torch.full((n1, total_q), EMPTY_LSE, dtype=torch.float32)

    qb_max = math.ceil(case["S1"] / sparse_q_block_size)
    for batch_idx in range(batch):
        q_len = q_lengths[batch_idx]
        kv_len = kv_lengths[batch_idx]
        for head_idx in range(n1):
            n2_idx = head_idx // group
            v_scale_value = float(v_scale[n2_idx].item())
            for qb_idx in range(qb_max):
                q_start = qb_idx * sparse_q_block_size
                if q_start >= q_len:
                    break
                q_end = min(q_start + sparse_q_block_size, q_len)
                q_indices = list(range(q_start, q_end))

                chunk_positions = _positions_from_sparse(
                    case, sparse_indices, sparse_seq_len, batch_idx, head_idx, qb_idx, kv_len)
                if not chunk_positions:
                    continue
                positions = [p for chunk in chunk_positions for p in chunk]

                # gather q 向量（批量），shape (nq, D)
                if layout_q == "BSND":
                    q_block = query[batch_idx, q_start:q_end, head_idx].to(torch.float32)
                    q_scale_block = q_scale[batch_idx, q_start:q_end, head_idx].to(torch.float32)
                elif layout_q == "NTD":
                    base = int(cu_seqlens_q[batch_idx].item())
                    q_block = query[head_idx, base + q_start:base + q_end].to(torch.float32)
                    q_scale_block = q_scale[head_idx, base + q_start:base + q_end].to(torch.float32)
                else:
                    base = int(cu_seqlens_q[batch_idx].item())
                    q_block = query[base + q_start:base + q_end, head_idx].to(torch.float32)
                    q_scale_block = q_scale[base + q_start:base + q_end, head_idx].to(torch.float32)
                nq = q_block.shape[0]

                # gather k/v/k_scale（按 positions 顺序），与参考一致使用 dense（padded S2）矩阵
                pos_tensor = torch.as_tensor(positions, dtype=torch.long)
                k_mat = dense_key[batch_idx, pos_tensor, n2_idx].to(torch.float32)      # (npos, D)
                v_mat = dense_value[batch_idx, pos_tensor, n2_idx].to(torch.float32)    # (npos, D)
                # k_scale：与参考 _get_k_scale 一致，TND 用 cu_seqlens 偏移；长度与 KV 矩阵一致
                if layout_q == "BSND":
                    k_scale_vec = k_scale[batch_idx, pos_tensor, n2_idx].to(torch.float32)  # (npos,)
                else:
                    kbase = int(cu_seqlens_kv[batch_idx].item())
                    k_scale_vec = k_scale[kbase + pos_tensor, n2_idx].to(torch.float32)     # (npos,)

                npos = pos_tensor.shape[0]
                # valid_mask: (nq, npos) — causal/actlen，与参考 _mask_positions 一致
                if case["mask_mode"] == 0:
                    valid_mask = torch.ones((nq, npos), dtype=torch.bool)
                elif case["mask_mode"] == 3:
                    q_idx_col = torch.as_tensor(q_indices, dtype=torch.long).view(nq, 1)
                    limit = q_idx_col + (kv_len - q_len)            # (nq,1)
                    valid_mask = pos_tensor.view(1, npos) <= limit  # (nq, npos)
                else:
                    raise ValueError(f"unsupported mask_mode: {case['mask_mode']}")

                # scores: (nq, npos)，标度在 matmul 之后施加（保持 FP8 桶一致）
                scores = torch.matmul(q_block, k_mat.transpose(0, 1))
                scores = scores * q_scale_block.view(nq, 1) * k_scale_vec.view(1, npos) * softmax_scale
                scores = torch.where(valid_mask, scores, torch.full_like(scores, MASK_VALUE))

                # 按 2-block chunk 的 online/flash FP8 数据流，向量化到 q 维
                m_run = torch.full((nq,), neg_inf, dtype=torch.float32)
                l_run = torch.zeros((nq,), dtype=torch.float32)
                acc = torch.zeros((nq, head_dim), dtype=torch.float32)
                offset = 0
                for chunk_pos_list in chunk_positions:
                    c0 = offset
                    c1 = offset + len(chunk_pos_list)
                    offset = c1
                    s_c = scores[:, c0:c1]
                    vm_c = valid_mask[:, c0:c1]
                    # 每行的局部 max（在该 chunk 内的有效位置上）；行内无有效位置时记为 -inf
                    masked_scores = torch.where(vm_c, s_c, torch.full_like(s_c, neg_inf))
                    chunk_max = masked_scores.max(dim=-1).values            # (nq,)
                    chunk_has = vm_c.any(dim=-1)                            # (nq,)
                    run_started = m_run != neg_inf
                    # m_new：未开始的行取 chunk_max（若该 chunk 无有效则保持 -inf）；已开始的行取 max
                    m_new = torch.where(run_started, torch.maximum(m_run, chunk_max), chunk_max)
                    # 该 chunk 对此行无有效位置 -> 此行本轮不更新
                    m_new = torch.where(chunk_has, m_new, m_run)
                    rescale = torch.where(run_started, torch.exp(m_run - m_new), torch.zeros_like(m_run))
                    rescale = torch.where(torch.isfinite(rescale), rescale, torch.zeros_like(rescale))
                    p_c = torch.exp(s_c - m_new.view(nq, 1))
                    p_c = torch.where(vm_c, p_c, torch.zeros_like(p_c))
                    # 仅对本轮有有效位置的行参与量化累加；其余行贡献 0
                    p_c = torch.where(chunk_has.view(nq, 1), p_c, torch.zeros_like(p_c))
                    p_quant = (p_c * p_scale_value).to(fp8_dtype).to(torch.float32)
                    pv = torch.matmul(p_quant, v_mat[c0:c1]) / p_scale_value * v_scale_value  # (nq, D)
                    acc = acc * rescale.view(nq, 1) + pv
                    l_run = l_run * rescale + p_c.sum(dim=-1)
                    m_run = torch.where(chunk_has, m_new, m_run)

                # any_valid 行：写回 attn=acc/l_run；空行保持 0 / EMPTY_LSE
                any_valid = valid_mask.any(dim=-1)                          # (nq,)
                safe_l = torch.where(l_run > 0, l_run, torch.ones_like(l_run))
                attn = acc / safe_l.view(nq, 1)                            # (nq, D)
                # lse = logsumexp(scores[valid]) per row
                lse_scores = torch.where(valid_mask, scores, torch.full_like(scores, neg_inf))
                lse = torch.logsumexp(lse_scores, dim=-1)                  # (nq,)

                for li, q_idx in enumerate(q_indices):
                    if not bool(any_valid[li].item()):
                        continue
                    _set_output(attention_out, layout_q, cu_seqlens_q, batch_idx, q_idx, head_idx,
                                attn[li].to(output_dtype))
                    _set_lse(softmax_lse, layout_q, cu_seqlens_q, batch_idx, q_idx, head_idx, lse[li])

    return attention_out, softmax_lse

def save_test_case(input_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    case_name = input_data["Testcase_Name"]
    input_filepath = os.path.join(output_dir, f"bsa_case_{case_name}.pt")
    torch.save(input_data, input_filepath)
    print(f"saved test case to: {input_filepath}")
    return input_filepath


def generate_and_save_testdata(params, save_pt=False, save_path=""):
    case = _normalize_params(params)
    check_valid_param.check_valid_param(case)

    rng = random.Random(case["seed"])
    generator = torch.Generator().manual_seed(case["seed"])

    batch = case["B"]
    s1 = case["S1"]
    s2 = case["S2"]
    n1 = case["N1"]
    n2 = case["N2"]
    head_dim = case["D"]
    sparse_q_block_size = case["sparse_q_block_size"]
    sparse_kv_block_size = case["sparse_kv_block_size"]

    q_lengths, q_max_len = _make_lengths(batch, s1, case["actlen_mode"], rng)
    kv_lengths, kv_max_len = _make_lengths(batch, s2, case["actlen_mode"], rng)
    s1, case["S1"] = q_max_len, q_max_len
    s2, case["S2"] = kv_max_len, kv_max_len
    if case["S1EQS2"]:
        kv_lengths = [min(length, s2) for length in q_lengths]

    cu_seqlens_q = _prefix(q_lengths)
    cu_seqlens_kv = _prefix(kv_lengths)
    seqused_q = torch.tensor(q_lengths, dtype=torch.int32)
    seqused_kv = torch.tensor(kv_lengths, dtype=torch.int32)

    if case["layout_q"] == "BSND":
        query = _rand_fp8((batch, s1, n1, head_dim), generator)
        cu_seqlens_q_input = None
        cu_seqlens_kv_input = None
    elif case["layout_q"] == "NTD":
        query = _rand_fp8((n1, int(cu_seqlens_q[-1].item()), head_dim), generator)
        cu_seqlens_q_input = cu_seqlens_q
        cu_seqlens_kv_input = cu_seqlens_kv
    else:
        query = _rand_fp8((int(cu_seqlens_q[-1].item()), n1, head_dim), generator)
        cu_seqlens_q_input = cu_seqlens_q
        cu_seqlens_kv_input = cu_seqlens_kv

    dense_key = _rand_fp8((batch, s2, n2, head_dim), generator)
    dense_value = _rand_fp8((batch, s2, n2, head_dim), generator)
    block_table = _make_block_table(batch, s2, sparse_kv_block_size, case["block_table_pattern"], rng)
    q_scale, k_scale, dense_k_scale, v_scale, p_scale = _make_scales(
        case, cu_seqlens_q, cu_seqlens_kv, kv_lengths)
    kv_cache_storage, kv_cache_meta = combined_kv_cache.pack_combined_kv_cache(
        dense_key, dense_value, dense_k_scale, block_table, sparse_kv_block_size,
        case["pa_block_padding_bytes"], case["layout_kv"])
    combined_kv_cache.assert_combined_kv_views(kv_cache_storage, kv_cache_meta)
    case["pa_block_stride"] = kv_cache_meta["pa_block_stride"]
    atten_mask = torch.tril(torch.ones((2048, 2048), dtype=torch.uint8)).T
    sparse_indices, sparse_seq_len = _make_sparse_indices(case, q_lengths, kv_lengths, rng)

    attention_out, softmax_lse = _reference_attention(
        case, query, dense_key, dense_value, q_scale, k_scale, v_scale, p_scale,
        sparse_indices, sparse_seq_len, cu_seqlens_q, cu_seqlens_kv, q_lengths, kv_lengths)

    if case["Testcase_Name"] is None:
        mode = "prefill" if case["layout_q"] in ("TND", "NTD") else "decode"
        case["Testcase_Name"] = (
            f"quantBlockSparseAttn_{mode}_{case['layout_q']}_{case['layout_kv']}_"
            f"{batch}_{n1}_{n2}_{s1}_{s2}_{head_dim}"
        )

    max_seqlen_q = max(q_lengths)
    max_seqlen_kv = max(kv_lengths)
    case["max_seqlen_q"] = max_seqlen_q
    case["max_seqlen_kv"] = max_seqlen_kv

    golden = {
        "attention_out": attention_out,
        "softmax_lse": softmax_lse,
    }
    input_data = {
        "Testcase_Name": case["Testcase_Name"],
        "params": case,
        "metadata_input": {
            "num_heads_q": n1,
            "num_heads_kv": n2,
            "head_dim": head_dim,
            "batch_size": batch,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_kv": max_seqlen_kv,
            "layout_q": case["layout_q"],
            "layout_kv": case["layout_kv"],
            "pa_block_stride": kv_cache_meta["pa_block_stride"],
        },
        "input": {
            "query": query,
            "kv_cache_storage": kv_cache_storage,
            "kv_cache_meta": kv_cache_meta,
            "q_descale": q_scale,
            "v_descale": v_scale,
            "p_scale": p_scale,
            "cu_seqlens_q": cu_seqlens_q_input,
            "cu_seqlens_kv": cu_seqlens_kv_input,
            "seqused_q": seqused_q,
            "seqused_kv": seqused_kv,
            "sparse_indices": sparse_indices,
            "sparse_seq_len": sparse_seq_len,
            "block_table": block_table,
            "atten_mask": atten_mask,
            "metadata": None,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_kv": max_seqlen_kv,
            "softmax_scale": case["softmax_scale"],
            "sparse_q_block_size": sparse_q_block_size,
            "sparse_kv_block_size": sparse_kv_block_size,
            "pa_block_stride": kv_cache_meta["pa_block_stride"],
            "layout_kv": case["layout_kv"],
            "layout_q": case["layout_q"],
            "layout_sparse_indices": case["layout_sparse_indices"],
            "layout_out": case["layout_out"],
            "quant_mode": case["quant_mode"],
            "mask_mode": case["mask_mode"],
            "return_softmax_lse": case["return_softmax_lse"],
        },
        "golden": golden,
        "cpu_output": attention_out,
        "cpu_softmax_lse": softmax_lse,
    }

    if save_pt:
        save_test_case(input_data, save_path)

    return input_data