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

import copy
import math
import os

import numpy as np
import torch


DATA_RANGE_QUERY_LEFT = -10
DATA_RANGE_QUERY_RIGHT = 100
DATA_RANGE_KV_LEFT = -5
DATA_RANGE_KV_RIGHT = 100
DATA_RANGE_ROPE_LEFT = -10
DATA_RANGE_ROPE_RIGHT = 10
DATA_RANGE_SCALE_LEFT = -100
DATA_RANGE_SCALE_RIGHT = 100


def _resolve_actual_seq(actual_seq, default_value, B, name):
    # 将可选的 actual_seq 统一整理成按 batch 对齐的长度列表。
    if actual_seq is None:
        return [default_value] * B
    if len(actual_seq) != B:
        raise ValueError(f"{name} length should equal B={B}, but got {len(actual_seq)}")
    return [int(value) for value in actual_seq]


def _build_query_tensors(B, S1, N1, D, rope_head_dim, q_type):
    # 构造 query 的 base 和 rope 部分，并在最后一维拼接。
    query_base = torch.tensor(
        np.random.uniform(DATA_RANGE_QUERY_LEFT, DATA_RANGE_QUERY_RIGHT,
                          (B, S1, N1, D)),
        dtype=q_type,
    )
    query_rope = torch.tensor(
        np.random.uniform(DATA_RANGE_QUERY_LEFT, DATA_RANGE_QUERY_RIGHT,
                          (B, S1, N1, rope_head_dim)),
        dtype=q_type,
    )
    return torch.cat((query_base, query_rope), dim=3)


def _build_kv_tensors(kv_dtype, kv_dim0, S2, N2, D, rope_head_dim, num_tiles, q_type):
    # 构造 K/V 存储张量，以及与 key 一起存放的 antiquant_scale。
    key_rope = torch.tensor(
        np.random.uniform(DATA_RANGE_ROPE_LEFT, DATA_RANGE_ROPE_RIGHT,
                          (kv_dim0, S2, N2, rope_head_dim)),
        dtype=q_type,
    )
    antiquant_scale = torch.tensor(
        np.random.uniform(DATA_RANGE_SCALE_LEFT, DATA_RANGE_SCALE_RIGHT,
                          (kv_dim0, S2, N2, num_tiles)),
        dtype=torch.float32,
    )

    if kv_dtype == "hifloat8":
        key_base = torch.tensor(
            np.random.uniform(DATA_RANGE_KV_LEFT, DATA_RANGE_KV_RIGHT,
                              (kv_dim0, S2, N2, D)),
            dtype=torch.uint8,
        )
        value = torch.tensor(
            np.random.uniform(DATA_RANGE_KV_LEFT, DATA_RANGE_KV_RIGHT,
                              (kv_dim0, S2, N2, D)),
            dtype=torch.uint8,
        )
        storage_dtype = torch.uint8
    elif kv_dtype in (None, "float8_e4m3fn"):
        key_base = torch.tensor(
            np.random.uniform(DATA_RANGE_KV_LEFT, 10, (kv_dim0, S2, N2, D))
        ).to(torch.float8_e4m3fn)
        value = torch.tensor(
            np.random.uniform(DATA_RANGE_KV_LEFT, 10, (kv_dim0, S2, N2, D))
        ).to(torch.float8_e4m3fn)
        storage_dtype = torch.float8_e4m3fn
    else:
        raise ValueError(f"Unsupported kv_dtype: {kv_dtype}")

    key_cat = torch.cat(
        (
            key_base,
            key_rope.view(storage_dtype),
            antiquant_scale.view(storage_dtype),
        ),
        dim=3,
    )
    return value, antiquant_scale, key_cat


def _build_pa_block_table(B, block_num, block_size, S2, actual_seq_kv_list):
    # PA 场景下为每个 batch 分配可用的缓存块索引。
    max_blocks_per_batch = max(
        math.ceil(S2 / block_size),
        math.ceil(max(actual_seq_kv_list) / block_size),
    )
    required_blocks = sum(math.ceil(seq / block_size) for seq in actual_seq_kv_list)
    if required_blocks > block_num:
        raise ValueError(
            f"block_num={block_num} is not enough for PA, expect at least {required_blocks}"
        )

    all_indices = np.random.permutation(np.arange(block_num, dtype=np.int32))
    block_table_np = np.full((B, max_blocks_per_batch), -1, dtype=np.int32)
    offset = 0
    for b in range(B):
        needed_blocks = math.ceil(actual_seq_kv_list[b] / block_size)
        block_table_np[b, :needed_blocks] = all_indices[offset:offset + needed_blocks]
        offset += needed_blocks
    return torch.tensor(block_table_np, dtype=torch.int32)


def _build_sparse_indices(B, S1, N2, K, sparse_block_size,
                          sparse_mode, actual_seq_q_list, actual_seq_kv_list):
    # 根据 sparse_mode 和实际长度生成每个 query 位置的稀疏 block 索引。
    sparse_indices_np = np.full((B, S1, N2, K), -1, dtype=np.int32)

    for b in range(B):
        actual_seq_q = actual_seq_q_list[b]
        actual_seq_kv = actual_seq_kv_list[b]
        valid_seq_q = min(S1, actual_seq_q)
        for s1_idx in range(valid_seq_q):
            if sparse_mode == 0:
                threshold = actual_seq_kv
            elif sparse_mode == 3:
                threshold = actual_seq_kv - actual_seq_q + s1_idx + 1
            else:
                threshold = actual_seq_kv

            if threshold <= 0:
                continue

            valid_blocks_max = math.ceil(max(0, threshold) / sparse_block_size)
            valid_blocks_topk = min(valid_blocks_max, K)
            if valid_blocks_topk <= 0:
                continue

            if valid_blocks_topk > 1:
                block_indices = np.random.permutation(valid_blocks_max - 1).astype(np.int32)
                sparse_indices_np[b, s1_idx, :, :valid_blocks_topk - 1] = block_indices[:valid_blocks_topk - 1]
            sparse_indices_np[b, s1_idx, :, valid_blocks_topk - 1] = valid_blocks_max - 1

    return torch.tensor(sparse_indices_np, dtype=torch.int32)


def _to_tnd(tensor_bsnd, actual_seq_list):
    # 将 BSND 数据按有效长度裁剪后压平为 TND。
    tensor_list = []
    for batch_index, actual_seq in enumerate(actual_seq_list):
        tensor_list.append(tensor_bsnd[batch_index, :actual_seq])
    if not tensor_list:
        return tensor_bsnd.reshape(0, tensor_bsnd.shape[2], tensor_bsnd.shape[3])
    return torch.cat(tensor_list, dim=0)


def _to_tnd_sparse_indices(sparse_indices_bsnd, actual_seq_q_list):
    # TND 下的 sparse_indices 也需要按有效 query token 重新拼接。
    tensor_list = []
    for batch_index, actual_seq in enumerate(actual_seq_q_list):
        tensor_list.append(sparse_indices_bsnd[batch_index, :actual_seq])
    if not tensor_list:
        shape = sparse_indices_bsnd.shape
        return sparse_indices_bsnd.reshape(0, shape[2], shape[3])
    return torch.cat(tensor_list, dim=0)


def _build_actual_seq_tensor(layout, actual_seq_list):
    # TND 需要前缀和形式，其他布局直接保持逐 batch 长度。
    if layout == "TND":
        return torch.tensor(np.cumsum(actual_seq_list), dtype=torch.int32)
    return torch.tensor(actual_seq_list, dtype=torch.int32)


def _save_test_case(input_data, output_dir):
    # batch 流程会先把输入固化成 pt，后续测试直接回放该文件。
    os.makedirs(output_dir, exist_ok=True)
    case_name = input_data["Testcase_Name"]
    filepath = os.path.join(output_dir, f"{case_name}.pt")
    torch.save(input_data, filepath)
    print(f"测试用例已保存到: {filepath}")
    return filepath


def generate_and_save_testdata(params, save_pt=False, save_path="", compute_golden=True):
    # 该接口负责把参数组装成 qsfa 直调所需的完整输入字典。
    (testcase_name, layout_query, layout_kv, q_type, kv_dtype,
     B, S1, S2, N1, N2, D, K,
     scale_value, key_quant_mode, value_quant_mode, sparse_block_size,
     tile_size, rope_head_dim, sparse_mode, attention_mode, quant_scale_repo_mode,
     block_size, block_num, actual_seq_q, actual_seq_kv) = params

    num_tiles = D // tile_size
    pa_mode = layout_kv == "PA_BSND"
    actual_seq_q_list = _resolve_actual_seq(actual_seq_q, S1, B, "actual_seq_q")
    actual_seq_kv_list = _resolve_actual_seq(actual_seq_kv, S2, B, "actual_seq_kv")

    if pa_mode:
        # PA_BSND 下 key/value 采用 block 形式存储。
        if block_num is None:
            raise ValueError("PA mode requires block_num")
        kv_dim0 = block_num
        S2_storage = block_size
    else:
        kv_dim0 = B
        S2_storage = S2

    query_bsnd = _build_query_tensors(B, S1, N1, D, rope_head_dim, q_type)
    value, antiquant_scale, key_cat = _build_kv_tensors(
        kv_dtype, kv_dim0, S2_storage, N2, D, rope_head_dim, num_tiles, q_type
    )

    final_key_dim = D + 2 * rope_head_dim + 4 * num_tiles
    key = torch.as_strided(
        key_cat,
        size=(kv_dim0, S2_storage, N2, final_key_dim),
        stride=(N2 * final_key_dim * S2_storage,
                N2 * final_key_dim,
                final_key_dim,
                1),
    )

    key_dequant_scale = antiquant_scale
    value_dequant_scale = antiquant_scale

    sparse_indices_bsnd = _build_sparse_indices(
        B, S1, N2, K, sparse_block_size,
        sparse_mode, actual_seq_q_list, actual_seq_kv_list,
    )

    if pa_mode:
        block_table = _build_pa_block_table(B, block_num, block_size, S2, actual_seq_kv_list)
        key_input = key
        value_input = value
        actual_seq_kv_tensor = torch.tensor(actual_seq_kv_list, dtype=torch.int32)
    else:
        block_table = None
        actual_seq_kv_tensor = _build_actual_seq_tensor(layout_kv, actual_seq_kv_list)
        if layout_kv == "TND":
            key_input = _to_tnd(key, actual_seq_kv_list)
            value_input = _to_tnd(value, actual_seq_kv_list)
        else:
            key_input = key
            value_input = value

    if layout_query == "TND":
        # TND 需要把每个 batch 的有效 token 拼成连续序列。
        query_input = _to_tnd(query_bsnd, actual_seq_q_list)
        sparse_indices = _to_tnd_sparse_indices(sparse_indices_bsnd, actual_seq_q_list)
    else:
        query_input = query_bsnd
        sparse_indices = sparse_indices_bsnd

    actual_seq_q_tensor = _build_actual_seq_tensor(layout_query, actual_seq_q_list)

    input_data = {
        "Testcase_Name": testcase_name,
        "params": params,
        "input": {
            "query": query_input,
            "key": key_input,
            "value": value_input,
            "sparse_indices": sparse_indices,
            "key_dequant_scale": key_dequant_scale,
            "value_dequant_scale": value_dequant_scale,
            "block_table": block_table,
            "actual_seq_lengths_query": actual_seq_q_tensor,
            "actual_seq_lengths_kv": actual_seq_kv_tensor,
            "scale_value": scale_value,
            "key_quant_mode": key_quant_mode,
            "value_quant_mode": value_quant_mode,
            "sparse_block_size": sparse_block_size,
            "layout_query": layout_query,
            "layout_kv": layout_kv,
            "sparse_mode": sparse_mode,
            "pre_tokens": (1 << 63) - 1,
            "next_tokens": (1 << 63) - 1,
            "attention_mode": attention_mode,
            "quant_scale_repo_mode": quant_scale_repo_mode,
            "tile_size": tile_size,
            "rope_head_dim": rope_head_dim,
            "kv_dtype": kv_dtype,
        },
        "cpu_output": None,
    }

    if compute_golden:
        input_data["cpu_output"] = compute_cpu_golden(input_data["input"])

    if save_pt:
        _save_test_case(input_data, save_path)
    return input_data


# =============================================================================
# CPU Golden 计算
# 以下函数从参考实现中提取，适配 pytest 工程的参数接口。
# =============================================================================

def _trans_tnd_actseq(prefix_sum_list):
    """TND 下将前缀和形式的 actual_seq 转回逐 batch 长度。"""
    result = [prefix_sum_list[0]]
    for i in range(len(prefix_sum_list) - 1):
        delta = prefix_sum_list[i + 1] - prefix_sum_list[i]
        if delta < 0:
            raise ValueError(f"actual_seq 前缀和不单调，第 {i} 项 delta={delta}")
        result.append(delta)
    return result


def _concat_tensor_bsnd(tensor1, tensor2, n):
    """沿最后一维拼接两个 BSND/TND/2D tensor，拼接前按 n 个 head 交错排列。"""
    s1, s2 = tensor1.shape, tensor2.shape
    if len(s1) != len(s2):
        raise ValueError(f"concat_tensor: 维数不同 {s1} vs {s2}")

    if len(s1) == 2:
        # (T, N*D) 格式：先 reshape 到 (..., N, D) 再拼
        d1, d2 = s1[1] // n, s2[1] // n
        t1 = tensor1.reshape(s1[0], n, d1)
        t2 = tensor2.reshape(s2[0], n, d2)
        return np.concatenate((t1, t2), axis=2).reshape(s1[0], s1[1] + s2[1])
    if len(s1) == 3:
        # TND：(T, N, D)，直接在 D 轴拼
        return np.concatenate((tensor1, tensor2), axis=2)
    if len(s1) == 4:
        # BSND：(B, S, N, D)，直接在 D 轴拼
        return np.concatenate((tensor1, tensor2), axis=3)
    raise ValueError(f"concat_tensor: 不支持 {len(s1)} 维")


def _trans_to_bnsd(tensor, layout, act_seq=None):
    """将 BSND/TND 格式转换为 BNSD (B, N, S, D)，用于统一计算。"""
    if layout in ("BSND", "PA_BSND"):
        # (B, S, N, D) -> (B, N, S, D)
        return tensor.transpose(0, 2, 1, 3)
    if layout == "TND":
        # (T, N, D) -> (B, N, S_max, D)，按 act_seq 分割并 padding
        T, N, D = tensor.shape
        B = len(act_seq)
        S_max = max(act_seq)
        out = np.zeros((B, N, S_max, D), dtype=tensor.dtype)
        t = 0
        for b, s in enumerate(act_seq):
            if s > 0:
                out[b, :, :s, :] = tensor[t:t + s].transpose(1, 0, 2)
            t += s
        return out
    raise ValueError(f"_trans_to_bnsd: 不支持 layout={layout}")


def _trans_bnsd_to_out_layout(tensor_bnsd, layout, act_q=None):
    """将 BNSD 计算结果转回输出 layout（torch tensor）。"""
    t = torch.from_numpy(tensor_bnsd.astype(np.float32))
    if layout in ("BSND", "PA_BSND"):
        return t.permute(0, 2, 1, 3).contiguous()
    if layout == "TND":
        T = sum(act_q)
        B, N, _, D = t.shape
        out = torch.zeros(T, N, D, dtype=t.dtype)
        pos = 0
        for b, s in enumerate(act_q):
            if s > 0:
                out[pos:pos + s] = t[b, :, :s, :].permute(1, 0, 2)
            pos += s
        return out
    raise ValueError(f"_trans_bnsd_to_out_layout: 不支持 layout={layout}")


def _softmax(x):
    """数值稳定的 float32 softmax（最后一维）。"""
    x = x.astype(np.float32)
    x_max = x.max(axis=-1, keepdims=True)
    e = np.exp(x - x_max)
    e_sum = e.sum(axis=-1, keepdims=True)
    return e / e_sum


def _gather_kv(k_bnsd, v_bnsd, sparse_indices, sparse_blocksize, sparse_blockcount,
               b, n2, s1, act_seq_kv, act_seq_q, sparse_mode):
    """按 sparse_indices 聚合稀疏 KV，返回 (emptyFlag, k_sparse, v_sparse)。"""
    if sparse_mode == 0:
        threshold = act_seq_kv
    else:  # sparse_mode == 3
        delta = act_seq_kv - act_seq_q
        threshold = delta + s1 + 1

    valid_count = min(sparse_blockcount, math.ceil(max(0, threshold) / sparse_blocksize))
    s2_indices = []
    for i in range(valid_count):
        blk_id = int(sparse_indices[i])
        if blk_id == -1:
            break
        begin = blk_id * sparse_blocksize
        end = min(begin + sparse_blocksize, act_seq_kv)
        if begin >= threshold:
            continue
        end = min(end, int(threshold))
        s2_indices.extend(range(begin, end))

    if not s2_indices:
        return True, None, None
    return False, k_bnsd[b, n2, s2_indices, :], v_bnsd[b, n2, s2_indices, :]


def _dequantize_kv(k_uint8_np, v_uint8_np, antiquant_scale_np, tile_size, kv_dtype):
    """将 float8 格式的 K/V（以 uint8 字节存储）按 tile 粒度反量化到 float32。

    k_uint8_np:       (kv_dim0, S, N, D) uint8 字节
    v_uint8_np:       (kv_dim0, S, N, D) uint8 字节
    antiquant_scale:  (kv_dim0, S, N, num_tiles) float32
    返回: k_fp32, v_fp32，形状同输入但 dtype=float32
    """
    kv_dim0, S, N, D = k_uint8_np.shape
    num_tiles = D // tile_size

    if kv_dtype == "hifloat8":
        # hifloat8 简单线性映射：有符号值 / 127.0
        k_raw = k_uint8_np.view(np.int8).astype(np.float32) / 127.0
        v_raw = v_uint8_np.view(np.int8).astype(np.float32) / 127.0
    else:
        # float8_e4m3fn：通过 torch 解码
        k_t = torch.from_numpy(k_uint8_np).view(torch.float8_e4m3fn).to(torch.float32)
        v_t = torch.from_numpy(v_uint8_np).view(torch.float8_e4m3fn).to(torch.float32)
        k_raw = k_t.numpy()
        v_raw = v_t.numpy()

    # antiquant_scale: (kv_dim0, S, N, num_tiles) -> 广播到 (kv_dim0, S, N, D)
    scale = antiquant_scale_np.reshape(kv_dim0, S, N, num_tiles, 1)
    scale = np.broadcast_to(scale, (kv_dim0, S, N, num_tiles, tile_size)).reshape(kv_dim0, S, N, D)

    return k_raw * scale, v_raw * scale


def _reconstruct_bsnd_from_blocks(key_block_np, v_block_np, block_table_np,
                                  block_size, B, S2_max, N2):
    """PA_BSND 场景：从 block cache 按 block_table 重建逻辑 BSND 布局。

    key_block_np:  (block_num, block_size, N, D_k) float32，D_k 含 rope 拼接
    v_block_np:    (block_num, block_size, N, D_v) float32，D_v 为 D_base
    block_table:   (B, max_blocks_per_batch) int32，-1 表示无效
    返回: k_bsnd (B, S2_max, N, D_k), v_bsnd (B, S2_max, N, D_v)
    """
    D_k = key_block_np.shape[3]
    D_v = v_block_np.shape[3]
    k_bsnd = np.zeros((B, S2_max, N2, D_k), dtype=np.float32)
    v_bsnd = np.zeros((B, S2_max, N2, D_v), dtype=np.float32)
    for b in range(B):
        for blk_i, blk_id in enumerate(block_table_np[b]):
            if blk_id < 0:
                continue
            s_begin = blk_i * block_size
            s_end = s_begin + block_size
            if s_begin >= S2_max:
                break
            s_end = min(s_end, S2_max)
            copy_len = s_end - s_begin
            k_bsnd[b, s_begin:s_end] = key_block_np[blk_id, :copy_len]
            v_bsnd[b, s_begin:s_end] = v_block_np[blk_id, :copy_len]
    return k_bsnd, v_bsnd


def _attention_bnsd(q_bnsd, k_bnsd, v_bnsd, sparse_indices_bnsd,
                    act_seq_q_list, act_seq_kv_list,
                    scale_value, sparse_blocksize, sparse_blockcount,
                    sparse_mode, q_dtype_str):
    """核心 attention 计算，工作在 BNSD 空间。

    q_bnsd:               (B, N1, S1, D_q)
    k_bnsd / v_bnsd:      (B, N2, S2, D_kv)
    sparse_indices_bnsd:  (B, N2, S1, K)
    返回: y (B, N1, S1, D_kv) float32
    """
    B, N1, S1, D_q = q_bnsd.shape
    N2 = k_bnsd.shape[1]
    D_kv = v_bnsd.shape[-1]
    g = N1 // N2  # GQA group factor
    y = np.zeros((B, N1, S1, D_kv), dtype=np.float32)

    for b in range(B):
        act_q = act_seq_q_list[b]
        act_kv = act_seq_kv_list[b]
        for n2_idx in range(N2):
            for s1_idx in range(act_q):
                if sparse_mode == 3 and s1_idx < act_q - act_kv:
                    # 该 query token 在因果窗口之外，输出保持 0
                    continue
                q_curr = q_bnsd[b, n2_idx * g:(n2_idx + 1) * g, s1_idx, :]
                sp_idx = sparse_indices_bnsd[b, n2_idx, s1_idx, :]

                empty, k_sp, v_sp = _gather_kv(
                    k_bnsd, v_bnsd, sp_idx,
                    sparse_blocksize, sparse_blockcount,
                    b, n2_idx, s1_idx, act_kv, act_q, sparse_mode
                )
                if empty:
                    continue

                # bmm1: (g, D_q) x (sparse_len, D_q)^T -> (g, sparse_len)
                bmm1 = np.matmul(q_curr.astype(np.float32),
                                 k_sp.astype(np.float32).T) * scale_value
                sm = _softmax(bmm1)

                # 转回 q_dtype 再做 bmm2，与 NPU 精度行为对齐
                if q_dtype_str == "float16":
                    sm_cast = sm.astype(np.float16).astype(np.float32)
                elif q_dtype_str == "bfloat16":
                    sm_cast = torch.from_numpy(sm.astype(np.float32)).to(torch.bfloat16).float().numpy()
                else:
                    sm_cast = sm

                bmm2 = np.matmul(sm_cast, v_sp.astype(np.float32))
                y[b, n2_idx * g:(n2_idx + 1) * g, s1_idx, :] = bmm2
    return y


def _build_fa_param(inp):
    """从 pytest input_data["input"] 构建 CPU 计算所需的 fa_param 字典。

    处理流程：
    1. 从打包的 key tensor 中拆出 key_base / key_rope / antiquant_scale
    2. 反量化 key_base / value 到 float32
    3. PA_BSND 场景下从 block cache 重建逻辑 BSND 格式的 K/V
    4. 将 Q、K、V、sparse_indices 转换到 BNSD 空间
    """
    layout_query = inp["layout_query"]
    layout_kv = inp["layout_kv"]
    kv_dtype = inp["kv_dtype"]
    tile_size = inp["tile_size"]
    rope_head_dim = inp["rope_head_dim"]
    sparse_block_size = inp["sparse_block_size"]
    sparse_mode = inp["sparse_mode"]
    scale_value = inp["scale_value"]
    q_dtype_str = "bfloat16" if inp["query"].dtype == torch.bfloat16 else "float16"

    # ---------- 解析 actual_seq ----------
    act_seq_q_raw = inp["actual_seq_lengths_query"].tolist()
    act_seq_kv_raw = inp["actual_seq_lengths_kv"].tolist()

    if layout_query == "TND":
        act_seq_q_list = _trans_tnd_actseq(act_seq_q_raw)
    else:
        act_seq_q_list = [int(v) for v in act_seq_q_raw]

    if layout_kv == "TND":
        act_seq_kv_list = _trans_tnd_actseq(act_seq_kv_raw)
    elif layout_kv == "PA_BSND":
        act_seq_kv_list = [int(v) for v in act_seq_kv_raw]
    else:
        act_seq_kv_list = [int(v) for v in act_seq_kv_raw]

    B = len(act_seq_q_list)

    # ---------- Query：转 numpy，BSND/TND -> BNSD ----------
    query_np = inp["query"].float().numpy()  # float16/bf16 -> float32
    if layout_query == "TND":
        # (T, N, D_full) -> BNSD
        N1 = query_np.shape[1]
        D_full = query_np.shape[2]
    else:
        N1 = query_np.shape[2]
        D_full = query_np.shape[3]
    D_base = D_full - rope_head_dim

    q_bnsd = _trans_to_bnsd(query_np, layout_query, act_seq_q_list)  # (B, N1, S1, D_full)
    # Q 的前 D_base 维是 q_base，后 rope_head_dim 维是 q_rope；
    # golden.py 中 preprocessing 把 k_rope 拼到 K，这里对 Q 同样只取 base 部分参与 attention 计算
    # （NPU 内部已将 rope 融合，golden 中 Q 和 K 均用 base+rope 拼接形式传入，
    #  但 attention 输出维度只有 D_base，因此切掉 rope 部分）
    q_bnsd_full = q_bnsd  # 保留完整以备 K 维度匹配

    # ---------- 拆分打包的 key tensor ----------
    # key 打包格式（参见 _build_kv_tensors）：
    #   cat(key_base, key_rope.view(storage_dtype), antiquant_scale.view(storage_dtype), dim=3)
    # 在 storage_dtype（uint8 或 float8_e4m3fn，均 1 字节）空间中：
    #   [0 : D]                                key_base   (D 个 storage_dtype)
    #   [D : D + rope_head_dim*2]              key_rope   (rope_head_dim 个 q_type，各 2 字节)
    #   [D + rope_head_dim*2 : end]            antiquant_scale (num_tiles 个 float32，各 4 字节)
    # 即 key_packed_D = D + 2*rope_head_dim + 4*num_tiles
    key_packed = inp["key"]  # torch tensor, storage_dtype (uint8 or float8_e4m3fn)
    kv_dim0, S2_blk, N2, key_packed_D = key_packed.shape
    # rope 区占用 slot 数
    rope_slots = rope_head_dim * 2   # q_type (fp16/bf16) 各 2 字节，共 2*rope_head_dim slots
    # scale 区占用 slot 数
    scale_slots_total = key_packed_D - D_base - rope_slots
    num_tiles = scale_slots_total // 4  # float32 各 4 字节

    key_base_raw = key_packed[..., :D_base].contiguous()
    key_rope_slots_raw = key_packed[..., D_base:D_base + rope_slots].contiguous()
    scale_start = D_base + rope_slots
    antiquant_scale_slots_raw = key_packed[..., scale_start:scale_start + scale_slots_total].contiguous()

    # 统一转为 uint8 字节数组便于后续解释
    if kv_dtype == "hifloat8":
        key_base_np = key_base_raw.numpy()  # 已是 uint8
        key_rope_slots_np = key_rope_slots_raw.numpy()
        antiquant_scale_slots_np = antiquant_scale_slots_raw.numpy()
    else:
        # float8_e4m3fn 没有对应 numpy dtype，通过 view(uint8) 取字节
        key_base_np = key_base_raw.view(torch.uint8).numpy()
        key_rope_slots_np = key_rope_slots_raw.view(torch.uint8).numpy()
        antiquant_scale_slots_np = antiquant_scale_slots_raw.view(torch.uint8).numpy()

    # 恢复 key_rope 为 float32（先取字节，再按 q_type 解释）
    key_rope_bytes = key_rope_slots_np.reshape(-1).tobytes()
    if q_dtype_str == "float16":
        key_rope_np = np.frombuffer(key_rope_bytes, dtype=np.float16).astype(np.float32)
    else:
        key_rope_np = torch.frombuffer(bytearray(key_rope_bytes), dtype=torch.bfloat16).float().numpy()
    key_rope_np = key_rope_np.reshape(kv_dim0, S2_blk, N2, rope_head_dim)

    # 恢复 antiquant_scale 为 float32
    scale_bytes = antiquant_scale_slots_np.reshape(-1).tobytes()
    antiquant_scale_np = np.frombuffer(scale_bytes, dtype=np.float32).reshape(kv_dim0, S2_blk, N2, num_tiles).copy()

    # ---------- Value ----------
    # value shape: (kv_dim0, S2_blk, N2, D) storage_dtype
    if kv_dtype == "hifloat8":
        value_np = inp["value"].contiguous().numpy()  # uint8
    else:
        value_np = inp["value"].contiguous().view(torch.uint8).numpy()

    # ---------- 反量化 K/V ----------
    k_fp32, v_fp32 = _dequantize_kv(
        key_base_np.reshape(kv_dim0, S2_blk, N2, D_base),
        value_np.reshape(kv_dim0, S2_blk, N2, D_base),
        antiquant_scale_np, tile_size, kv_dtype
    )

    # 将 key_rope 拼回 k（与 preprocessing 逻辑一致）
    # k_full: (kv_dim0, S2_blk, N2, D_base + rope_head_dim)
    k_full = np.concatenate((k_fp32, key_rope_np.astype(np.float32)), axis=3)

    # ---------- PA_BSND: 从 block cache 重建 BSND ----------
    if layout_kv == "PA_BSND":
        block_table_np = inp["block_table"].numpy()  # (B, max_blk)
        block_size = S2_blk
        S2_max = max(act_seq_kv_list)
        k_bsnd_np, v_bsnd_np = _reconstruct_bsnd_from_blocks(
            k_full, v_fp32, block_table_np, block_size, B, S2_max, N2
        )
        # 转 BNSD
        k_bnsd = k_bsnd_np.transpose(0, 2, 1, 3)
        v_bnsd = v_bsnd_np.transpose(0, 2, 1, 3)
    else:
        # BSND / TND
        k_bnsd = _trans_to_bnsd(k_full, layout_kv, act_seq_kv_list)
        v_bnsd = _trans_to_bnsd(v_fp32, layout_kv, act_seq_kv_list)

    # ---------- sparse_indices ----------
    sparse_np = inp["sparse_indices"].numpy()  # BSND 或 TND 格式
    if layout_query == "TND":
        sp_bnsd = _trans_to_bnsd(sparse_np, "TND", act_seq_q_list)
    else:
        sp_bnsd = sparse_np.transpose(0, 2, 1, 3)  # BSND -> BNSD

    K = sp_bnsd.shape[3]

    fa_param = {
        "layout_query": layout_query,
        "layout_kv": layout_kv,
        "q_bnsd": q_bnsd_full,
        "k_bnsd": k_bnsd,
        "v_bnsd": v_bnsd,
        "sp_bnsd": sp_bnsd,
        "act_seq_q": act_seq_q_list,
        "act_seq_kv": act_seq_kv_list,
        "scale_value": float(scale_value),
        "sparse_blocksize": sparse_block_size,
        "sparse_blockcount": K,
        "sparse_mode": sparse_mode,
        "q_dtype_str": q_dtype_str,
    }
    return fa_param


def compute_cpu_golden(inp):
    """CPU golden 计算入口，接收 input_data["input"] 字典，返回 float32 torch tensor。

    返回 tensor 的 layout 与 layout_query 一致（BSND 或 TND）。
    """
    fa = _build_fa_param(inp)

    y_bnsd = _attention_bnsd(
        fa["q_bnsd"], fa["k_bnsd"], fa["v_bnsd"], fa["sp_bnsd"],
        fa["act_seq_q"], fa["act_seq_kv"],
        fa["scale_value"], fa["sparse_blocksize"], fa["sparse_blockcount"],
        fa["sparse_mode"], fa["q_dtype_str"],
    )

    # y_bnsd 最后一维来自 v_bnsd（D_base 维），不包含 rope，无需裁剪。
    return _trans_bnsd_to_out_layout(y_bnsd, fa["layout_query"], fa["act_seq_q"])
