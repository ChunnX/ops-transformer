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

import numpy as np
import torch


QUERY_LOW = -10.0
QUERY_HIGH = 10.0
KEY_LOW = -5.0
KEY_HIGH = 10.0
ROPE_LOW = -10.0
ROPE_HIGH = 10.0


def _resolve_actual_seq(actual_seq, default_value, B, name):
    if actual_seq is None:
        return [int(default_value)] * B
    if len(actual_seq) != B:
        raise ValueError(f"{name} 长度应等于 B={B}，当前: {len(actual_seq)}")
    return [int(v) for v in actual_seq]


def _rand_tensor(shape, dtype, low, high):
    return torch.tensor(np.random.uniform(low, high, shape)).to(dtype)


def _build_sparse_indices(B, S1, N2, K, sparse_block_size, sparse_mode, actual_seq_q, actual_seq_kv):
    # 这里按 sparse block 粒度生成可见 kv block 索引，未命中的位置统一填 -1。
    q_blocks = math.ceil(S1 / sparse_block_size)
    sparse_indices = np.full((B, q_blocks, N2, K), -1, dtype=np.int32)
    for b in range(B):
        q_blocks_valid = math.ceil(actual_seq_q[b] / sparse_block_size)
        for q_blk in range(q_blocks_valid):
            q_end = min((q_blk + 1) * sparse_block_size, actual_seq_q[b])
            if sparse_mode == 0:
                visible_tokens = actual_seq_kv[b]
            else:
                visible_tokens = max(0, actual_seq_kv[b] - actual_seq_q[b] + q_end)
            kv_blocks = math.ceil(visible_tokens / sparse_block_size)
            topk = min(K, kv_blocks)
            if topk <= 0:
                continue
            candidates = list(range(kv_blocks))
            if topk < len(candidates):
                picked = sorted(random.sample(candidates, topk - 1)) + [kv_blocks - 1]
            else:
                picked = candidates
            sparse_indices[b, q_blk, :, :len(picked)] = np.array(picked, dtype=np.int32)
    return torch.tensor(sparse_indices, dtype=torch.int32)


def _build_block_table(B, S2, block_size, block_num, actual_seq_kv):
    blocks_per_batch = max(math.ceil(S2 / block_size), max(math.ceil(seq / block_size) for seq in actual_seq_kv))
    table = np.full((B, blocks_per_batch), -1, dtype=np.int32)
    cursor = 0
    for b in range(B):
        need = math.ceil(actual_seq_kv[b] / block_size)
        if cursor + need > block_num:
            raise ValueError(f"block_num={block_num} 不足以覆盖实际 KV 长度，batch={b} 需要 {need} 个块")
        table[b, :need] = np.arange(cursor, cursor + need, dtype=np.int32)
        cursor += need
    return torch.tensor(table, dtype=torch.int32)


def _to_tnd(tensor_bsnd, actual_seq):
    pieces = [tensor_bsnd[b, :actual_seq[b]] for b in range(len(actual_seq))]
    if not pieces:
        return tensor_bsnd.reshape(0, tensor_bsnd.shape[2], tensor_bsnd.shape[3])
    return torch.cat(pieces, dim=0)


def _to_tnd_sparse_indices(sparse_indices_bsnd, actual_seq_q, sparse_block_size):
    pieces = []
    for b, seq in enumerate(actual_seq_q):
        q_blocks = math.ceil(seq / sparse_block_size)
        pieces.append(sparse_indices_bsnd[b, :q_blocks])
    if not pieces:
        shape = sparse_indices_bsnd.shape
        return sparse_indices_bsnd.reshape(0, shape[2], shape[3])
    return torch.cat(pieces, dim=0)


def _save_test_case(test_data, save_path):
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f"{test_data['Testcase_Name']}.pt")
    torch.save(test_data, file_path)
    print(f"测试用例已保存到: {file_path}")
    return file_path


def generate_and_save_testdata(params, save_pt=False, save_path="", compute_golden=False):
    (Testcase_Name, layout_query, layout_kv, q_type,
     B, S1, S2, N1, N2, D, K,
     scale_value, sparse_block_size, rope_head_dim,
     sparse_mode, attention_mode, return_softmax_lse,
     block_size, block_num, actual_seq_q, actual_seq_kv) = params

    sparse_block_size = int(sparse_block_size)
    rope_head_dim = int(rope_head_dim)
    block_size = int(block_size)
    block_num = None if block_num is None else int(block_num)

    actual_seq_q_list = _resolve_actual_seq(actual_seq_q, S1, B, "actual_seq_q")
    actual_seq_kv_list = _resolve_actual_seq(actual_seq_kv, S2, B, "actual_seq_kv")

    query_bsnd = _rand_tensor((B, S1, N1, D), q_type, QUERY_LOW, QUERY_HIGH)
    query_rope_bsnd = _rand_tensor((B, S1, N1, rope_head_dim), q_type, ROPE_LOW, ROPE_HIGH)
    sparse_indices_bsnd = _build_sparse_indices(
        B, S1, N2, K, sparse_block_size, sparse_mode, actual_seq_q_list, actual_seq_kv_list
    )

    if layout_kv == "PA_BSND":
        if block_num is None:
            raise ValueError("PA_BSND 场景必须提供 block_num")
        key = _rand_tensor((block_num, block_size, N2, D), q_type, KEY_LOW, KEY_HIGH)
        value = _rand_tensor((block_num, block_size, N2, D), q_type, KEY_LOW, KEY_HIGH)
        key_rope = _rand_tensor((block_num, block_size, N2, rope_head_dim), q_type, ROPE_LOW, ROPE_HIGH)
        block_table = _build_block_table(B, S2, block_size, block_num, actual_seq_kv_list)
    else:
        key_bsnd = _rand_tensor((B, S2, N2, D), q_type, KEY_LOW, KEY_HIGH)
        value_bsnd = _rand_tensor((B, S2, N2, D), q_type, KEY_LOW, KEY_HIGH)
        key_rope_bsnd = _rand_tensor((B, S2, N2, rope_head_dim), q_type, ROPE_LOW, ROPE_HIGH)
        block_table = None
        if layout_kv == "TND":
            key = _to_tnd(key_bsnd, actual_seq_kv_list)
            value = _to_tnd(value_bsnd, actual_seq_kv_list)
            key_rope = _to_tnd(key_rope_bsnd, actual_seq_kv_list)
        else:
            key = key_bsnd
            value = value_bsnd
            key_rope = key_rope_bsnd

    if layout_query == "TND":
        query = _to_tnd(query_bsnd, actual_seq_q_list)
        query_rope = _to_tnd(query_rope_bsnd, actual_seq_q_list)
        sparse_indices = _to_tnd_sparse_indices(sparse_indices_bsnd, actual_seq_q_list, sparse_block_size)
    else:
        query = query_bsnd
        query_rope = query_rope_bsnd
        sparse_indices = sparse_indices_bsnd

    test_data = {
        "Testcase_Name": Testcase_Name,
        "params": params,
        "input": {
            "query": query,
            "key": key,
            "value": value,
            "sparse_indices": sparse_indices,
            "block_table": block_table,
            "actual_seq_lengths_query": torch.tensor(actual_seq_q_list, dtype=torch.int32),
            "actual_seq_lengths_kv": torch.tensor(actual_seq_kv_list, dtype=torch.int32),
            "query_rope": query_rope,
            "key_rope": key_rope,
            "scale_value": scale_value,
            "sparse_block_size": sparse_block_size,
            "layout_query": layout_query,
            "layout_kv": layout_kv,
            "sparse_mode": sparse_mode,
            "pre_tokens": (1 << 63) - 1,
            "next_tokens": (1 << 63) - 1,
            "attention_mode": attention_mode,
            "return_softmax_lse": return_softmax_lse,
        },
        "cpu_output": None if not compute_golden else None,
    }

    if save_pt:
        _save_test_case(test_data, save_path)
    return test_data
