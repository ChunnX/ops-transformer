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


def _resolve_actual_seq(actual_seq, default_value, batch_size, name):
    if actual_seq is None:
        return [int(default_value)] * batch_size
    if len(actual_seq) != batch_size:
        raise ValueError(f"{name} 长度应等于 B={batch_size}，当前: {len(actual_seq)}")
    return [int(value) for value in actual_seq]


def _rand_tensor(shape, dtype, low, high):
    return torch.tensor(np.random.uniform(low, high, shape)).to(dtype)


def _build_sparse_indices(batch_size, seq_len_q, num_heads_kv, topk, sparse_block_size,
                          sparse_mode, actual_seq_q, actual_seq_kv):
    # 这里按 sparse block 粒度生成可见 kv block 索引，未命中的位置统一填 -1。
    q_block_count = math.ceil(seq_len_q / sparse_block_size)
    sparse_indices = np.full((batch_size, q_block_count, num_heads_kv, topk), -1, dtype=np.int32)

    for batch_index in range(batch_size):
        valid_q_block_count = math.ceil(actual_seq_q[batch_index] / sparse_block_size)
        for q_block_index in range(valid_q_block_count):
            q_end = min((q_block_index + 1) * sparse_block_size, actual_seq_q[batch_index])
            if sparse_mode == 0:
                visible_tokens = actual_seq_kv[batch_index]
            else:
                visible_tokens = max(0, actual_seq_kv[batch_index] - actual_seq_q[batch_index] + q_end)

            kv_block_count = math.ceil(visible_tokens / sparse_block_size)
            selected_block_count = min(topk, kv_block_count)
            if selected_block_count <= 0:
                continue

            candidates = list(range(kv_block_count))
            if selected_block_count < len(candidates):
                picked_blocks = sorted(random.sample(candidates, selected_block_count - 1))
                picked_blocks.append(kv_block_count - 1)
            else:
                picked_blocks = candidates

            sparse_indices[batch_index, q_block_index, :, :len(picked_blocks)] = np.array(
                picked_blocks, dtype=np.int32
            )
    return torch.tensor(sparse_indices, dtype=torch.int32)


def _build_block_table(batch_size, seq_len_kv, block_size, block_num, actual_seq_kv):
    blocks_per_batch = max(
        math.ceil(seq_len_kv / block_size),
        max(math.ceil(seq / block_size) for seq in actual_seq_kv),
    )
    block_table = np.full((batch_size, blocks_per_batch), -1, dtype=np.int32)
    cursor = 0

    for batch_index in range(batch_size):
        needed_blocks = math.ceil(actual_seq_kv[batch_index] / block_size)
        if cursor + needed_blocks > block_num:
            raise ValueError(
                f"block_num={block_num} 不足以覆盖实际 KV 长度，batch={batch_index} 需要 {needed_blocks} 个块"
            )
        block_table[batch_index, :needed_blocks] = np.arange(cursor, cursor + needed_blocks, dtype=np.int32)
        cursor += needed_blocks
    return torch.tensor(block_table, dtype=torch.int32)


def _to_tnd(tensor_bsnd, actual_seq):
    pieces = [tensor_bsnd[batch_index, :actual_seq[batch_index]] for batch_index in range(len(actual_seq))]
    if not pieces:
        return tensor_bsnd.reshape(0, tensor_bsnd.shape[2], tensor_bsnd.shape[3])
    return torch.cat(pieces, dim=0)


def _to_tnd_sparse_indices(sparse_indices_bsnd, actual_seq_q, sparse_block_size):
    pieces = []
    for batch_index, seq in enumerate(actual_seq_q):
        q_block_count = math.ceil(seq / sparse_block_size)
        pieces.append(sparse_indices_bsnd[batch_index, :q_block_count])
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
    (_, layout_query, layout_kv, q_type,
     batch_size, seq_len_q, seq_len_kv, num_heads_q, num_heads_kv, head_dim, topk,
     scale_value, sparse_block_size, rope_head_dim,
     sparse_mode, attention_mode, return_softmax_lse,
     block_size, block_num, actual_seq_q, actual_seq_kv) = params

    sparse_block_size = int(sparse_block_size)
    rope_head_dim = int(rope_head_dim)
    block_size = int(block_size)
    block_num = None if block_num is None else int(block_num)

    actual_seq_q_list = _resolve_actual_seq(actual_seq_q, seq_len_q, batch_size, "actual_seq_q")
    actual_seq_kv_list = _resolve_actual_seq(actual_seq_kv, seq_len_kv, batch_size, "actual_seq_kv")

    query_bsnd = _rand_tensor((batch_size, seq_len_q, num_heads_q, head_dim), q_type, QUERY_LOW, QUERY_HIGH)
    query_rope_bsnd = _rand_tensor(
        (batch_size, seq_len_q, num_heads_q, rope_head_dim), q_type, ROPE_LOW, ROPE_HIGH
    )
    sparse_indices_bsnd = _build_sparse_indices(
        batch_size, seq_len_q, num_heads_kv, topk, sparse_block_size, sparse_mode,
        actual_seq_q_list, actual_seq_kv_list,
    )

    if layout_kv == "PA_BSND":
        if block_num is None:
            raise ValueError("PA_BSND 场景必须提供 block_num")
        key = _rand_tensor((block_num, block_size, num_heads_kv, head_dim), q_type, KEY_LOW, KEY_HIGH)
        value = _rand_tensor((block_num, block_size, num_heads_kv, head_dim), q_type, KEY_LOW, KEY_HIGH)
        key_rope = _rand_tensor(
            (block_num, block_size, num_heads_kv, rope_head_dim), q_type, ROPE_LOW, ROPE_HIGH
        )
        block_table = _build_block_table(batch_size, seq_len_kv, block_size, block_num, actual_seq_kv_list)
    else:
        key_bsnd = _rand_tensor((batch_size, seq_len_kv, num_heads_kv, head_dim), q_type, KEY_LOW, KEY_HIGH)
        value_bsnd = _rand_tensor((batch_size, seq_len_kv, num_heads_kv, head_dim), q_type, KEY_LOW, KEY_HIGH)
        key_rope_bsnd = _rand_tensor(
            (batch_size, seq_len_kv, num_heads_kv, rope_head_dim), q_type, ROPE_LOW, ROPE_HIGH
        )
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
        "Testcase_Name": params[0],
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
