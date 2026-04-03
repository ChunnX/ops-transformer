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
UNSUPPORTED_GOLDEN_MSG = (
    "当前 CPU golden 计算链路仍存在已知问题，容易导致 CPU 与 NPU 精度对比失败，"
    "当前 pytest 框架暂不支持 golden CPU 对比。"
)


def _resolve_actual_seq(actual_seq, default_value, batch_size, name):
    if actual_seq is None:
        return [default_value] * batch_size
    if len(actual_seq) != batch_size:
        raise ValueError(f"{name} length should equal B={batch_size}, but got {len(actual_seq)}")
    return [int(value) for value in actual_seq]


def _build_query_tensors(batch_size, seq_q, head_num_q, head_dim, rope_head_dim, q_type):
    query_base = torch.tensor(
        np.random.uniform(DATA_RANGE_QUERY_LEFT, DATA_RANGE_QUERY_RIGHT,
                          (batch_size, seq_q, head_num_q, head_dim)),
        dtype=q_type,
    )
    query_rope = torch.tensor(
        np.random.uniform(DATA_RANGE_QUERY_LEFT, DATA_RANGE_QUERY_RIGHT,
                          (batch_size, seq_q, head_num_q, rope_head_dim)),
        dtype=q_type,
    )
    return torch.cat((query_base, query_rope), dim=3)


def _build_kv_tensors(kv_dtype, kv_dim0, seq_kv, head_num_kv, head_dim, rope_head_dim, num_tiles, q_type):
    key_rope = torch.tensor(
        np.random.uniform(DATA_RANGE_ROPE_LEFT, DATA_RANGE_ROPE_RIGHT,
                          (kv_dim0, seq_kv, head_num_kv, rope_head_dim)),
        dtype=q_type,
    )
    antiquant_scale = torch.tensor(
        np.random.uniform(DATA_RANGE_SCALE_LEFT, DATA_RANGE_SCALE_RIGHT,
                          (kv_dim0, seq_kv, head_num_kv, num_tiles)),
        dtype=torch.float32,
    )

    if kv_dtype == "hifloat8":
        key_base = torch.tensor(
            np.random.uniform(DATA_RANGE_KV_LEFT, DATA_RANGE_KV_RIGHT,
                              (kv_dim0, seq_kv, head_num_kv, head_dim)),
            dtype=torch.uint8,
        )
        value = torch.tensor(
            np.random.uniform(DATA_RANGE_KV_LEFT, DATA_RANGE_KV_RIGHT,
                              (kv_dim0, seq_kv, head_num_kv, head_dim)),
            dtype=torch.uint8,
        )
        storage_dtype = torch.uint8
    elif kv_dtype in (None, "float8_e4m3fn"):
        key_base = torch.tensor(
            np.random.uniform(DATA_RANGE_KV_LEFT, 10, (kv_dim0, seq_kv, head_num_kv, head_dim))
        ).to(torch.float8_e4m3fn)
        value = torch.tensor(
            np.random.uniform(DATA_RANGE_KV_LEFT, 10, (kv_dim0, seq_kv, head_num_kv, head_dim))
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


def _build_pa_block_table(batch_size, block_num, block_size, seq_kv, actual_seq_kv_list):
    max_blocks_per_batch = max(
        math.ceil(seq_kv / block_size),
        math.ceil(max(actual_seq_kv_list) / block_size),
    )
    required_blocks = sum(math.ceil(seq / block_size) for seq in actual_seq_kv_list)
    if required_blocks > block_num:
        raise ValueError(
            f"block_num={block_num} is not enough for PA, expect at least {required_blocks}"
        )

    all_indices = np.random.permutation(np.arange(block_num, dtype=np.int32))
    block_table_np = np.full((batch_size, max_blocks_per_batch), -1, dtype=np.int32)
    offset = 0
    for batch_index in range(batch_size):
        needed_blocks = math.ceil(actual_seq_kv_list[batch_index] / block_size)
        block_table_np[batch_index, :needed_blocks] = all_indices[offset:offset + needed_blocks]
        offset += needed_blocks
    return torch.tensor(block_table_np, dtype=torch.int32)


def _build_sparse_indices(batch_size, seq_q, head_num_kv, sparse_count, sparse_block_size,
                          sparse_mode, actual_seq_q_list, actual_seq_kv_list):
    sparse_indices_np = np.full((batch_size, seq_q, head_num_kv, sparse_count), -1, dtype=np.int32)

    for batch_index in range(batch_size):
        actual_seq_q = actual_seq_q_list[batch_index]
        actual_seq_kv = actual_seq_kv_list[batch_index]
        valid_seq_q = min(seq_q, actual_seq_q)
        for query_index in range(valid_seq_q):
            if sparse_mode == 0:
                threshold = actual_seq_kv
            elif sparse_mode == 3:
                threshold = actual_seq_kv - actual_seq_q + query_index + 1
            else:
                threshold = actual_seq_kv

            if threshold <= 0:
                continue

            valid_blocks_max = math.ceil(max(0, threshold) / sparse_block_size)
            valid_blocks_topk = min(valid_blocks_max, sparse_count)
            if valid_blocks_topk <= 0:
                continue

            if valid_blocks_topk > 1:
                block_indices = np.random.permutation(valid_blocks_max - 1).astype(np.int32)
                sparse_indices_np[batch_index, query_index, :, :valid_blocks_topk - 1] = block_indices[:valid_blocks_topk - 1]
            sparse_indices_np[batch_index, query_index, :, valid_blocks_topk - 1] = valid_blocks_max - 1

    return torch.tensor(sparse_indices_np, dtype=torch.int32)


def _to_tnd(tensor_bsnd, actual_seq_list):
    tensor_list = []
    for batch_index, actual_seq in enumerate(actual_seq_list):
        tensor_list.append(tensor_bsnd[batch_index, :actual_seq])
    if not tensor_list:
        return tensor_bsnd.reshape(0, tensor_bsnd.shape[2], tensor_bsnd.shape[3])
    return torch.cat(tensor_list, dim=0)


def _to_tnd_sparse_indices(sparse_indices_bsnd, actual_seq_q_list):
    tensor_list = []
    for batch_index, actual_seq in enumerate(actual_seq_q_list):
        tensor_list.append(sparse_indices_bsnd[batch_index, :actual_seq])
    if not tensor_list:
        shape = sparse_indices_bsnd.shape
        return sparse_indices_bsnd.reshape(0, shape[2], shape[3])
    return torch.cat(tensor_list, dim=0)


def _build_actual_seq_tensor(layout, actual_seq_list):
    if layout == "TND":
        return torch.tensor(np.cumsum(actual_seq_list), dtype=torch.int32)
    return torch.tensor(actual_seq_list, dtype=torch.int32)


def _save_test_case(input_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    case_name = input_data["Testcase_Name"]
    filepath = os.path.join(output_dir, f"{case_name}.pt")
    torch.save(input_data, filepath)
    print(f"测试用例已保存到: {filepath}")
    return filepath


def generate_and_save_testdata(params, save_pt=False, save_path="", compute_golden=True):
    (testcase_name, layout_query, layout_kv, q_type, kv_dtype,
     batch_size, seq_q, seq_kv, head_num_q, head_num_kv, head_dim, sparse_count,
     scale_value, key_quant_mode, value_quant_mode, sparse_block_size,
     tile_size, rope_head_dim, sparse_mode, attention_mode, quant_scale_repo_mode,
     block_size, block_num, actual_seq_q, actual_seq_kv) = params

    num_tiles = head_dim // tile_size
    pa_mode = layout_kv == "PA_BSND"
    actual_seq_q_list = _resolve_actual_seq(actual_seq_q, seq_q, batch_size, "actual_seq_q")
    actual_seq_kv_list = _resolve_actual_seq(actual_seq_kv, seq_kv, batch_size, "actual_seq_kv")

    if pa_mode:
        if block_num is None:
            raise ValueError("PA mode requires block_num")
        kv_dim0 = block_num
        seq_kv_storage = block_size
    else:
        kv_dim0 = batch_size
        seq_kv_storage = seq_kv

    query_bsnd = _build_query_tensors(batch_size, seq_q, head_num_q, head_dim, rope_head_dim, q_type)
    value, antiquant_scale, key_cat = _build_kv_tensors(
        kv_dtype, kv_dim0, seq_kv_storage, head_num_kv, head_dim, rope_head_dim, num_tiles, q_type
    )

    final_key_dim = head_dim + 2 * rope_head_dim + 4 * num_tiles
    key = torch.as_strided(
        key_cat,
        size=(kv_dim0, seq_kv_storage, head_num_kv, final_key_dim),
        stride=(head_num_kv * final_key_dim * seq_kv_storage,
                head_num_kv * final_key_dim,
                final_key_dim,
                1),
    )

    key_dequant_scale = antiquant_scale
    value_dequant_scale = antiquant_scale

    sparse_indices_bsnd = _build_sparse_indices(
        batch_size, seq_q, head_num_kv, sparse_count, sparse_block_size,
        sparse_mode, actual_seq_q_list, actual_seq_kv_list,
    )

    if pa_mode:
        block_table = _build_pa_block_table(batch_size, block_num, block_size, seq_kv, actual_seq_kv_list)
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
        query_input = _to_tnd(query_bsnd, actual_seq_q_list)
        sparse_indices = _to_tnd_sparse_indices(sparse_indices_bsnd, actual_seq_q_list)
    else:
        query_input = query_bsnd
        sparse_indices = sparse_indices_bsnd

    actual_seq_q_tensor = _build_actual_seq_tensor(layout_query, actual_seq_q_list)

    if compute_golden:
        print(UNSUPPORTED_GOLDEN_MSG)
    cpu_result = None

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
        "cpu_output": cpu_result,
    }

    if save_pt:
        _save_test_case(input_data, save_path)
    return input_data
