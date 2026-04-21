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
import re
import numpy as np
import torch
import tools
import copy
import time
import tensorflow as tf
from ml_dtypes import float8_e4m3fn
import generate_tensor_data


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

    if compute_golden:
        # 当前版本不再生成 CPU golden，只输出说明信息。
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


def trans_np_hifuint8_tensor_to_float32(in_tensor):
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    print(f"[trans_np_hifuint8_tensor_to_float32]total size:{multi_shape}")
    out_tensor = np.zeros(multi_shape).astype(np.float32)
    in_tensor = in_tensor.reshape(multi_shape)
    for i in range(multi_shape):
        out_tensor[i] = cvt_hifuint8_to_float(in_tensor[i])
    out_tensor = out_tensor.reshape(shape_tensor).astype(np.float32)
    return out_tensor


def cvt_hifuint8_to_float(x, over_mode=True):
    x = int(x)
    if x == 0:
        return float(0)
    elif x == 128:
        if over_mode:
            return np.nan
        else:
            return float(0)
    elif x == 239:
        if over_mode:
            return -np.inf
        else:
            return -32768
    elif x == 111:
        if over_mode:
            return np.inf
        else:
            return 32768
    else:
        if x >= 128:
            sign = -1.0
        else:
            sign = 1.0
        dot_4_bits = x & 120 #b01111000 = 120
        dot_4_value = dot_4_bits >> 3
        if dot_4_value >= 12:
            #b1100 =12 D4
            exponet = x & 30 #b00011110 = 30
            exponet_int = exponet >> 1
            if exponet_int >= 8:
                #b1000 = 8
                exponet_value = -exponet_int
            else:
                exponet_value = exponet_int + 8

            fra_int = x & 1 #b00000001
            m_value = 1.0 + fra_int * 0.5
            #print(f"dot_4_value:{dot_4_value} exponet_value:{exponet_value} fra_int:{fra_int} value:{sign * pow(2, exponet_value) * (1 + fra_int / pow(2, 1))}")
        elif dot_4_value >= 8:
            #b1000 =8 D3
            exponet = x & 28 #b00011100 = 28
            exponet_int = exponet >> 2
            if exponet_int >= 4:
                #b100 = 4
                exponet_value = -exponet_int
            else:
                exponet_value = exponet_int + 4
            fra_int = x & 3 #b00000011
            m_value = 1.0 + fra_int * 0.25
            #print(f"dot_4_value:{dot_4_value} exponet_int:{exponet_int} exponet_value:{exponet_value} fra_int:{fra_int}")
        elif dot_4_value >= 4:
            #b0100 =8 D2
            exponet = x & 24  # b00011000 = 24
            exponet_int = exponet >> 3
            if exponet_int >= 2:
                # b10 = 2
                exponet_value = -exponet_int
            else:
                exponet_value = exponet_int + 2
            fra_int = x & 7  # b00000111
            m_value = 1.0 + fra_int * 0.125
        elif dot_4_value >= 2:
            #b0010 =2 D1
            exponet = x & 8 # b00001000 = 8
            exponet_sign = exponet >> 3
            if exponet_sign >= 1:
                # b10 = 2
                exponet_value = -1
            else:
                exponet_value = 1
            fra_int = x & 7  # b00000111
            m_value = 1.0 + fra_int * 0.125
            #print(f"dot_4_value:{dot_4_value} exponet_value:{exponet_value} fra_int:{fra_int} value:{sign * pow(2, exponet_value) * (1 + fra_int / pow(2, 3))}")
        elif dot_4_value == 1:
            #d0
            exponet_value = 0
            fra_int = x & 7  # b00000111
            m_value = 1.0 + fra_int * 0.125
            #print(f"dot_4_value:{dot_4_value} exponet_value:{exponet_value} fra_int:{fra_int} value:{sign*pow(2,exponet_value)*(1+fra_int/pow(2,3))}")
        elif dot_4_value == 0:
            #dml
            m_value = 1
            exponet_value = (x & 7) - 23  # b00000111 = 7
            #print(f"dml exponet_value:{x & 7 - 23} x:{x}")
        else:
            print("error,dot error")
            m_value = 0.0
            exponet_value = 0
        return sign*pow(2.0, exponet_value)*m_value


def concat_tensor(tensor1, shape1, tensor2, shape2, n, tnd_flag=False):
    if len(shape1) != len(shape2):
        print(f"[ERROR]concat_tensor: 相加的两个tensor 维数不同! shape1 = {shape1}, shape2 =  {shape2}")
        return None, None
    elif len(shape1) == 2:
        if shape1[0] != shape2[0]:
            print(f"[ERROR]concat_tensor: 相加的两个tensor 维度非法! shape1 = {shape1}, shape2 =  {shape2}")
            return None, None
        d1 = int(shape1[1] / n)
        d2 = int(shape2[1] / n)
        tensor1_1nd = tensor1.reshape(shape1[0], n, d1)
        tensor2_1nd = tensor2.reshape(shape2[0], n, d2)
        concatenated_tensor = np.concatenate((tensor1_1nd, tensor2_1nd), axis=2)
        concatenated_shape = [shape1[0], shape1[1] + shape2[1]]
        concatenated_tensor = concatenated_tensor.reshape(concatenated_shape)
    elif len(shape1) == 3:
        if shape1[0] != shape2[0] or shape1[1] != shape2[1]:
            print(f"[ERROR]concat_tensor: 相加的两个tensor 维度非法! shape1 = {shape1}, shape2 =  {shape2}")
            return None, None
        if tnd_flag:
            concatenated_tensor = np.concatenate((tensor1, tensor2), axis=2)
            concatenated_shape = [shape1[0], shape1[1], shape1[2] + shape2[2]]
        else:
            d1 = int(shape1[2] / n)
            d2 = int(shape2[2] / n)
            tensor1_bsnd = tensor1.reshape(shape1[0], shape1[1], n, d1)
            tensor2_bsnd = tensor2.reshape(shape2[0], shape2[1], n, d2)
            concatenated_tensor = np.concatenate((tensor1_bsnd, tensor2_bsnd), axis=3)
            concatenated_shape = [shape1[0], shape1[1], shape1[2] + shape2[2]]
            concatenated_tensor = concatenated_tensor.reshape(concatenated_shape)
    elif len(shape1) == 4:
        if shape1[0] != shape2[0] or shape1[1] != shape2[1] or shape1[2] != shape2[2]:
            print(f"[ERROR]concat_tensor: 相加的两个tensor 维度非法! shape1 = {shape1}, shape2 =  {shape2}")
            return None, None
        concatenated_tensor = np.concatenate((tensor1, tensor2), axis=3)
        concatenated_shape = [shape1[0], shape1[1], shape1[2], shape1[3] + shape2[3]]
    else:
        print(f"[ERROR]concat_tensor: shape1 维度非法! shape1 = {shape1}")
        return None, None
    return concatenated_tensor, concatenated_shape


def preprocessing(fa_param):
    params = fa_param['params']
    tensor = fa_param['tensor']
    flags = fa_param['flags']
    numHeads = fa_param['numHeads']
    numKeyValueHeads = fa_param['numKeyValueHeads']
    
    # 备份拼接前的信息
    fa_param['tensor_old'] = {
        'query': copy.deepcopy(tensor['query']),
        'key': copy.deepcopy(tensor['key']),
    }
    fa_param['shape_old'] = {
        'query': copy.deepcopy(params['shape_input']['query']),
        'key': copy.deepcopy(params['shape_input']['key']),
    }

    q_new_tensor, q_new_shape = concat_tensor(tensor['query'], params['shape_input']['query'],
                                              tensor['query_rope'], params['shape_input']['query_rope'], 
                                              numHeads, flags['tnd'])
    if q_new_tensor is not None:
        tensor['query'] = q_new_tensor
        params['shape_input']['query'] = q_new_shape
    else:
        print("[ERROR]q tensor的预处理异常，输出空tensor！")

    key = tensor['key']
    value = tensor['key']
    k_dtype = trans_input_dtype(params['dtype_input']['key'])
    if k_dtype == "hifloat8":
        key = trans_np_hifuint8_tensor_to_float32(tensor['key'])
        value = trans_np_hifuint8_tensor_to_float32(tensor['key'])
    elif k_dtype == "float8_e4m3fn":
        key = tensor['key'].astype(np.float32)
        value = tensor['key'].astype(np.float32)
    if value is not None:
        tensor['key'] = value
    else:
        print("[ERROR]k tensor的预处理异常，输出空tensor！")

    k_new_tensor, k_new_shape = concat_tensor(key, params['shape_input']['key'],
                                              tensor['key_rope'], params['shape_input']['key_rope'], 
                                              numKeyValueHeads, flags['kv_tnd'])
    if k_new_tensor is not None:
        tensor['key'] = k_new_tensor
        params['shape_input']['key'] = k_new_shape
    else:
        print("[ERROR]k tensor的预处理异常，输出空tensor！")

    return fa_param


def _n_trans_shape_to_bnsd(tensor, shape, layout, headnums=None, act_seq=None, tensor_name=None):
    if layout in ["BSH", "BSH_NBSD"]:
        if headnums is None:
            return tensor, shape
        B = shape[0]
        S = shape[1]
        if tensor_name != 'sparseIndices':
            H = shape[2]
            N = headnums
            D = H // N
            tensor = tensor.reshape(B, S, N, D).transpose(0, 2, 1, 3)
            return tensor, [B, N, S, D]
        else:
            tensor = tensor.transpose(0, 2, 1, 3)
            return tensor, [B, headnums, S, shape[3]]
    elif layout in ["BSND", "PA_BSND"]:
        B = shape[0]
        S = shape[1]
        N = shape[2]
        D = shape[3]
        tensor = tensor.transpose(0, 2, 1, 3)
        return tensor, [B, N, S, D]
    elif layout in ["TND", "TND_NTD"]:
        T = shape[0]
        N = shape[1]
        D = shape[2]
        B = len(act_seq)
        S = max(act_seq)
        new_tensor = np.zeros((B, N, S, D), dtype=tensor.dtype)
        t_start = 0
        for b_index in range(B):
            act_s = act_seq[b_index]
            t_end = t_start + act_s

            # print(f"[TEMP]b_index:{b_index} act_s:{act_s}, t_start:{t_start}, t_end:{t_end}")
            if act_s == 0:
                continue
            for n_index in range(N):
                new_tensor[b_index, n_index, 0:act_s, :] = tensor[t_start:t_end, n_index, :]
            t_start += act_s
        # print(f"[TEMP]new_tensor:{new_tensor}")
        return new_tensor, [B, N, S, D]
    else:
        return tensor, shape


def trans_tnd_actseq(list):
    list_len = len(list)
    list_new = []
    list_new.append(list[0])
    for i in range(list_len - 1):
        new_item = list[i + 1] - list[i]
        if new_item >= 0:
            list_new.append(new_item)
        else:
            print(f"[ERROR]trans_tnd_actseq: Wrong input actseq:{list}, in loop {i}, item {new_item} < 0")
    print(f"[INFO]before trans: {list}")
    print(f"[INFO]after trans: {list_new}")
    return list_new


def trans_bnsd_to_layout(tensor, shape, layout, act_q=None):
    if layout == "BSH":
        output = tensor.permute(0, 2, 1, 3).contiguous().view(shape)
        return output
    elif layout in ["BSND", "PA_BSND"]:
        output = tensor.permute(0, 2, 1, 3).contiguous()
        return output
    elif layout in ["BSND_NBSD", "BNSD_NBSD", "BSH_NBSD"]:
        output = tensor.permute(1, 0, 2, 3).contiguous()
        return output
    elif layout in ["TND", "TND_NTD"]:
        T = sum(act_q)
        B = tensor.shape[0]
        N = tensor.shape[1]
        D = tensor.shape[3]
        output = torch.zeros(size=(T, N, D))
        # output = torch.zeros(size=(T, N, D), dtype=tensor.dtype)
        t_start = 0
        for b_index in range(B):
            act_s = act_q[b_index]
            t_end = t_start + act_s

            if act_s == 0:
                continue
            for n_index in range(N):
                slice_data = tensor[b_index, n_index, :act_s, :]
                if isinstance(slice_data, np.ndarray):
                    output[t_start:t_end, n_index, :] = torch.from_numpy(slice_data)
                else:
                    output[t_start:t_end, n_index, :] = slice_data
            t_start += act_s
        if layout == "TND_NTD":
            output = output.permute(1, 0, 2).contiguous()
        return output
    else:
        return tensor
 

def trans_input_dtype(input_dtype):
    if input_dtype == 'fp16':
        return 'float16'
    elif input_dtype == 'int8':
        return 'int8'
    elif input_dtype == 'uint8':
        return 'uint8'
    elif input_dtype == 'bf16':
        return 'bfloat16'
    elif input_dtype == 'bool':
        return 'bool'
    elif input_dtype == 'int32':
        return 'int32'
    elif input_dtype == 'fp32':
        return 'float32'
    elif input_dtype == 'int4':
        return 'int4'
    else:
        return input_dtype


# 融合表格参数获取函数
def get_param_fus(params):
    fa_param = {
        'params': params,
        'flags': {
            'normal': True,
            'pa': False,
            'tnd': False,
            'kv_tnd': False,
        },
    }
    
    # ===tensor数据===
    tensor_keys = ['query', 'key', 'sparse_indices', 'block_table', 'query_rope', 'key_rope', 'dequant_scale']
    fa_param['tensor'] = {}
    for key in tensor_keys:
        fa_param['tensor'][key] = generate_tensor_data.gen_tensor_data(params, key, None)
    
    # ===派生参数缓存===
    shape_input = params['shape_input']
    dtype_input = params['dtype_input']
    
    fa_param['numHeads'] = shape_input['query'][-2]
    fa_param['numKeyValueHeads'] = shape_input['key'][-2]
    fa_param['sparse_blockcount'] = shape_input['sparse_indices'][-1]
    fa_param['sparse_blocksize'] = params['sparse_blocksize']
    fa_param['blocksize'] = 1
    if params['layout_kv'] in ["PA_BSND"]:
        fa_param['blocksize'] = shape_input['key_cache'][1]
    
    fa_param['flags']['pa'] = params['layout_kv'] in ["PA_BSND"]
    fa_param['flags']['tnd'] = params['layout_query'] in ["TND"]
    fa_param['flags']['kv_tnd'] = params['layout_kv'] in ["TND"]
    
    # ===异常判断===
    if 0 in shape_input['query']:
        fa_param['flags']['normal'] = False
        print('[WARNING]:异常-->  q为空场景！')
    if len(shape_input['key']) == 0 or 0 in shape_input['key']:
        fa_param['flags']['normal'] = False
        print('[WARNING]:异常-->  k为空场景！')
    if len(shape_input['value']) == 0 or 0 in shape_input['value']:
        fa_param['flags']['normal'] = False
        print('[WARNING]:异常-->  v为空场景！')

    return fa_param


def kv_concat_nopa_preprocessing(fa_param, json_data):
    params = fa_param['params']
    tensor = fa_param['tensor']
    tensor_old = fa_param['tensor_old']
    
    q_tensor = tensor['query']
    k_tensor = tensor_old['key']
    k_rope_tensor = tensor['key_rope'].astype(trans_input_dtype(params['dtype_input']['key_rope']))
    k_dequant_scale_tensor = tensor['dequant_scale']
    kv_layout = params['layout_kv']
    
    if kv_layout != "PA_BSND":
        k_dtype = trans_input_dtype(params['dtype_input']['key'])
        if k_dtype == "int8":
            key_concat = np.concatenate((k_tensor, k_rope_tensor.view(np.int8), k_dequant_scale_tensor.view(np.int8)),axis=-1)
        elif k_dtype == "float8_e4m3fn":
            key_concat = np.concatenate((k_tensor, k_rope_tensor.view(float8_e4m3fn), k_dequant_scale_tensor.view(float8_e4m3fn)),axis=-1)
        elif k_dtype == "hifloat8":
            key_concat = np.concatenate((k_tensor, k_rope_tensor.view(np.uint8), k_dequant_scale_tensor.view(np.uint8)),axis=-1)

        tensor['query_cache'] = tools.modify_alcnn_input_file(name='query_cache',
                                      tensors=q_tensor,
                                      params=fa_param,
                                      json_data=json_data,
                                      data_dtype=params['dtype_input']['query'])
        tensor['key_cache'] = tools.modify_alcnn_input_file(name='key_cache',
                                      tensors=key_concat,
                                      params=fa_param,
                                      json_data=json_data,
                                      data_dtype=params['dtype_input']['key'])
        tensor['value_cache'] = tools.modify_alcnn_input_file(name='value_cache',
                                      tensors=key_concat,
                                      params=fa_param,
                                      json_data=json_data,
                                      data_dtype=params['dtype_input']['value'])
    else:
        print(f"[ERROR]Wrong input layout_kv: {kv_layout}")


def aclnn_op_func_cpu(params, json_data):
    print("cpu执行中...")
    fa_param = get_param_fus(params)
    
    if not fa_param['flags']['normal']:
        out_shape = params['shape_input']['query']
        return fa_param
    
    # ==========预处理=============
    # >> 1.参数预处理
    actualSeqLengths_q = params['actualseqlengths']
    actualSeqLengths_kv = params['actualseqlengthskv']
    layoutQuery = params['layout_query']
    layout_kv = params['layout_kv']
    numHeads = fa_param['numHeads']
    numKeyValueHeads = fa_param['numKeyValueHeads']
    out_shape = params['shape_input']['query']
    k_ori_shape = copy.deepcopy(params['shape_input']['key'])
    v_ori_shape = copy.deepcopy(params['shape_input']['value'])
    k_dtype = trans_input_dtype(params['dtype_input']['key'])
    v_dtype = trans_input_dtype(params['dtype_input']['value'])
    sparse_mode = params['sparsemode']

    # >> 2.将q与qRope拼接 将k与kRope拼接
    fa_param = preprocessing(fa_param)

    # >> 3.将q、k、v、topkIndices转换为bnsd
    tensor = fa_param['tensor']
    shape_input = params['shape_input']
    
    if fa_param['flags']['tnd']:
        fa_param['actualSeqLengths_q'] = trans_tnd_actseq(actualSeqLengths_q)
        if fa_param['flags']['pa']:
            fa_param['actualSeqLengths_kv'] = actualSeqLengths_kv
        q_bnsd_tensor, q_bnsd_shape = _n_trans_shape_to_bnsd(tensor['query'], shape_input['query'], layoutQuery,
                                                             numHeads, fa_param['actualSeqLengths_q'])
    else:
        q_bnsd_tensor, q_bnsd_shape = _n_trans_shape_to_bnsd(tensor['query'], shape_input['query'], layoutQuery,
                                                             numHeads, actualSeqLengths_q)
    if fa_param['flags']['kv_tnd']:
        fa_param['actualSeqLengths_kv'] = trans_tnd_actseq(actualSeqLengths_kv)
        k_bnsd_tensor, k_bnsd_shape = _n_trans_shape_to_bnsd(tensor['key'], shape_input['key'],
                                                             layout_kv, numKeyValueHeads, fa_param['actualSeqLengths_kv'])
        v_bnsd_tensor, v_bnsd_shape = k_bnsd_tensor, k_bnsd_shape
        k_dequant_scale_bnsd_tensor, k_dequant_scale_bnsd_shape = _n_trans_shape_to_bnsd(
            tensor['dequant_scale'], shape_input['dequant_scale'],
            layout_kv, numKeyValueHeads, fa_param['actualSeqLengths_kv'])
        v_dequant_scale_bnsd_tensor, v_dequant_scale_bnsd_shape = k_dequant_scale_bnsd_tensor, k_dequant_scale_bnsd_shape
    else:
        k_bnsd_tensor, k_bnsd_shape = _n_trans_shape_to_bnsd(tensor['key'], shape_input['key'],
                                                             layout_kv, numKeyValueHeads, actualSeqLengths_kv)
        v_bnsd_tensor, v_bnsd_shape = k_bnsd_tensor, k_bnsd_shape
        k_dequant_scale_bnsd_tensor, k_dequant_scale_bnsd_shape = _n_trans_shape_to_bnsd(
            tensor['dequant_scale'], shape_input['dequant_scale'],
            layout_kv, numKeyValueHeads, actualSeqLengths_kv)
        v_dequant_scale_bnsd_tensor, v_dequant_scale_bnsd_shape = k_dequant_scale_bnsd_tensor, k_dequant_scale_bnsd_shape
    if fa_param['flags']['tnd']:
        sparse_indices_bnsd_tensor, sparse_indices_bnsd_shape = _n_trans_shape_to_bnsd(
            tensor['sparse_indices'], shape_input['sparse_indices'], layoutQuery,
            numKeyValueHeads, fa_param['actualSeqLengths_q'], 'sparseIndices')
    else:
        sparse_indices_bnsd_tensor, sparse_indices_bnsd_shape = _n_trans_shape_to_bnsd(
            tensor['sparse_indices'], shape_input['sparse_indices'], layoutQuery,
            numKeyValueHeads, actualSeqLengths_q, 'sparseIndices')

    # 中间计算结果存放在顶层
    fa_param['q_bnsd_tensor'] = q_bnsd_tensor
    fa_param['q_bnsd_shape'] = q_bnsd_shape
    fa_param['k_bnsd_tensor'] = k_bnsd_tensor
    fa_param['k_bnsd_shape'] = k_bnsd_shape
    fa_param['v_bnsd_tensor'] = v_bnsd_tensor
    fa_param['v_bnsd_shape'] = v_bnsd_shape
    fa_param['k_dequant_scale_bnsd_tensor'] = k_dequant_scale_bnsd_tensor
    fa_param['k_dequant_scale_bnsd_shape'] = k_dequant_scale_bnsd_shape
    fa_param['v_dequant_scale_bnsd_tensor'] = v_dequant_scale_bnsd_tensor
    fa_param['v_dequant_scale_bnsd_shape'] = v_dequant_scale_bnsd_shape
    fa_param['sparse_indices_bnsd_tensor'] = sparse_indices_bnsd_tensor
    fa_param['sparse_indices_bnsd_shape'] = sparse_indices_bnsd_shape
    fa_param['ks_max'] = 0
    
    # >> 4.act seq处理
    if not fa_param['flags']['tnd']:
        if len(actualSeqLengths_q) == 0:
            fa_param['actualSeqLengths_q'] = [q_bnsd_shape[2]] * q_bnsd_shape[0]
        elif len(actualSeqLengths_q) == 1:
            fa_param['actualSeqLengths_q'] = [actualSeqLengths_q[0]] * q_bnsd_shape[0]
        else:
            fa_param['actualSeqLengths_q'] = actualSeqLengths_q
    if not fa_param['flags']['kv_tnd']:
        if len(actualSeqLengths_kv) == 0:
            fa_param['actualSeqLengths_kv'] = [k_bnsd_shape[2]] * k_bnsd_shape[0]
        elif len(actualSeqLengths_kv) == 1:
            fa_param['actualSeqLengths_kv'] = [actualSeqLengths_kv[0]] * k_bnsd_shape[0]
        else:
            fa_param['actualSeqLengths_kv'] = actualSeqLengths_kv

    # >> 5.生成sparseIndices
    if params["action_type"] != "bm_output_gold":
        seed = int(time.time())
        sparse_indices_bnsd_shape = fa_param['sparse_indices_bnsd_shape']
        sparse_indices_tensor = fa_param['sparse_indices_bnsd_tensor']

        for b in range(sparse_indices_bnsd_shape[0]):
            act_seqlen_kv = fa_param['actualSeqLengths_kv'][b]
            act_seqlen_q = fa_param['actualSeqLengths_q'][b]

            for n in range(sparse_indices_bnsd_shape[1]):
                for s in range(sparse_indices_bnsd_shape[2]):
                    if sparse_mode == 0:
                        threshold = act_seqlen_kv
                    elif sparse_mode == 3:
                        threshold = act_seqlen_kv - act_seqlen_q + s + 1

                    if threshold <= 0:
                        sparse_indices_tensor[b, n, s, :] = [-1] * (fa_param['sparse_blockcount'])
                        continue
                    valid_blocks_max = math.ceil(max(0, threshold) / fa_param['sparse_blocksize'])
                    dtype = getattr(np, trans_input_dtype(params['dtype_input']['sparse_indices']))
                    block_indices = torch.randperm(valid_blocks_max - 1).numpy().astype(dtype)
                    valid_blocks_topk = min(valid_blocks_max, fa_param['sparse_blockcount'])
                    sparse_indices_tensor[b, n, s, :valid_blocks_topk - 1] = block_indices[0: valid_blocks_topk - 1]
                    sparse_indices_tensor[b, n, s, valid_blocks_topk - 1:valid_blocks_topk] = valid_blocks_max - 1
                    sparse_indices_tensor[b, n, s, valid_blocks_topk:] = [-1] * (
                                fa_param['sparse_blockcount'] - valid_blocks_topk)

        # 打印信息：kv首地址，topk值
        if fa_param['flags']['tnd']:
            sparse_indices_tensor_tnd = trans_bnsd_to_layout(sparse_indices_tensor, out_shape, layoutQuery,
                                                             fa_param['actualSeqLengths_q'])
            tensor['sparse_indices'] = tools.modify_alcnn_input_file(name='sparse_indices',
                                          tensors=torch.tensor(sparse_indices_tensor_tnd, dtype=torch.int32),
                                          params=fa_param,
                                          json_data=json_data,
                                          data_dtype=params['dtype_input']['sparse_indices'])

        else:
            tensor['sparse_indices'] = tools.modify_alcnn_input_file(name='sparse_indices',
                                          tensors=sparse_indices_tensor.transpose(0, 2, 1, 3),
                                          params=fa_param,
                                          json_data=json_data,
                                          data_dtype=params['dtype_input']['sparse_indices'])

        fa_param['sparse_indices_bnsd_tensor'] = sparse_indices_tensor
        if fa_param['flags']['pa']:
            # 生成blocktable
            blockSize = fa_param['blocksize']
            blockTableShape = shape_input['block_table']
            blockNum = shape_input['key_cache'][0]
            if 0 in blockTableShape:
                print('[WARNING]:block_table为空场景，输出空tensor！')
                fa_param['cpu_result'] = torch.zeros(out_shape)
                return fa_param
            blockNumPerBlock = []
            block_num_min = 0
            for actual_seq in actualSeqLengths_kv:
                blockNumPerBlock.append(math.ceil(actual_seq / blockSize))
                block_num_min += math.ceil(actual_seq / blockSize)
            if block_num_min > blockNum:
                print(f"[ERROR]Wrong input k_cache_shape: get blockNum = {blockNum}, but expect blockNum > {block_num_min}")
                exit(1)
            block_idx_list = np.arange(0, blockNum, 1)
            block_idx_list = np.random.permutation(block_idx_list).astype(np.int32)
            block_idx = 0
            block_table = [-1] * blockTableShape[1]
            block_table = np.tile(block_table, (blockTableShape[0], 1)).astype(np.int32)
            block_table_batch_idx = 0
            for idx in blockNumPerBlock:
                for j in range(idx):
                    block_table[block_table_batch_idx][j] = (block_idx_list[block_idx])
                    block_idx += 1
                block_table_batch_idx += 1
            fa_param['block_table'] = block_table
            # 将block_table 覆盖原有的input
            tensor['block_table'] = tools.modify_alcnn_input_file(name='block_table',
                                          tensors=torch.tensor(block_table, dtype=torch.int32),
                                          params=fa_param,
                                          json_data=json_data)
            k_np_dtype = tools.get_np_dtype(params['dtype_input']['key'])
            v_np_dtype = tools.get_np_dtype(params['dtype_input']['value'])
            fa_param['k_np_dtype'] = k_np_dtype
            fa_param['v_np_dtype'] = v_np_dtype
            print(f"[PageAtten]Input Kdtype:{params['dtype_input']['key']} Vdtype:{params['dtype_input']['value']}")
            k_cache = np.zeros(shape_input['key_cache']).astype(k_np_dtype)
            v_cache = np.zeros(shape_input['value_cache']).astype(v_np_dtype)
            print('[INFO]:PA + Deepseek！')
            deepseek_ds_pa_preprocessing(fa_param, json_data)
        else:
            kv_concat_nopa_preprocessing(fa_param, json_data)

    # ===计算入口===
    y_all = _t_increattention_bnsd(fa_param)
    y_all = torch.from_numpy(y_all)
    y_all = trans_bnsd_to_layout(y_all, out_shape, layoutQuery, fa_param['actualSeqLengths_q'])
    
    # 存储CPU计算结果
    fa_param['cpu_result'] = y_all
    return fa_param


def gatherKV(k_tensor, v_tensor, sparse_indices_Indices, sparse_blocksize, sparse_blockcount, batch, n2Idx, s1Idx,
             curr_actualSeq, curr_actualSeq_q, sparse_mode):
    s2_sparse = list()
    if sparse_mode == 0:
        threshold = curr_actualSeq
    elif sparse_mode == 3:
        delta_s = curr_actualSeq - curr_actualSeq_q
        threshold = delta_s + s1Idx + 1
    validCount = min(sparse_blockcount, math.ceil(threshold / sparse_blocksize))
    for i in range(validCount):
        sparseIndicesId = sparse_indices_Indices[i]

        if sparseIndicesId == -1:
            break
        begin_Idx = sparseIndicesId * sparse_blocksize
        end_Idx = begin_Idx + sparse_blocksize if begin_Idx + sparse_blocksize <= curr_actualSeq else curr_actualSeq
        if begin_Idx >= threshold:
            continue
        if end_Idx <= threshold:
            s2_sparse.extend(np.arange(begin_Idx, end_Idx))
        else:
            s2_sparse.extend(np.arange(begin_Idx, threshold))

    emptyFlag = False
    if len(s2_sparse) == 0:
        k_sparse, v_sparse = [], []
        emptyFlag = True
    else:
        k_sparse, v_sparse = k_tensor[batch, n2Idx, s2_sparse, :], v_tensor[batch, n2Idx, s2_sparse, :]

    return emptyFlag, k_sparse, v_sparse


def softmax(x):
    x = x.astype(np.float32)
    x_max = x.max(axis=-1, keepdims=True)
    x_sub = x - x_max
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    ans = y / x_sum
    return ans


def _t_increattention_bnsd(fa_param):
    params = fa_param['params']
    batch_size = len(params['actualseqlengths'])
    numheads = fa_param['numHeads']
    numKeyValueHeads = fa_param['numKeyValueHeads']
    actualSeqLengths_q = fa_param['actualSeqLengths_q']
    actualSeqLengths_kv = fa_param['actualSeqLengths_kv']
    scaleValue = params['scalevalue']
    sparse_blocksize = fa_param['sparse_blocksize']
    sparse_blockcount = fa_param['sparse_blockcount']
    sparseIndicesIndices = fa_param['sparse_indices_bnsd_tensor']
    out_shape_bnsd = fa_param['q_bnsd_shape']
    out_shape_bnsd[-1] = fa_param['v_bnsd_shape'][-1]
    sparse_mode = params['sparsemode']
    g = numheads // numKeyValueHeads
    q_dtype = trans_input_dtype(params['dtype_input']['query'])

    q_bnsd_tensor = fa_param['q_bnsd_tensor']
    k_bnsd_tensor = fa_param['k_bnsd_tensor'].astype(np.float32)
    k_dequant_scale_bnsd_tensor = fa_param['k_dequant_scale_bnsd_tensor']
    v_bnsd_tensor = fa_param['v_bnsd_tensor'].astype(np.float32)
    v_dequant_scale_bnsd_tensor = fa_param['v_dequant_scale_bnsd_tensor']
    tile_size = 128
    if q_dtype == "float16":
        k_bnsd_tensor[..., :512] = (k_bnsd_tensor[..., :512] * np.repeat(k_dequant_scale_bnsd_tensor, tile_size,
                                                                        axis=-1)).astype(np.float16)
        v_bnsd_tensor[..., :512] = (v_bnsd_tensor[..., :512] * np.repeat(v_dequant_scale_bnsd_tensor, tile_size,
                                                                        axis=-1)).astype(np.float16)
    elif q_dtype == "bfloat16":
        k_bnsd_tensor[..., :512] = (k_bnsd_tensor[..., :512] * np.repeat(k_dequant_scale_bnsd_tensor, tile_size,
                                                                        axis=-1)).astype(tf.bfloat16.as_numpy_dtype)
        v_bnsd_tensor[..., :512] = (v_bnsd_tensor[..., :512] * np.repeat(v_dequant_scale_bnsd_tensor, tile_size,
                                                                        axis=-1)).astype(tf.bfloat16.as_numpy_dtype)
    elif q_dtype == "float32":
        k_bnsd_tensor[..., :512] = (k_bnsd_tensor[..., :512] * np.repeat(k_dequant_scale_bnsd_tensor, tile_size,
                                                                        axis=-1)).astype(np.float32)
        v_bnsd_tensor[..., :512] = (v_bnsd_tensor[..., :512] * np.repeat(v_dequant_scale_bnsd_tensor, tile_size,
                                                                        axis=-1)).astype(np.float32)
    matmul_dtype = np.float32
    y = np.zeros(out_shape_bnsd, dtype = np.float32)

    for batch in range(batch_size):
        curr_actualSeq_q = actualSeqLengths_q[batch]
        curr_actualSeq = actualSeqLengths_kv[batch]
        for n2Idx in range(numKeyValueHeads):
            for s1Idx in range(curr_actualSeq_q):
                if s1Idx < curr_actualSeq_q - curr_actualSeq and sparse_mode != 0:
                    y[batch, n2Idx * g: (n2Idx + 1) * g, s1Idx, :] = np.zeros([g, out_shape_bnsd[-1]], dtype=np.float32)
                    continue
                q_curr = q_bnsd_tensor[batch, n2Idx * g: (n2Idx + 1) * g, s1Idx, :]
                sparse_indices_Indices = sparseIndicesIndices[batch, n2Idx, s1Idx, :]

                emptyFlag, k_sparse, v_sparse = gatherKV(k_bnsd_tensor, v_bnsd_tensor, sparse_indices_Indices,
                                                         sparse_blocksize, sparse_blockcount, batch, n2Idx, s1Idx,
                                                         curr_actualSeq, curr_actualSeq_q, sparse_mode)
                if emptyFlag:
                    continue
                bmm1Res = np.matmul(q_curr, k_sparse.T, dtype=matmul_dtype)
                scaleRes = bmm1Res * scaleValue
                softmax_res = softmax(scaleRes)
                if q_dtype == "float16":
                    bmm2Res = np.matmul(softmax_res.astype(np.float16).astype(np.float32), v_sparse, dtype=matmul_dtype)
                elif q_dtype == "bfloat16":
                    bmm2Res = np.matmul(softmax_res.astype(tf.bfloat16.as_numpy_dtype).astype(np.float32), v_sparse,
                                        dtype=matmul_dtype)
                else:
                    bmm2Res = np.matmul(softmax_res, v_sparse, dtype=matmul_dtype)
                y[batch, n2Idx * g: (n2Idx + 1) * g, s1Idx, :] = bmm2Res
    return y


def deepseek_ds_pa_preprocessing(fa_param, json_data):  # concat + 生成kv_cache
    params = fa_param['params']
    tensor = fa_param['tensor']
    tensor_old = fa_param['tensor_old']
    shape_old = fa_param['shape_old']
    
    k_np_dtype = fa_param['k_np_dtype']
    v_np_dtype = fa_param['v_np_dtype']
    k_ori_shape_list = shape_old['key']
    numKeyValueHeads = fa_param['numKeyValueHeads']
    numHeads = fa_param['numHeads']
    kv_layout = params['layout_kv']
    actualSeqLengths_kv = fa_param['actualSeqLengths_kv']
    actualSeqLengths_q = fa_param['actualSeqLengths_q']
    blockSize = fa_param['blocksize']
    blockTableShape = params['shape_input']['block_table']
    block_table = fa_param['block_table']
    layoutQuery = params['layout_query']
    layout_kv = params['layout_kv']
    k_cache_shape = params['shape_input']['key_cache'].copy()
    k_cache_shape[-1] = 512
    k_cache = np.zeros(k_cache_shape).astype(k_np_dtype)
    v_cache = np.zeros(k_cache_shape).astype(v_np_dtype)
    k_rope_cache_shape = params['shape_input']['key_cache'].copy()
    k_rope_cache_shape[-1] = 64
    k_dequant_scale_cache_shape = params['shape_input']['key_cache'].copy()
    k_dequant_scale_cache_shape[-1] = 512 // 128
    k_rope_cache = np.zeros(k_rope_cache_shape).astype(trans_input_dtype(params['dtype_input']['key_rope']))
    k_dequant_scale_cache = np.zeros(k_dequant_scale_cache_shape).astype(trans_input_dtype(params['dtype_input']['dequant_scale']))
    k_dtype = trans_input_dtype(params['dtype_input']['key'])
    q_tensor = tensor['query']
    q_rope_tensor = tensor['query_rope']

    B = len(fa_param['actualSeqLengths_kv'])

    if layout_kv in ["PA_BSND"]:
        k_tensor_bsnd = tensor_old['key']
        if k_tensor_bsnd.shape[1] != blockTableShape[1] * blockSize:
            k_tensor_old = k_tensor_bsnd
            k_tensor_bsnd = np.zeros(
                (k_tensor_old.shape[0], blockTableShape[1] * blockSize, k_tensor_old.shape[2], k_tensor_old.shape[3]),
                dtype=k_np_dtype
            )
            k_tensor_bsnd[:, :k_tensor_old.shape[1], :, :] = k_tensor_old
        kvs = k_tensor_bsnd.shape[1]
        kvn = k_tensor_bsnd.shape[2]
        kvd = k_tensor_bsnd.shape[3]

        for b in range(B):
            for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                block_offset = block_i * blockSize
                if kv_cache_blk_id == -1:
                    continue
                else:
                    for n in range(params['shape_input']['key_cache'][2]):
                        k_cache[kv_cache_blk_id, 0:blockSize, n:n + 1, :] = k_tensor_bsnd[b,
                                                                            block_offset:(block_offset + blockSize), n:n + 1, :]

        v_dequant_scale_bsnd = tensor['dequant_scale']

        k_rope_bsnd = tensor['key_rope']
        if k_rope_bsnd.shape[1] != blockTableShape[1] * blockSize:
            k_rope_tensor_old = k_rope_bsnd
            k_rope_bsnd = np.zeros(
                (k_rope_tensor_old.shape[0], blockTableShape[1] * blockSize, k_rope_tensor_old.shape[2],
                 k_rope_tensor_old.shape[3]),
                dtype=trans_input_dtype(params['dtype_input']['key_rope'])
            )
            k_rope_bsnd[:, :k_rope_tensor_old.shape[1], :, :] = k_rope_tensor_old
        k_dequant_scale_bsnd = tensor['dequant_scale']
        if k_dequant_scale_bsnd.shape[1] != blockTableShape[1] * blockSize:
            k_dequant_scale_tensor_old = k_dequant_scale_bsnd
            k_dequant_scale_bsnd = np.zeros(
                (k_dequant_scale_tensor_old.shape[0], blockTableShape[1] * blockSize,
                 k_dequant_scale_tensor_old.shape[2], k_dequant_scale_tensor_old.shape[3]),
                dtype=trans_input_dtype(params['dtype_input']['dequant_scale'])
            )
            k_dequant_scale_bsnd[:, :k_dequant_scale_tensor_old.shape[1], :, :] = k_dequant_scale_tensor_old
        v_dequant_scale_bsnd = tensor['dequant_scale']
        if v_dequant_scale_bsnd.shape[1] != blockTableShape[1] * blockSize:
            v_dequant_scale_tensor_old = v_dequant_scale_bsnd
            v_dequant_scale_bsnd = np.zeros(
                (v_dequant_scale_tensor_old.shape[0], blockTableShape[1] * blockSize,
                 v_dequant_scale_tensor_old.shape[2], v_dequant_scale_tensor_old.shape[3]),
                dtype=trans_input_dtype(params['dtype_input']['dequant_scale'])
            )
            v_dequant_scale_bsnd[:, :v_dequant_scale_tensor_old.shape[1], :, :] = v_dequant_scale_tensor_old
        kropes = k_rope_bsnd.shape[1]
        kropen = k_rope_bsnd.shape[2]
        kroped = k_rope_bsnd.shape[3]

        for b in range(B):
            for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                block_offset = block_i * blockSize
                if kv_cache_blk_id == -1:
                    continue
                else:
                    for n in range(params['shape_input']['key_rope'][2]):
                        k_rope_cache[kv_cache_blk_id, 0:blockSize, n:n + 1, :] = k_rope_bsnd[b,
                                                                                 block_offset:(block_offset + blockSize),
                                                                                 n:n + 1, :]
                    for n in range(params['shape_input']['dequant_scale'][2]):
                        k_dequant_scale_cache[kv_cache_blk_id, 0:blockSize, n:n + 1, :] = k_dequant_scale_bsnd[b,
                                                                                          block_offset:(block_offset + blockSize),
                                                                                          n:n + 1, :]
    else:
        print(f"[ERROR]Wrong input layout_kv: {kv_layout}")
    # 将kv cache 生成新的bin文件
    if k_dtype == "int8":
        k_cache_concat = np.concatenate((k_cache, k_rope_cache.view(np.int8), k_dequant_scale_cache.view(np.int8)), axis=-1)
    elif k_dtype == "float8_e4m3fn":
        k_cache_concat = np.concatenate((k_cache, k_rope_cache.view(float8_e4m3fn), k_dequant_scale_cache.view(float8_e4m3fn)), axis=-1)
    elif k_dtype == "hifloat8":
        k_cache_concat = np.concatenate((k_cache, k_rope_cache.view(np.uint8), k_dequant_scale_cache.view(np.uint8)), axis=-1)
    tensor['query_cache'] = tools.modify_alcnn_input_file(name='query_cache',
                                  tensors=q_tensor,
                                  params=fa_param,
                                  json_data=json_data,
                                  data_dtype=params['dtype_input']['query'])
    tensor['key_cache'] = tools.modify_alcnn_input_file(name='key_cache',
                                  tensors=k_cache_concat,
                                  params=fa_param,
                                  json_data=json_data,
                                  data_dtype=params['dtype_input']['key'])
    tensor['value_cache'] = tools.modify_alcnn_input_file(name='value_cache',
                                  tensors=k_cache_concat,
                                  params=fa_param,
                                  json_data=json_data,
                                  data_dtype=params['dtype_input']['value'])


def qtensor_seqlength(q_shape, inputLayout):
    if inputLayout == "SH":  # SH格式
        return q_shape[0]
    elif inputLayout == "BSH":
        return q_shape[1]
    elif inputLayout == "NSD":
        return q_shape[1]
    elif inputLayout == "BSND":
        return q_shape[1]
    else:  # BNSD
        return q_shape[2]


def qtensor_dim(q_shape, layoutQuery, numhead):
    if layoutQuery == "SH":  # SH格式
        return q_shape[0]
    elif layoutQuery == "BSH" or layoutQuery == "BSH_NBSD":
        return int(q_shape[2] / numhead)
    else:
        return q_shape[-1]


def aclnn_op_func(params, json_data):
    np.random.seed(0)  # 定位场景再打开

    action_type = params["action_type"]

    if action_type in ("bm"):
        device_type = "cpu"
    else:
        device_type = "npu"
    print("device_type", device_type)

    q_shape = params['shape_input']['query']
    layout_query = params['layout_query']

    numhead = q_shape[-2]
    qtensor_length = qtensor_seqlength(q_shape, layout_query)
    qdim = qtensor_dim(q_shape, layout_query, numhead)
    q_dtype = tools.get_np_dtype(params['dtype_input']['query'])
    kv_dtype = tools.get_np_dtype(params['dtype_input']['key'])
    print("fa", "*" * 30)
    print('params: ', params)
    return aclnn_op_func_cpu(params, json_data)
