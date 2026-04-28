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
import torch
import copy
import time

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
    if actual_seq is None:
        return [default_value] * B
    if len(actual_seq) != B:
        raise ValueError(f"{name} length should equal B={B}, but got {len(actual_seq)}")
    return [int(value) for value in actual_seq]


def _build_query_tensors(B, S1, N1, D, rope_head_dim, q_type):
    query_base = torch.empty((B, S1, N1, D), dtype=torch.float32).uniform_(
        DATA_RANGE_QUERY_LEFT, DATA_RANGE_QUERY_RIGHT).to(q_type)
    query_rope = torch.empty((B, S1, N1, rope_head_dim), dtype=torch.float32).uniform_(
        DATA_RANGE_QUERY_LEFT, DATA_RANGE_QUERY_RIGHT).to(q_type)
    return torch.cat((query_base, query_rope), dim=3)


def _build_kv_tensors(kv_dtype, kv_dim0, S2, N2, D, rope_head_dim, num_tiles, q_type):
    key_rope = torch.empty((kv_dim0, S2, N2, rope_head_dim), dtype=torch.float32).uniform_(
        DATA_RANGE_ROPE_LEFT, DATA_RANGE_ROPE_RIGHT).to(q_type)
    antiquant_scale = torch.empty((kv_dim0, S2, N2, num_tiles), dtype=torch.float32).uniform_(
        DATA_RANGE_SCALE_LEFT, DATA_RANGE_SCALE_RIGHT)

    if kv_dtype == "hifloat8":
        key_base = torch.randint(0, 256, (kv_dim0, S2, N2, D), dtype=torch.uint8)
        value = torch.randint(0, 256, (kv_dim0, S2, N2, D), dtype=torch.uint8)
        storage_dtype = torch.uint8
    elif kv_dtype in (None, "float8_e4m3fn"):
        key_base = torch.empty((kv_dim0, S2, N2, D), dtype=torch.float32).uniform_(
            DATA_RANGE_KV_LEFT, 10).to(torch.float8_e4m3fn)
        value = torch.empty((kv_dim0, S2, N2, D), dtype=torch.float32).uniform_(
            DATA_RANGE_KV_LEFT, 10).to(torch.float8_e4m3fn)
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
    max_blocks_per_batch = max(
        math.ceil(S2 / block_size),
        math.ceil(max(actual_seq_kv_list) / block_size),
    )
    required_blocks = sum(math.ceil(seq / block_size) for seq in actual_seq_kv_list)
    if required_blocks > block_num:
        raise ValueError(
            f"block_num={block_num} is not enough for PA, expect at least {required_blocks}"
        )

    all_indices = torch.randperm(block_num, dtype=torch.int32)
    block_table = torch.full((B, max_blocks_per_batch), -1, dtype=torch.int32)
    offset = 0
    for b in range(B):
        needed_blocks = math.ceil(actual_seq_kv_list[b] / block_size)
        block_table[b, :needed_blocks] = all_indices[offset:offset + needed_blocks]
        offset += needed_blocks
    return block_table


def _build_sparse_indices(B, S1, N2, K, sparse_block_size,
                           sparse_mode, actual_seq_q_list, actual_seq_kv_list):
    sparse_indices = torch.full((B, S1, N2, K), -1, dtype=torch.int32)

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
                block_indices = torch.randperm(valid_blocks_max - 1, dtype=torch.int32)
                sparse_indices[b, s1_idx, :, :valid_blocks_topk - 1] = block_indices[:valid_blocks_topk - 1]
            sparse_indices[b, s1_idx, :, valid_blocks_topk - 1] = valid_blocks_max - 1

    return sparse_indices


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
        return torch.tensor(actual_seq_list, dtype=torch.int32).cumsum(0)
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
    flat = in_tensor.reshape(-1)
    multi_shape = flat.numel()
    print(f"[trans_np_hifuint8_tensor_to_float32]total size:{multi_shape}")
    out_flat = torch.zeros(multi_shape, dtype=torch.float32)
    for i in range(multi_shape):
        out_flat[i] = cvt_hifuint8_to_float(flat[i].item())
    return out_flat.reshape(shape_tensor)


def cvt_hifuint8_to_float(x, over_mode=True):
    x = int(x)
    if x == 0:
        return float(0)
    elif x == 128:
        if over_mode:
            return float('nan')
        else:
            return float(0)
    elif x == 239:
        if over_mode:
            return float('-inf')
        else:
            return -32768
    elif x == 111:
        if over_mode:
            return float('inf')
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
        elif dot_4_value == 1:
            exponet_value = 0
            fra_int = x & 7  # b00000111
            m_value = 1.0 + fra_int * 0.125
        elif dot_4_value == 0:
            m_value = 1
            exponet_value = (x & 7) - 23  # b00000111 = 7
        else:
            print("error,dot error")
            m_value = 0.0
            exponet_value = 0
        return sign*pow(2.0, exponet_value)*m_value


def concat_tensor(tensor1, shape1, tensor2, shape2, n, tnd_flag=False):
    if len(shape1) != len(shape2):
        print(f"[ERROR]concat_tensor: 相加的两个tensor 维数不同! shape1 = {shape1}, shape2 =  {shape2}")
        return None, None

    t1 = tensor1
    t2 = tensor2

    if len(shape1) == 2:
        # 量化参数1h
        if shape1[0] != shape2[0]:
            print(f"[ERROR]concat_tensor: 相加的两个tensor 维度非法! shape1 = {shape1}, shape2 =  {shape2}")
            return None, None
        d1 = int(shape1[1] / n)
        d2 = int(shape2[1] / n)
        tensor1_1nd = t1.reshape(shape1[0], n, d1)
        tensor2_1nd = t2.reshape(shape2[0], n, d2)
        concatenated_tensor = torch.cat((tensor1_1nd, tensor2_1nd), dim=2)
        concatenated_shape = [shape1[0], shape1[1] + shape2[1]]
        concatenated_tensor = concatenated_tensor.reshape(concatenated_shape)
    elif len(shape1) == 3:
        if shape1[0] != shape2[0] or shape1[1] != shape2[1]:
            print(f"[ERROR]concat_tensor: 相加的两个tensor 维度非法! shape1 = {shape1}, shape2 =  {shape2}")
            return None, None
        if tnd_flag:
            concatenated_tensor = torch.cat((t1, t2), dim=2)
            concatenated_shape = [shape1[0], shape1[1], shape1[2] + shape2[2]]
        else:
            d1 = int(shape1[2] / n)
            d2 = int(shape2[2] / n)
            tensor1_bsnd = t1.reshape(shape1[0], shape1[1], n, d1)
            tensor2_bsnd = t2.reshape(shape2[0], shape2[1], n, d2)
            concatenated_tensor = torch.cat((tensor1_bsnd, tensor2_bsnd), dim=3)
            concatenated_shape = [shape1[0], shape1[1], shape1[2] + shape2[2]]
            concatenated_tensor = concatenated_tensor.reshape(concatenated_shape)
    elif len(shape1) == 4:
        if shape1[0] != shape2[0] or shape1[1] != shape2[1] or shape1[2] != shape2[2]:
            print(f"[ERROR]concat_tensor: 相加的两个tensor 维度非法! shape1 = {shape1}, shape2 =  {shape2}")
            return None, None
        concatenated_tensor = torch.cat((t1, t2), dim=3)
        concatenated_shape = [shape1[0], shape1[1], shape1[2], shape1[3] + shape2[3]]
    else:
        print(f"[ERROR]concat_tensor: shape1 维度非法! shape1 = {shape1}")
        return None, None
    return concatenated_tensor, concatenated_shape


def preprocessing(fa_param):
    numHeads = fa_param['numHeads']
    numKeyValueHeads = fa_param['numKeyValueHeads']

    def _clone(t):
        if torch.is_tensor(t):
            return t.clone()
        return copy.deepcopy(t)

    def _trans_hifloat8_to_float32(t):
        return trans_np_hifuint8_tensor_to_float32(t)

    # 备份拼接前的信息
    fa_param['q_tensor_old'] = _clone(fa_param['q_tensor'])
    fa_param['q_shape_old'] = copy.deepcopy(fa_param['q_shape'])
    fa_param['k_tensor_old'] = _clone(fa_param['k_tensor'])
    fa_param['k_shape_old'] = copy.deepcopy(fa_param['k_shape'])

    q_new_tensor, q_new_shape = concat_tensor(fa_param['q_tensor'], fa_param['q_shape'],
                                              fa_param['q_rope_tensor'],
                                              fa_param['q_rope_shape'], numHeads, fa_param['tnd_flag'])
    if q_new_tensor is not None:
        fa_param['q_tensor'] = q_new_tensor
        fa_param['q_shape'] = q_new_shape
    else:
        print("[ERROR]q tensor的预处理异常，输出空tensor！")

    key = fa_param['k_tensor']
    value = fa_param['v_tensor']
    if fa_param['k_dtype'] == "hifloat8":
        key = _trans_hifloat8_to_float32(fa_param['k_tensor'])
        value = _trans_hifloat8_to_float32(fa_param['v_tensor'])
    if value is not None:
        fa_param['v_tensor'] = value
    else:
        print("[ERROR]k tensor的预处理异常，输出空tensor！")

    k_new_tensor, k_new_shape = concat_tensor(key, fa_param['k_shape'],
                                              fa_param['k_rope_tensor'],
                                              fa_param['k_rope_shape'], numKeyValueHeads,
                                              fa_param['kv_tnd_flag'])
    if k_new_tensor is not None:
        fa_param['k_tensor'] = k_new_tensor
        fa_param['k_shape'] = k_new_shape
    else:
        print("[ERROR]k tensor的预处理异常，输出空tensor！")

    return fa_param


def _n_trans_shape_to_bnsd(tensor, shape, layout, headnums=None, act_seq=None, tensor_name=None):
    t = tensor

    if layout in ["BSH", "BSH_NBSD"]:
        if headnums is None:
            return t, shape
        B = shape[0]
        S = shape[1]
        if tensor_name != 'sparseIndices':
            H = shape[2]
            N = headnums
            D = H // N
            out = t.reshape(B, S, N, D).permute(0, 2, 1, 3).contiguous()
            return out, [B, N, S, D]
        else:
            out = t.permute(0, 2, 1, 3).contiguous()
            return out, [B, headnums, S, shape[3]]
    elif layout in ["BSND", "PA_BSND"]:
        B = shape[0]
        S = shape[1]
        N = shape[2]
        D = shape[3]
        out = t.permute(0, 2, 1, 3).contiguous()
        return out, [B, N, S, D]
    elif layout in ["TND", "TND_NTD"]:
        T = shape[0]
        N = shape[1]
        D = shape[2]
        B = len(act_seq)
        S = max(act_seq)
        new_tensor = torch.zeros((B, N, S, D), dtype=t.dtype)
        t_start = 0
        for b_index in range(B):
            act_s = act_seq[b_index]
            t_end = t_start + act_s
            if act_s == 0:
                continue
            for n_index in range(N):
                new_tensor[b_index, n_index, 0:act_s, :] = t[t_start:t_end, n_index, :]
            t_start += act_s
        return new_tensor, [B, N, S, D]
    else:
        return t, shape


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
        output = torch.zeros(size=(T, N, D), dtype=tensor.dtype)
        t_start = 0
        for b_index in range(B):
            act_s = act_q[b_index]
            t_end = t_start + act_s

            if act_s == 0:
                continue
            for n_index in range(N):
                output[t_start:t_end, n_index, :] = tensor[b_index, n_index, :act_s, :]
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
def get_param_fus(input_tensor_dict, params):
    fa_param = {}
    fa_param['normal_flag'] = True
    # ===参数获取===
    # >> attr info
    fa_param['actualSeqLengths_q_raw'] = params['actualseqlengths']
    fa_param['actualSeqLengths_raw'] = params['actualseqlengthskv']
    fa_param['scaleValue'] = params['scalevalue']
    fa_param['layout_query'] = params['layout_query']
    fa_param['layout_kv'] = params['layout_kv']
    fa_param['sparse_mode'] = params['sparsemode']
    fa_param['sparse_blocksize'] = params['sparse_blocksize']

    # >> q info
    fa_param['q_shape'] = params['shape_input']['query']
    fa_param['q_dtype'] = trans_input_dtype(params['dtype_input']['query'])
    fa_param['q_tensor'] = input_tensor_dict['query']

    # >> k info
    fa_param['k_shape'] = params['shape_input']['key']
    fa_param['k_dtype'] = trans_input_dtype(params['dtype_input']['key'])
    fa_param['k_tensor'] = input_tensor_dict['key']

    # >> v info  MLA场景 KV要保持一致
    fa_param['v_shape'] = fa_param['k_shape']
    fa_param['v_dtype'] = fa_param['k_dtype']
    fa_param['v_tensor'] = fa_param['k_tensor']

    # >> sparse_indices_shape info
    fa_param['sparse_indices_shape'] = params['shape_input']['sparse_indices']
    fa_param['sparse_indices_dtype'] = trans_input_dtype(params['dtype_input']['sparse_indices'])
    fa_param['sparse_indices_tensor'] = input_tensor_dict['sparse_indices']

    # >> block_table info
    fa_param['bt_shape'] = params['shape_input']['block_table']
    fa_param['bt_dtype'] = trans_input_dtype(params['dtype_input']['block_table'])
    fa_param['block_table'] = input_tensor_dict['block_table']

    # >> q_concat info
    fa_param['q_concat_shape'] = params['shape_input']['query_cache']
    fa_param['q_concat_dtype'] = trans_input_dtype(params['dtype_input']['query_cache'])

    # >> k_concat info
    fa_param['k_concat_shape'] = params['shape_input']['key_cache']
    fa_param['k_concat_dtype'] = trans_input_dtype(params['dtype_input']['key_cache'])

    # >> v_concat info
    fa_param['v_concat_shape'] = params['shape_input']['value_cache']
    fa_param['v_concat_dtype'] = trans_input_dtype(params['dtype_input']['value_cache'])

    # >> q_rope info
    fa_param['q_rope_shape'] = params['shape_input']['query_rope']
    fa_param['q_rope_dtype'] = trans_input_dtype(params['dtype_input']['query_rope'])
    fa_param['q_rope_tensor'] = input_tensor_dict['query_rope']

    # >> k_rope info
    fa_param['k_rope_shape'] = params['shape_input']['key_rope']
    fa_param['k_rope_dtype'] = trans_input_dtype(params['dtype_input']['key_rope'])
    fa_param['k_rope_tensor'] = input_tensor_dict['key_rope']

    # >> k_dequant_scale info
    fa_param['k_dequant_scale_shape'] = params['shape_input']['dequant_scale']
    fa_param['k_dequant_scale_dtype'] = trans_input_dtype(params['dtype_input']['dequant_scale'])
    fa_param['k_dequant_scale_tensor'] = input_tensor_dict['dequant_scale']

    # >> v_dequant_scale info, 和k 一致
    fa_param['v_dequant_scale_shape'] = params['shape_input']['dequant_scale']
    fa_param['v_dequant_scale_dtype'] = trans_input_dtype(params['dtype_input']['dequant_scale'])
    fa_param['v_dequant_scale_tensor'] = input_tensor_dict['dequant_scale']

    # >> out info
    fa_param['out_shape'] = fa_param['q_shape']
    fa_param['out_dtype'] = trans_input_dtype(params['dtype_output'][0])

    # >> batch_size info
    if fa_param['layout_query'] in ["TND"]:
        fa_param['b'] = len(fa_param['actualSeqLengths_q_raw'])
        fa_param['numHeads'] = fa_param['q_shape'][1]
    else:
        fa_param['b'] = fa_param['q_shape'][0]
        fa_param['numHeads'] = fa_param['q_shape'][2]

    fa_param['sparse_blockcount'] = fa_param['sparse_indices_shape'][-1]

    fa_param['blocksize'] = 1
    fa_param['numKeyValueHeads'] = fa_param['k_shape'][-2]
    if fa_param['layout_kv'] in ["PA_BSND"]:
        fa_param['blocksize'] = fa_param['k_concat_shape'][1]

    fa_param['pa_flag'] = True if fa_param['layout_kv'] in ["PA_BSND"] else False
    fa_param['tnd_flag'] = True if fa_param['layout_query'] in ["TND"] else False
    fa_param['kv_tnd_flag'] = True if fa_param['layout_kv'] in ["TND"] else False

    # ===异常判断===
    if 0 in fa_param['q_shape']:
        fa_param['normal_flag'] = False
        print('[WARNING]:异常-->  q为空场景！')
    if len(fa_param['k_shape']) == 0 or 0 in fa_param['k_shape']:
        fa_param['normal_flag'] = False
        print('[WARNING]:异常-->  k为空场景！')
    if len(fa_param['v_shape']) == 0 or 0 in fa_param['v_shape']:
        fa_param['normal_flag'] = False
        print('[WARNING]:异常-->  v为空场景！')

    return fa_param




def kv_concat_nopa_preprocessing(input_tensor_dict, fa_param, params):
    k_tensor = fa_param['_raw_key'] if fa_param.get('_raw_key') is not None else fa_param['k_tensor_old']
    k_rope_tensor = fa_param['_raw_key_rope'] if fa_param.get('_raw_key_rope') is not None else fa_param['k_rope_tensor']
    k_dequant_scale_tensor = fa_param['_raw_dequant_scale'] if fa_param.get('_raw_dequant_scale') is not None else fa_param['k_dequant_scale_tensor']
    kv_layout = fa_param['layout_kv']
    if kv_layout != "PA_BSND":
        k_dtype = fa_param['k_dtype']
        _k_dtype_map = {"hifloat8": torch.uint8, "float8_e4m3fn": torch.float8_e4m3fn, "int8": torch.int8}
        k_torch_dtype = _k_dtype_map.get(k_dtype, torch.uint8)

        # 全程 torch tensor，raw bytes 用 .view(torch.uint8) 重解释后拼接
        t_k = k_tensor.contiguous()                            # uint8 / float8 / int8
        t_rope = k_rope_tensor.contiguous().view(torch.uint8)  # bfloat16 → uint8
        t_scale = k_dequant_scale_tensor.contiguous().view(torch.uint8)  # float32 → uint8

        key_concat = torch.cat((t_k.view(torch.uint8), t_rope, t_scale), dim=-1).view(k_torch_dtype)

        # query_cache: raw query + raw query_rope 拼接后转目标 dtype
        q_raw = fa_param['_raw_query'] if fa_param.get('_raw_query') is not None else fa_param['q_tensor']
        q_rope_raw = fa_param.get('_raw_query_rope')
        if q_rope_raw is not None:
            q_cat = torch.cat((q_raw, q_rope_raw), dim=-1)
        else:
            q_cat = q_raw
        q_dtype_str = params['dtype_input'].get('query', 'bf16')
        _q_dtype_map = {'bf16': torch.bfloat16, 'bfloat16': torch.bfloat16, 'fp16': torch.float16,
                        'float16': torch.float16, 'fp32': torch.float32, 'float32': torch.float32}
        q_torch_dtype = _q_dtype_map.get(q_dtype_str)
        q_torch = q_cat.to(q_torch_dtype) if q_torch_dtype is not None else q_cat

        input_tensor_dict['query_cache'] = q_torch.to('npu')
        input_tensor_dict['key_cache'] = key_concat.to('npu')
        input_tensor_dict['value_cache'] = key_concat.to('npu')
    else:
        print(f"[ERROR]Wrong input layout_kv: {fa_param['layout_kv']}")


def generate_input_tensors(params):
    tensor_keys = ['query', 'key', 'value', 'sparse_indices', 'block_table',
                   'query_cache', 'key_cache', 'value_cache',
                   'query_rope', 'key_rope', 'dequant_scale', 'v_dequant_scale']
    raw = {}
    for key in tensor_keys:
        raw[key] = generate_tensor_data.gen_tensor_data(params, key).to('npu')

    layout_query = params["layout_query"]
    layout_kv = params["layout_kv"]
    B = params.get("shape_input", {}).get("block_table", [1])[0]
    S1 = params["shape_input"]["query"][1] if layout_query == "BSND" else params["shape_input"]["query"][0]
    S2 = params["shape_input"]["key"][1] if layout_kv == "BSND" else params["shape_input"]["key"][0]
    actual_seq_q_list = _resolve_actual_seq(params["actualseqlengths"], S1, B, "actual_seq_q")
    actual_seq_kv_list = _resolve_actual_seq(params["actualseqlengthskv"], S2, B, "actual_seq_kv")
    if layout_kv == "PA_BSND":
        actual_seq_kv_tensor = torch.tensor(actual_seq_kv_list, dtype=torch.int32)
    else:
        actual_seq_kv_tensor = _build_actual_seq_tensor(layout_kv, actual_seq_kv_list)
    actual_seq_q_tensor = _build_actual_seq_tensor(layout_query, actual_seq_q_list)

    return {
        "query": raw["query"],
        "key": raw["key"],
        "value": raw["value"],
        "sparse_indices": raw["sparse_indices"],
        "query_cache": raw["query_cache"],
        "key_cache": raw["key_cache"],
        "value_cache": raw["value_cache"],
        "query_rope": raw["query_rope"],
        "key_rope": raw["key_rope"],
        "block_table": raw["block_table"],
        "dequant_scale": raw["dequant_scale"],
        "v_dequant_scale": raw["v_dequant_scale"],
        "key_dequant_scale": raw["dequant_scale"],
        "value_dequant_scale": raw["v_dequant_scale"],
        "actual_seq_lengths_query": actual_seq_q_tensor,
        "actual_seq_lengths_kv": actual_seq_kv_tensor,
        "scale_value": params["scalevalue"],
        "key_quant_mode": params["key_quant_mode"],
        "value_quant_mode": params["value_quant_mode"],
        "sparse_block_size": params["sparse_blocksize"],
        "layout_query": layout_query,
        "layout_kv": layout_kv,
        "sparse_mode": params["sparsemode"],
        "attention_mode": params["attention_mode"],
        "quant_scale_repo_mode": params["quant_scale_repo_mode"],
        "tile_size": params["tile_size"],
        "rope_head_dim": params["rope_head_dim"],
        "kv_dtype": params["dtype_input"]["key"],
        "pre_tokens": (1 << 63) - 1,
        "next_tokens": (1 << 63) - 1,
    }


def compute_cpu(input_tensor_dict, params):
    try:
        out_data = compute_golden(input_tensor_dict, params)
        return out_data, input_tensor_dict, params
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] compute_cpu failed: {e}")
        return None, input_tensor_dict, params


def _prepare_fa_param(input_tensor_dict, params):
    """步骤1-4：参数初始化、q/k拼接、bnsd转换、actseq处理，返回完整 fa_param。"""
    fa_param = get_param_fus(input_tensor_dict, params)
    if not fa_param['normal_flag']:
        return fa_param

    actualSeqLengths_q = fa_param['actualSeqLengths_q_raw']
    actualSeqLengths_kv = fa_param['actualSeqLengths_raw']
    layoutQuery = fa_param['layout_query']
    numHeads = fa_param['numHeads']
    numKeyValueHeads = fa_param['numKeyValueHeads']

    fa_param = preprocessing(fa_param)

    if fa_param['tnd_flag']:
        fa_param['actualSeqLengths_q'] = trans_tnd_actseq(actualSeqLengths_q)
        if fa_param['pa_flag']:
            fa_param['actualSeqLengths_kv'] = actualSeqLengths_kv
        q_bnsd_tensor, q_bnsd_shape = _n_trans_shape_to_bnsd(fa_param['q_tensor'], fa_param['q_shape'], layoutQuery,
                                                             numHeads, fa_param['actualSeqLengths_q'])
    else:
        q_bnsd_tensor, q_bnsd_shape = _n_trans_shape_to_bnsd(fa_param['q_tensor'], fa_param['q_shape'], layoutQuery,
                                                             numHeads, actualSeqLengths_q)
    if fa_param['kv_tnd_flag']:
        fa_param['actualSeqLengths_kv'] = trans_tnd_actseq(actualSeqLengths_kv)
        k_bnsd_tensor, k_bnsd_shape = _n_trans_shape_to_bnsd(fa_param['k_tensor'], fa_param['k_shape'],
                                                             fa_param['layout_kv'],
                                                             numKeyValueHeads, fa_param['actualSeqLengths_kv'])
        v_bnsd_tensor, v_bnsd_shape = _n_trans_shape_to_bnsd(fa_param['v_tensor'], fa_param['v_shape'],
                                                             fa_param['layout_kv'],
                                                             numKeyValueHeads, fa_param['actualSeqLengths_kv'])
        k_dequant_scale_bnsd_tensor, k_dequant_scale_bnsd_shape = _n_trans_shape_to_bnsd(
            fa_param['k_dequant_scale_tensor'], fa_param['k_dequant_scale_shape'],
            fa_param['layout_kv'], numKeyValueHeads, fa_param['actualSeqLengths_kv'])
        v_dequant_scale_bnsd_tensor, v_dequant_scale_bnsd_shape = _n_trans_shape_to_bnsd(
            fa_param['v_dequant_scale_tensor'], fa_param['v_dequant_scale_shape'],
            fa_param['layout_kv'], numKeyValueHeads, fa_param['actualSeqLengths_kv'])
    else:
        k_bnsd_tensor, k_bnsd_shape = _n_trans_shape_to_bnsd(fa_param['k_tensor'], fa_param['k_shape'],
                                                             fa_param['layout_kv'],
                                                             numKeyValueHeads, actualSeqLengths_kv)
        v_bnsd_tensor, v_bnsd_shape = _n_trans_shape_to_bnsd(fa_param['v_tensor'], fa_param['v_shape'],
                                                             fa_param['layout_kv'],
                                                             numKeyValueHeads, actualSeqLengths_kv)
        k_dequant_scale_bnsd_tensor, k_dequant_scale_bnsd_shape = _n_trans_shape_to_bnsd(
            fa_param['k_dequant_scale_tensor'], fa_param['k_dequant_scale_shape'],
            fa_param['layout_kv'], numKeyValueHeads, actualSeqLengths_kv)
        v_dequant_scale_bnsd_tensor, v_dequant_scale_bnsd_shape = _n_trans_shape_to_bnsd(
            fa_param['v_dequant_scale_tensor'], fa_param['v_dequant_scale_shape'],
            fa_param['layout_kv'], numKeyValueHeads, actualSeqLengths_kv)
    if fa_param['tnd_flag']:
        sparse_indices_bnsd_tensor, sparse_indices_bnsd_shape = _n_trans_shape_to_bnsd(
            fa_param['sparse_indices_tensor'], fa_param['sparse_indices_shape'], layoutQuery,
            numKeyValueHeads, fa_param['actualSeqLengths_q'], 'sparseIndices')
    else:
        sparse_indices_bnsd_tensor, sparse_indices_bnsd_shape = _n_trans_shape_to_bnsd(
            fa_param['sparse_indices_tensor'], fa_param['sparse_indices_shape'], layoutQuery,
            numKeyValueHeads, actualSeqLengths_q, 'sparseIndices')

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

    if not fa_param['tnd_flag']:
        if len(actualSeqLengths_q) == 0:
            fa_param['actualSeqLengths_q'] = [q_bnsd_shape[2]] * q_bnsd_shape[0]
        elif len(actualSeqLengths_q) == 1:
            fa_param['actualSeqLengths_q'] = [actualSeqLengths_q[0]] * q_bnsd_shape[0]
        else:
            fa_param['actualSeqLengths_q'] = actualSeqLengths_q
    if not fa_param['kv_tnd_flag']:
        if len(actualSeqLengths_kv) == 0:
            fa_param['actualSeqLengths_kv'] = [k_bnsd_shape[2]] * k_bnsd_shape[0]
        elif len(actualSeqLengths_kv) == 1:
            fa_param['actualSeqLengths_kv'] = [actualSeqLengths_kv[0]] * k_bnsd_shape[0]
        else:
            fa_param['actualSeqLengths_kv'] = actualSeqLengths_kv

    return fa_param


def _generate_sparse_indices(input_tensor_dict, fa_param, params, out_shape, layoutQuery):
    sparse_indices_bnsd_shape = fa_param['sparse_indices_bnsd_shape']
    sparse_indices_tensor = torch.full(sparse_indices_bnsd_shape, -1, dtype=torch.int32)

    for b in range(sparse_indices_bnsd_shape[0]):
        act_seqlen_kv = fa_param['actualSeqLengths_kv'][b]
        act_seqlen_q = fa_param['actualSeqLengths_q'][b]

        for n in range(sparse_indices_bnsd_shape[1]):
            for s in range(sparse_indices_bnsd_shape[2]):
                if fa_param['sparse_mode'] == 0:
                    threshold = act_seqlen_kv
                elif fa_param['sparse_mode'] == 3:
                    threshold = act_seqlen_kv - act_seqlen_q + s + 1

                if threshold <= 0:
                    continue
                valid_blocks_max = math.ceil(max(0, threshold) / fa_param['sparse_blocksize'])
                block_indices = torch.randperm(valid_blocks_max - 1, dtype=torch.int32)
                valid_blocks_topk = min(valid_blocks_max, fa_param['sparse_blockcount'])
                sparse_indices_tensor[b, n, s, :valid_blocks_topk - 1] = block_indices[:valid_blocks_topk - 1]
                sparse_indices_tensor[b, n, s, valid_blocks_topk - 1] = valid_blocks_max - 1

    if fa_param['tnd_flag']:
        sparse_indices_tensor_tnd = trans_bnsd_to_layout(sparse_indices_tensor, out_shape, layoutQuery,
                                                         fa_param['actualSeqLengths_q'])
        input_tensor_dict['sparse_indices'] = sparse_indices_tensor_tnd.to(torch.int32).to('npu')
    else:
        input_tensor_dict['sparse_indices'] = sparse_indices_tensor.permute(0, 2, 1, 3).contiguous().to('npu')

    fa_param['sparse_indices_bnsd_tensor'] = sparse_indices_tensor


def _generate_block_table_and_cache(input_tensor_dict, fa_param, params, actualSeqLengths_kv, out_shape):
    """生成 block_table 以及 query/key/value_cache，写入 input_tensor_dict。
    PA 场景返回 None 表示正常继续；若 block_table 为空则返回 zeros tensor 提前退出。"""
    if fa_param['pa_flag']:
        blockSize = fa_param['blocksize']
        blockTableShape = fa_param['bt_shape']
        blockNum = fa_param['k_concat_shape'][0]
        if 0 in fa_param['bt_shape']:
            print('[WARNING]:block_table为空场景，输出空tensor！')
            return torch.zeros(out_shape)
        blockNumPerBlock = []
        block_num_min = 0
        for actual_seq in actualSeqLengths_kv:
            blockNumPerBlock.append(math.ceil(actual_seq / blockSize))
            block_num_min += math.ceil(actual_seq / blockSize)
        if block_num_min > blockNum:
            print(f"[ERROR]Wrong input k_cache_shape: get blockNum = {blockNum}, but expect blockNum > {block_num_min}")
            exit(1)
        block_idx_list = torch.randperm(blockNum, dtype=torch.int32)
        block_idx = 0
        block_table = torch.full((blockTableShape[0], blockTableShape[1]), -1, dtype=torch.int32)
        block_table_batch_idx = 0
        for idx in blockNumPerBlock:
            for j in range(idx):
                block_table[block_table_batch_idx][j] = block_idx_list[block_idx]
                block_idx += 1
            block_table_batch_idx += 1
        fa_param['block_table'] = block_table
        input_tensor_dict['block_table'] = block_table.to('npu')
        print(f"[PageAtten]Input Kdtype:{params['dtype_input']['key']} Vdtype:{params['dtype_input']['value']}")
        print('[INFO]:PA + Deepseek！')
        deepseek_ds_pa_preprocessing(input_tensor_dict, fa_param, params)
    else:
        kv_concat_nopa_preprocessing(input_tensor_dict, fa_param, params)
    return None


def compute_golden(input_tensor_dict, params):
    print("cpu执行中...")
    # CPU golden 链路工作在 float32 torch tensor
    def _to_cpu_float32_tensor(val):
        if torch.is_tensor(val):
            if val.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                return val.float().cpu()
            return val.cpu().float()
        return val

    cpu_input = {key: _to_cpu_float32_tensor(val) for key, val in input_tensor_dict.items()}
    fa_param = _prepare_fa_param(cpu_input, params)
    if not fa_param['normal_flag']:
        return torch.zeros(fa_param['out_shape'])

    out_shape = fa_param['out_shape']
    layoutQuery = fa_param['layout_query']
    actualSeqLengths_kv = fa_param['actualSeqLengths_raw']

    # 保存原始 raw bytes 张量到 fa_param，供 kv cache 构建函数使用（原始 dtype，不经 float32 转换）
    def _to_cpu_raw(val):
        if torch.is_tensor(val):
            return val.cpu()
        return val
    fa_param['_raw_key'] = _to_cpu_raw(input_tensor_dict.get('key'))
    fa_param['_raw_key_rope'] = _to_cpu_raw(input_tensor_dict.get('key_rope'))
    fa_param['_raw_dequant_scale'] = _to_cpu_raw(input_tensor_dict.get('dequant_scale'))
    fa_param['_raw_v_dequant_scale'] = _to_cpu_raw(input_tensor_dict.get('v_dequant_scale'))
    fa_param['_raw_query'] = _to_cpu_raw(input_tensor_dict.get('query'))
    fa_param['_raw_query_rope'] = _to_cpu_raw(input_tensor_dict.get('query_rope'))

    _generate_sparse_indices(input_tensor_dict, fa_param, params, out_shape, layoutQuery)
    early_exit = _generate_block_table_and_cache(input_tensor_dict, fa_param, params, actualSeqLengths_kv, out_shape)
    if early_exit is not None:
        return early_exit

    # ===计算入口===
    y_all = _t_increattention_bnsd(fa_param)
    y_all = trans_bnsd_to_layout(y_all, out_shape, layoutQuery, fa_param['actualSeqLengths_q'])
    return y_all


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
            s2_sparse.extend(range(begin_Idx, end_Idx))
        else:
            s2_sparse.extend(range(begin_Idx, threshold))

    emptyFlag = False
    if len(s2_sparse) == 0:
        k_sparse, v_sparse = [], []
        emptyFlag = True
    else:
        k_sparse, v_sparse = k_tensor[batch, n2Idx, s2_sparse, :], v_tensor[batch, n2Idx, s2_sparse, :]

    return emptyFlag, k_sparse, v_sparse


def softmax(x):
    x = x.float()
    x_max = x.max(dim=-1, keepdim=True).values
    x_sub = x - x_max
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdim=True)
    ans = y / x_sum
    return ans


def _t_increattention_bnsd(fa_param):
    batch_size = fa_param['b']
    numheads = fa_param['numHeads']
    numKeyValueHeads = fa_param['numKeyValueHeads']
    actualSeqLengths_q = fa_param['actualSeqLengths_q']
    actualSeqLengths_kv = fa_param['actualSeqLengths_kv']
    scaleValue = fa_param['scaleValue']
    sparse_blocksize = fa_param['sparse_blocksize']
    sparse_blockcount = fa_param['sparse_blockcount']
    sparseIndicesIndices = fa_param['sparse_indices_bnsd_tensor']
    out_shape_bnsd = fa_param['q_bnsd_shape']
    out_shape_bnsd[-1] = fa_param['v_bnsd_shape'][-1]
    sparse_mode = fa_param['sparse_mode']
    g = numheads // numKeyValueHeads

    q_bnsd_tensor = fa_param['q_bnsd_tensor']
    k_bnsd_tensor = fa_param['k_bnsd_tensor'].float()
    k_dequant_scale_bnsd_tensor = fa_param['k_dequant_scale_bnsd_tensor'].float()
    v_bnsd_tensor = fa_param['v_bnsd_tensor'].float()
    v_dequant_scale_bnsd_tensor = fa_param['v_dequant_scale_bnsd_tensor'].float()
    tile_size = 128
    if fa_param['q_dtype'] == "float16":
        k_bnsd_tensor[..., :512] = (k_bnsd_tensor[..., :512] * torch.repeat_interleave(k_dequant_scale_bnsd_tensor, tile_size, dim=-1)).to(torch.float16)
    elif fa_param['q_dtype'] == "bfloat16":
        k_bnsd_tensor[..., :512] = (k_bnsd_tensor[..., :512] * torch.repeat_interleave(k_dequant_scale_bnsd_tensor, tile_size, dim=-1)).to(torch.bfloat16)
    elif fa_param['q_dtype'] == "float32":
        k_bnsd_tensor[..., :512] = (k_bnsd_tensor[..., :512] * torch.repeat_interleave(k_dequant_scale_bnsd_tensor, tile_size, dim=-1)).to(torch.float32)
    v_bnsd_tensor = k_bnsd_tensor[..., :512].clone()
    matmul_dtype = torch.float32
    y = torch.zeros(out_shape_bnsd, dtype=torch.float32)

    q_bnsd_tensor = fa_param['q_bnsd_tensor'].float()

    for batch in range(batch_size):
        curr_actualSeq_q = actualSeqLengths_q[batch]
        curr_actualSeq = actualSeqLengths_kv[batch]
        for n2Idx in range(numKeyValueHeads):
            for s1Idx in range(curr_actualSeq_q):
                if s1Idx < curr_actualSeq_q - curr_actualSeq and sparse_mode != 0:
                    y[batch, n2Idx * g: (n2Idx + 1) * g, s1Idx, :] = torch.zeros([g, out_shape_bnsd[-1]], dtype=torch.float32)
                    continue
                q_curr = q_bnsd_tensor[batch, n2Idx * g: (n2Idx + 1) * g, s1Idx, :]
                sparse_indices_Indices = sparseIndicesIndices[batch, n2Idx, s1Idx, :]

                emptyFlag, k_sparse, v_sparse = gatherKV(k_bnsd_tensor, v_bnsd_tensor, sparse_indices_Indices,
                                                         sparse_blocksize, sparse_blockcount, batch, n2Idx, s1Idx,
                                                         curr_actualSeq, curr_actualSeq_q, sparse_mode)
                if emptyFlag:
                    continue
                bmm1Res = torch.matmul(q_curr.float(), k_sparse.float().T)
                scaleRes = bmm1Res * scaleValue
                softmax_res = softmax(scaleRes)
                if fa_param['q_dtype'] == "float16":
                    bmm2Res = torch.matmul(softmax_res.to(torch.float16).float(), v_sparse.float())
                elif fa_param['q_dtype'] == "bfloat16":
                    bmm2Res = torch.matmul(softmax_res.to(torch.bfloat16).float(), v_sparse.float())
                else:
                    bmm2Res = torch.matmul(softmax_res.float(), v_sparse.float())
                y[batch, n2Idx * g: (n2Idx + 1) * g, s1Idx, :] = bmm2Res
    return y


def deepseek_ds_pa_preprocessing(input_tensor_dict, fa_param, params):
    k_dtype = fa_param['k_dtype']
    k_ori_shape_list = fa_param['k_shape_old']
    numKeyValueHeads = fa_param['numKeyValueHeads']
    numHeads = fa_param['numHeads']
    kv_layout = fa_param['layout_kv']
    actualSeqLengths_kv = fa_param['actualSeqLengths_kv']
    actualSeqLengths_q = fa_param['actualSeqLengths_q']
    blockSize = fa_param['blocksize']
    blockTableShape = fa_param['bt_shape']
    block_table = fa_param['block_table']
    layoutQuery = fa_param['layout_query']
    layout_kv = fa_param['layout_kv']

    _k_dtype_map = {"hifloat8": torch.uint8, "float8_e4m3fn": torch.float8_e4m3fn, "int8": torch.int8}
    k_torch_dtype = _k_dtype_map.get(k_dtype, torch.uint8)

    k_cache_shape = fa_param['k_concat_shape'].copy()
    k_cache_shape[-1] = 512
    k_rope_cache_shape = fa_param['k_concat_shape'].copy()
    k_rope_cache_shape[-1] = 64
    k_dequant_scale_cache_shape = fa_param['k_concat_shape'].copy()
    k_dequant_scale_cache_shape[-1] = 512 // 128

    k_cache = torch.zeros(k_cache_shape, dtype=k_torch_dtype)
    k_rope_cache = torch.zeros(k_rope_cache_shape, dtype=torch.bfloat16)
    k_dequant_scale_cache = torch.zeros(k_dequant_scale_cache_shape, dtype=torch.float32)

    q_tensor = fa_param['_raw_query'] if fa_param.get('_raw_query') is not None else fa_param['q_tensor']

    B = len(fa_param['actualSeqLengths_kv'])

    if layout_kv in ["PA_BSND"]:
        k_tensor_bsnd = fa_param['_raw_key'] if fa_param.get('_raw_key') is not None else fa_param['k_tensor_old']
        k_tensor_bsnd = k_tensor_bsnd.contiguous()
        if k_tensor_bsnd.shape[1] != blockTableShape[1] * blockSize:
            k_tensor_old = k_tensor_bsnd
            k_tensor_bsnd = torch.zeros(
                (k_tensor_old.shape[0], blockTableShape[1] * blockSize, k_tensor_old.shape[2], k_tensor_old.shape[3]),
                dtype=k_tensor_old.dtype,
            )
            k_tensor_bsnd[:, :k_tensor_old.shape[1], :, :] = k_tensor_old

        for b in range(B):
            for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                block_offset = block_i * blockSize
                if kv_cache_blk_id == -1:
                    continue
                for n in range(fa_param['k_concat_shape'][2]):
                    k_cache[kv_cache_blk_id, 0:blockSize, n:n + 1, :] = k_tensor_bsnd[b,
                                                                          block_offset:(block_offset + blockSize),
                                                                          n:n + 1, :]

        k_rope_bsnd = fa_param['_raw_key_rope'] if fa_param.get('_raw_key_rope') is not None else fa_param['k_rope_tensor']
        k_rope_bsnd = k_rope_bsnd.contiguous()
        if k_rope_bsnd.shape[1] != blockTableShape[1] * blockSize:
            k_rope_tensor_old = k_rope_bsnd
            k_rope_bsnd = torch.zeros(
                (k_rope_tensor_old.shape[0], blockTableShape[1] * blockSize,
                 k_rope_tensor_old.shape[2], k_rope_tensor_old.shape[3]),
                dtype=k_rope_tensor_old.dtype,
            )
            k_rope_bsnd[:, :k_rope_tensor_old.shape[1], :, :] = k_rope_tensor_old

        k_dequant_scale_bsnd = fa_param['_raw_dequant_scale'] if fa_param.get('_raw_dequant_scale') is not None else fa_param['k_dequant_scale_tensor']
        k_dequant_scale_bsnd = k_dequant_scale_bsnd.contiguous()
        if k_dequant_scale_bsnd.shape[1] != blockTableShape[1] * blockSize:
            k_dequant_scale_tensor_old = k_dequant_scale_bsnd
            k_dequant_scale_bsnd = torch.zeros(
                (k_dequant_scale_tensor_old.shape[0], blockTableShape[1] * blockSize,
                 k_dequant_scale_tensor_old.shape[2], k_dequant_scale_tensor_old.shape[3]),
                dtype=k_dequant_scale_tensor_old.dtype,
            )
            k_dequant_scale_bsnd[:, :k_dequant_scale_tensor_old.shape[1], :, :] = k_dequant_scale_tensor_old

        for b in range(B):
            for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                block_offset = block_i * blockSize
                if kv_cache_blk_id == -1:
                    continue
                for n in range(fa_param['k_rope_shape'][2]):
                    k_rope_cache[kv_cache_blk_id, 0:blockSize, n:n + 1, :] = k_rope_bsnd[b,
                                                                               block_offset:(block_offset + blockSize),
                                                                               n:n + 1, :]
                for n in range(fa_param['k_dequant_scale_shape'][2]):
                    k_dequant_scale_cache[kv_cache_blk_id, 0:blockSize, n:n + 1, :] = k_dequant_scale_bsnd[b,
                                                                                        block_offset:(block_offset + blockSize),
                                                                                        n:n + 1, :]
    else:
        print(f"[ERROR]Wrong input layout_kv: {fa_param['layout_kv']}")
        return

    if k_dtype == "int8":
        k_cache_concat = torch.cat((k_cache.contiguous(), k_rope_cache.contiguous().view(torch.int8),
                                    k_dequant_scale_cache.contiguous().view(torch.int8)), dim=-1)
    elif k_dtype == "float8_e4m3fn":
        k_cache_concat = torch.cat((k_cache.contiguous(), k_rope_cache.contiguous().view(torch.float8_e4m3fn),
                                    k_dequant_scale_cache.contiguous().view(torch.float8_e4m3fn)), dim=-1)
    elif k_dtype == "hifloat8":
        k_cache_concat = torch.cat((k_cache.contiguous(), k_rope_cache.contiguous().view(torch.uint8),
                                    k_dequant_scale_cache.contiguous().view(torch.uint8)), dim=-1)

    q_raw = fa_param['_raw_query'] if fa_param.get('_raw_query') is not None else fa_param['q_tensor']
    q_rope_raw = fa_param.get('_raw_query_rope')
    if q_rope_raw is not None:
        q_cat = torch.cat((q_raw, q_rope_raw), dim=-1)
    else:
        q_cat = q_raw
    q_dtype_str = params['dtype_input'].get('query', 'bf16')
    _q_dtype_map = {'bf16': torch.bfloat16, 'bfloat16': torch.bfloat16, 'fp16': torch.float16,
                    'float16': torch.float16, 'fp32': torch.float32, 'float32': torch.float32}
    q_torch_dtype = _q_dtype_map.get(q_dtype_str)
    q_torch = q_cat.to(q_torch_dtype) if q_torch_dtype is not None else q_cat
    input_tensor_dict['query_cache'] = q_torch.to('npu')
    input_tensor_dict['key_cache'] = k_cache_concat.to('npu')
    input_tensor_dict['value_cache'] = k_cache_concat.to('npu')


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



