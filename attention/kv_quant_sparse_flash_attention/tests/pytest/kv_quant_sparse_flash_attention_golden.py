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
import copy
import time
import random

np.random.seed(42)
torch.manual_seed(42)

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


def find_output_value(input_list, output_value_list):
    temp_list = []
    is_list_or_tuple = False
    for item in input_list:
        if not isinstance(item, list) and not isinstance(item, tuple):
            output_value_list.append(item)
        else:
            for elment in item:
                temp_list.append(elment)
    if len(temp_list):
        is_list_or_tuple = True
    return is_list_or_tuple, temp_list


def get_output_list(is_list_or_tuple, input_list, output_value_list):
    while True:
        if not is_list_or_tuple:
            return output_value_list
        else:
            is_list_or_tuple, input_list = find_output_value(input_list, output_value_list)


def qtensor_dim(q_shape, layoutQuery, numhead):
    if layoutQuery == "SH":  # SH格式
        return q_shape[0]
    elif layoutQuery == "BSH" or layoutQuery == "BSH_NBSD":
        return int(q_shape[2] / numhead)
    else:
        return q_shape[-1]


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


def str_to_bool_list(s: str):
    bool_list = []
    for item in s:
        if item == 0:
            bool_list.append(False)
        else:
            bool_list.append(True)
    return bool_list


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


def concat_tensor(tensor1, shape1, tensor2, shape2, n, tnd_flag=False):
    if len(shape1) != len(shape2):
        print(f"[ERROR]concat_tensor: 相加的两个tensor 维数不同! shape1 = {shape1}, shape2 =  {shape2}")
        return None, None
    elif len(shape1) == 2:
        # 量化参数1h
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
            concatenated_tensor = torch.cat((tensor1, tensor2), dim=2)
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
        concatenated_tensor = torch.cat((tensor1, tensor2), dim=3)
        concatenated_shape = [shape1[0], shape1[1], shape1[2], shape1[3] + shape2[3]]
    else:
        print(f"[ERROR]concat_tensor: shape1 维度非法! shape1 = {shape1}")
        return None, None
    return concatenated_tensor, concatenated_shape


# 融合表格参数获取函数
def get_param_fus(torch_tensor_list, params):
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
    fa_param['q_shape'] = params['shape_input'][0]
    fa_param['q_dtype'] = torch.bfloat16 if trans_input_dtype(params['dtype_input'][0]) == "bfloat16" else torch.float16
    fa_param['q_tensor'] = torch_tensor_list[0]

    # >> k info
    fa_param['k_shape'] = params['shape_input'][1]
    fa_param['k_dtype'] = trans_input_dtype(params['dtype_input'][1])
    fa_param['k_tensor'] = torch_tensor_list[1]

    # >> v info  MLA场景 KV要保持一致
    fa_param['v_shape'] = params['shape_input'][2]
    fa_param['v_dtype'] = trans_input_dtype(params['dtype_input'][2])
    fa_param['v_tensor'] = torch_tensor_list[2]

    # >> sparse_indices_shape info
    fa_param['sparse_indices_shape'] = params['shape_input'][3]
    fa_param['sparse_indices_dtype'] = trans_input_dtype(params['dtype_input'][3])
    fa_param['sparse_indices_tensor'] = torch_tensor_list[3]

    # >> block_table info
    fa_param['bt_shape'] = params['shape_input'][4]
    fa_param['bt_dtype'] = trans_input_dtype(params['dtype_input'][4])
    fa_param['block_table'] = torch_tensor_list[4]

    # >> k cache info
    fa_param['k_cache_shape'] = params['shape_input'][5]
    fa_param['k_cache_dtype'] = trans_input_dtype(params['dtype_input'][5])

    # >> v cache info
    fa_param['v_cache_shape'] = params['shape_input'][6]
    fa_param['v_cache_dtype'] = trans_input_dtype(params['dtype_input'][6])

    # >> out info
    fa_param['out_shape'] = fa_param['q_shape']
    fa_param['out_dtype'] = trans_input_dtype(params['dtype_output'][0])
    # ===flag_list判断===
    flag_list = str_to_bool_list(params['flaglist'])
    fa_param['pa_flag'] = True if fa_param['layout_kv'] in ["PA_BSND"] else False
    fa_param['tnd_flag'] = True if fa_param['layout_query'] in ["TND"] else False
    fa_param['kv_tnd_flag'] = True if fa_param['layout_kv'] in ["TND"] else False
    fa_param['return_softmax'] = params['return_softmax']
    if fa_param['return_softmax']==True:
        fa_param['softmax_max_shape'] = params['shape_output'][1]
        fa_param['softmax_max_dtype'] = trans_input_dtype(params['dtype_output'][1])
        fa_param['softmax_sum_shape'] = params['shape_output'][2]
        fa_param['softmax_sum_dtype'] = trans_input_dtype(params['dtype_output'][2])

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
        fa_param['blocksize'] = fa_param['k_cache_shape'][1]

    fa_param['rope_head_dim'] = params['rope_head_dim']

    # ===异常判断===
    if not flag_list[0] or not flag_list[1] or not flag_list[2]:
        fa_param['normal_flag'] = False
        print('[WARNING]:异常-->  q/k/v/不输入场景！')
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


def _n_trans_shape_to_bnsd(tensor, shape, layout, headnums=None, act_seq=None, tensor_name=None):
    if layout in ["BSND", "PA_BSND"]:
        B = shape[0]
        S = shape[1]
        N = shape[2]
        D = shape[3]
        tensor = tensor.transpose(1, 2)
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


def trans_bnsd_to_input_layout(tensor, shape, layout, act_q=None):
    if layout in ["BSH", "BSH_NBSD"]:
        output = tensor.permute(0, 2, 1, 3).contiguous().view(shape)
        return output
    elif layout in ["BSND", "BSND_NBSD"]:
        output = tensor.permute(0, 2, 1, 3).contiguous()
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
        return output
    else:
        return tensor


def gatherKV(k_tensor, v_tensor, sparse_indices_Indices, sparse_blocksize, sparse_blockcount, batch, n2Idx, s1Idx,curr_actualSeq, curr_actualSeq_q, sparse_mode):
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

    return emptyFlag, k_sparse, v_sparse, torch.tensor(s2_sparse)


def softmax(x):
    x = x.to(torch.float32)
    x_max = x.max(dim=-1, keepdim=True)[0]
    x_sub = x - x_max
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdim=True)
    ans = y / x_sum
    
    return ans, x_max, x_sum


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
    out_shape_bnsd[-1] = out_shape_bnsd[-1] - fa_param['rope_head_dim']
    sparse_mode = fa_param['sparse_mode']
    g = numheads // numKeyValueHeads

    q_bnsd_tensor = fa_param['q_bnsd_tensor']
    k_bnsd_tensor = fa_param['k_bnsd_tensor']
    v_bnsd_tensor = fa_param['v_bnsd_tensor']
    y = np.zeros(out_shape_bnsd, dtype=np.float32)
    softmax_max = None
    softmax_sum = None
    if fa_param['return_softmax'] == True:
        softmax_max = np.zeros(fa_param['softmax_max_shape'], dtype=np.float32)
        softmax_sum = np.zeros(fa_param['softmax_sum_shape'], dtype=np.float32)

    for batch in range(batch_size):
        curr_actualSeq_q = actualSeqLengths_q[batch]
        curr_actualSeq = actualSeqLengths_kv[batch]
        qLastFrefill = 0 if batch == 0 else sum(actualSeqLengths_q[0: batch])

        for n2Idx in range(numKeyValueHeads):
            for s1Idx in range(curr_actualSeq_q):
                if s1Idx < curr_actualSeq_q - curr_actualSeq and sparse_mode != 0:
                    y[batch, n2Idx * g: (n2Idx + 1) * g, s1Idx, :] = np.zeros([g, out_shape_bnsd[-1]], dtype=np.float32)
                    continue
                q_curr = q_bnsd_tensor[batch, n2Idx * g: (n2Idx + 1) * g, s1Idx, :]
                sparse_indices_Indices = sparseIndicesIndices[batch, n2Idx, s1Idx, :]
                emptyFlag, k_sparse, v_sparse, s2_index = gatherKV(k_bnsd_tensor, v_bnsd_tensor, sparse_indices_Indices,
                                                         sparse_blocksize, sparse_blockcount, batch, n2Idx, s1Idx,
                                                         curr_actualSeq, curr_actualSeq_q, sparse_mode)
                if emptyFlag:
                    continue
                bmm1Res = torch.matmul(q_curr, k_sparse.T)
                scaleRes = bmm1Res * scaleValue
                if sparse_mode == 3:
                    threshold = curr_actualSeq - curr_actualSeq_q + s1Idx + 1
                    mask_index = s2_index >= threshold
                    scaleRes[:, mask_index] = -1e12
                softmax_res, x_max, x_sum = softmax(scaleRes)
                if fa_param['q_dtype'] in (torch.float16, torch.bfloat16):
                    bmm2Res = torch.matmul(softmax_res.to(fa_param['q_dtype']).to(torch.float32), v_sparse.to(torch.float32))
                else:
                    bmm2Res = torch.matmul(softmax_res, v_sparse)
                y[batch, n2Idx * g: (n2Idx + 1) * g, s1Idx, :] = bmm2Res
                if fa_param['return_softmax'] :
                    if fa_param['layout_query'] == 'BSND':
                        softmax_max[batch, n2Idx, s1Idx, :] = x_max[:, 0]
                        softmax_sum[batch, n2Idx, s1Idx, :] = x_sum[:, 0]
                    else:

                        softmax_max[n2Idx, qLastFrefill + s1Idx, :] = x_max[:, 0]
                        softmax_sum[n2Idx, qLastFrefill + s1Idx, :] = x_sum[:, 0]

    return y, softmax_max, softmax_sum


def get_attention_mask_batch_num(npu_m_shape, q_bnsd_shape):
    batch, numhead = None, None
    if len(npu_m_shape) == 2:
        s1 = npu_m_shape[0]
        s2 = npu_m_shape[1]
        return batch, numhead, s1, s2
    if len(npu_m_shape) == 3:
        batch = npu_m_shape[0]
        s1 = npu_m_shape[1]
        s2 = npu_m_shape[2]
        return batch, numhead, s1, s2
    if len(npu_m_shape) == 4:
        batch = npu_m_shape[0]
        numhead = npu_m_shape[1]
        s1 = npu_m_shape[2]
        s2 = npu_m_shape[3]
        return batch, numhead, s1, s2


def _t_broadcastKV_sigle(numHeads, numKeyValueHeads, kv_tensor, input_dtype):
    factor = numHeads // numKeyValueHeads
    kv_shape = kv_tensor.shape
    B = kv_shape[0]
    S = kv_shape[2]
    D = kv_shape[3]
    kv_res = np.zeros([B, numHeads, S, D], dtype=kv_tensor.dtype)
    for i in range(numHeads):
        j = i // factor
        kv_res[:, i:i + 1, :, :] = kv_tensor[:, j:j + 1, :, :]
    return kv_res, kv_res.shape


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

def deepseek_ds_pa_preprocessing(fa_param, params):
    k_np_dtype = fa_param['k_np_dtype']
    v_np_dtype = fa_param['v_np_dtype']
    k_ori_shape_list = fa_param['k_shape_old']
    numKeyValueHeads = fa_param['numKeyValueHeads']
    numHeads = fa_param['numHeads']
    kv_layout = fa_param['layout_kv']
    actualSeqLengths_kv = fa_param['actualSeqLengths_kv']
    actualSeqLengths_q = fa_param['actualSeqLengths_q']
    # k_shape_bnsd_list = fa_param['k_shape_bnsd_list']
    blockSize = fa_param['blocksize']
    blockTableShape = fa_param['bt_shape']
    block_table = fa_param['block_table']
    layoutQuery = fa_param['layout_query']
    layout_kv = fa_param['layout_kv']
    k_cache = np.zeros(fa_param['k_cache_shape']).astype(k_np_dtype)
    v_cache = np.zeros(fa_param['v_cache_shape']).astype(v_np_dtype)
    k_rope_cache = np.zeros(fa_param['k_rope_cache_shape']).astype(k_np_dtype)
    k_dtype = fa_param['k_dtype']

    B = len(fa_param['actualSeqLengths_kv'])

    if len(fa_param['k_cache_shape']) != len(fa_param['k_rope_cache_shape']):
        print('[WARNING]:k_cache_shape与k_rope_cache_shape 维数不符，输出空tensor！')
        # return torch.zeros(out_shape)
    if layout_kv in ["PA_BSND"]:  # len(fa_param['k_cache_shape']) == 3
        # import pdb; pdb.set_trace()
        k_tensor_bsnd = fa_param['k_tensor_old']
        if fa_param['k_tensor_old'].shape[1] != blockTableShape[1] * blockSize:
            k_tensor_old = fa_param['k_tensor_old']
            k_tensor_bsnd = np.zeros(
                (k_tensor_old.shape[0], blockTableShape[1] * blockSize, k_tensor_old.shape[2], k_tensor_old.shape[3]),
                dtype=k_np_dtype
            )
            k_tensor_bsnd[:, :k_tensor_old.shape[1], :, :] = k_tensor_old
        kvs = k_tensor_bsnd.shape[1]
        kvn = k_tensor_bsnd.shape[2]
        kvd = k_tensor_bsnd.shape[3]
        print(f"[INFO]k_tensor_bsnd: {k_tensor_bsnd.shape}")
        # k_tensor_bbnd[:, :kvs, :, :] = k_tensor_bsnd[:, :, :, :]

        # import pdb; pdb.set_trace()
        for b in range(B):
            for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                block_offset = block_i * blockSize
                if kv_cache_blk_id == -1:
                    continue
                else:
                    for n in range(fa_param['k_cache_shape'][2]):
                        # print(f"b:{b}, block_i:{block_i}, kv_cache_blk_id:{kv_cache_blk_id}, blockSize:{blockSize}, block_offset:{block_offset}")
                        # print(f"shape0:{k_cache[kv_cache_blk_id, 0:blockSize, n:n + 1, :].shape}, shape1:{k_tensor_bsnd[b,block_offset:(block_offset + blockSize), n:n + 1,:].shape}")
                        # import pdb; pdb.set_trace()
                        k_cache[kv_cache_blk_id, 0:blockSize, n:n + 1, :] = k_tensor_bsnd[b,
                                                                            block_offset:(
                                                                                    block_offset + blockSize), n:n + 1,
                                                                            :]

        k_rope_bsnd = fa_param['k_rope_tensor']
        if fa_param['k_rope_tensor'].shape[1] != blockTableShape[1] * blockSize:
            k_rope_tensor_old = fa_param['k_rope_tensor']
            k_rope_bsnd = np.zeros(
                (k_rope_tensor_old.shape[0], blockTableShape[1] * blockSize, k_rope_tensor_old.shape[2],
                 k_rope_tensor_old.shape[3]),
                dtype=k_np_dtype
            )
            k_rope_bsnd[:, :k_rope_tensor_old.shape[1], :, :] = k_rope_tensor_old
        kropes = k_rope_bsnd.shape[1]
        kropen = k_rope_bsnd.shape[2]
        kroped = k_rope_bsnd.shape[3]
        print(f"[INFO]k_rope_bsnd: {k_rope_bsnd.shape}")
        # k_rope_tensor_bbnd[:, :, :kropes, :] = k_rope_bsnd_tensor[:, :, :, :]

        # import pdb; pdb.set_trace()
        for b in range(B):
            for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                block_offset = block_i * blockSize
                if kv_cache_blk_id == -1:
                    continue
                else:
                    for n in range(fa_param['k_rope_cache_shape'][2]):
                        k_rope_cache[kv_cache_blk_id, 0:blockSize, n:n + 1, :] = k_rope_bsnd[b,
                                                                                 block_offset:(
                                                                                         block_offset + blockSize),
                                                                                 n:n + 1,
                                                                                 :]
    else:
        print(f"[ERROR]Wrong input layout_kv: {fa_param['layout_kv']}")
        # return torch.zeros(out_shape)
    # 将kv cache 生成新的bin文件
    k_cache_index = 5
    v_cache_index = 6
    k_rope_cache_index = 9
    tools.modify_alcnn_input_file(ids=k_cache_index, origin_index=[k_cache_index], type='tensor',
                                  mode='rewrite',
                                  tensors=k_cache,
                                  params=params,
                                  data_dtype=params['dtype_input'][1])
    tools.modify_alcnn_input_file(ids=v_cache_index, origin_index=[v_cache_index], type='tensor',
                                  mode='rewrite',
                                  tensors=k_cache,
                                  params=params,
                                  data_dtype=params['dtype_input'][2])
    tools.modify_alcnn_input_file(ids=k_rope_cache_index, origin_index=[k_rope_cache_index], type='tensor',
                                  mode='rewrite',
                                  tensors=k_rope_cache,
                                  params=params,
                                  data_dtype=params['dtype_input'][2])


def aclnn_op_func_cpu(torch_tensor_list, params):
    print("cpu执行中...")
    fa_param = get_param_fus(torch_tensor_list, params)
    if not fa_param['normal_flag']:
        return torch.zeros(fa_param['out_shape'])
    # ==========预处理=============
    # >> 1.参数预处理
    actualSeqLengths_q = fa_param['actualSeqLengths_q_raw']
    actualSeqLengths_kv = fa_param['actualSeqLengths_raw']
    layoutQuery = fa_param['layout_query']
    numHeads = fa_param['numHeads']
    numKeyValueHeads = fa_param['numKeyValueHeads']
    out_shape = fa_param['out_shape']
    k_ori_shape = copy.deepcopy(fa_param['k_shape'])
    v_ori_shape = copy.deepcopy(fa_param['v_shape'])
    k_dtype = fa_param['k_dtype']
    v_dtype = fa_param['v_dtype']
    sparse_mode = fa_param['sparse_mode']
    # fa_param['kv_layout'] = 'BNSD' if fa_param['tnd_flag'] and fa_param['pa_flag'] else layoutQuery
    # >> 3.将q、k、v、topkIndices转换为bnsd(TND时只有Q和TopK写成TND格式，其余均写成BSND格式)
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
    else:
        k_bnsd_tensor, k_bnsd_shape = _n_trans_shape_to_bnsd(fa_param['k_tensor'], fa_param['k_shape'],
                                                             fa_param['layout_kv'],
                                                             numKeyValueHeads, actualSeqLengths_kv)
        v_bnsd_tensor, v_bnsd_shape = _n_trans_shape_to_bnsd(fa_param['v_tensor'], fa_param['v_shape'],
                                                             fa_param['layout_kv'],
                                                             numKeyValueHeads, actualSeqLengths_kv)
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
    fa_param['sparse_indices_bnsd_tensor'] = sparse_indices_bnsd_tensor
    fa_param['sparse_indices_bnsd_shape'] = sparse_indices_bnsd_shape
    fa_param['return_softmax'] = params.get('return_softmax',False)
    print(f"sparse_indices_bnsd_shape, {sparse_indices_bnsd_shape}")
    fa_param['ks_max'] = 0
    # >> 4.act seq处理
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

    # >> 5.生成sparseIndices
    if params["action_type"] != "bm_output_gold":
        seed = int(time.time())
        # random.seed(seed)
        sparse_indices_bnsd_shape = fa_param['sparse_indices_bnsd_shape']
        sparse_indices_tensor = fa_param['sparse_indices_bnsd_tensor']

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
                        sparse_indices_tensor[b, n, s, :] = [-1] * (fa_param['sparse_blockcount'])
                        continue
                    valid_blocks_max = math.ceil(max(0, threshold) / fa_param['sparse_blocksize'])
                    dtype = getattr(np, fa_param['sparse_indices_dtype'])
                    block_indices = torch.randperm(valid_blocks_max - 1).numpy().astype(dtype)
                    valid_blocks_topk = min(valid_blocks_max, fa_param['sparse_blockcount'])
                    sparse_indices_tensor[b, n, s, :valid_blocks_topk - 1] = block_indices[0: valid_blocks_topk - 1]
                    sparse_indices_tensor[b, n, s, valid_blocks_topk - 1:valid_blocks_topk] = valid_blocks_max - 1
                    sparse_indices_tensor[b, n, s, valid_blocks_topk:] = [-1] * (
                                fa_param['sparse_blockcount'] - valid_blocks_topk)

        # 打印信息：kv首地址，topk值
        if fa_param['tnd_flag']:
            sparse_indices_tensor_tnd = trans_bnsd_to_layout(sparse_indices_tensor, out_shape, layoutQuery,
                                                             fa_param['actualSeqLengths_q'])
            tools.modify_alcnn_input_file(ids=3, origin_index=[3], type='tensor',
                                          mode='rewrite',
                                          tensors=torch.tensor(sparse_indices_tensor_tnd, dtype=torch.int32),
                                          params=params,
                                          data_dtype=params['dtype_input'][3])

        else:
            tools.modify_alcnn_input_file(ids=3, origin_index=[3], type='tensor',
                                          mode='rewrite',
                                          tensors=sparse_indices_tensor.transpose(0, 2, 1, 3),
                                          params=params,
                                          data_dtype=params['dtype_input'][3])

        fa_param['sparse_indices_bnsd_tensor'] = sparse_indices_tensor
        if fa_param['pa_flag']:
            # 生成blocktable
            blockSize = fa_param['blocksize']
            blockTableShape = fa_param['bt_shape']
            blockNum = fa_param['k_cache_shape'][0]
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
            tools.modify_alcnn_input_file(ids=4, origin_index=[4], type='tensor', mode='rewrite',
                                          tensors=torch.tensor(block_table, dtype=torch.int32),
                                          params=params)
            k_np_dtype = tools.get_np_dtype(params['dtype_input'][1])
            v_np_dtype = tools.get_np_dtype(params['dtype_input'][2])
            fa_param['k_np_dtype'] = k_np_dtype
            fa_param['v_np_dtype'] = v_np_dtype
            print(f"[PageAtten]Input Kdtype:{params['dtype_input'][1]} Vdtype:{params['dtype_input'][2]}")
            k_cache = np.zeros(fa_param['k_cache_shape']).astype(k_np_dtype)
            v_cache = np.zeros(fa_param['v_cache_shape']).astype(v_np_dtype)
            print('[INFO]:PA + Deepseek！')
            deepseek_ds_pa_preprocessing(fa_param, params)

    # ===计算入口===
    y_all, softmax_max, softmax_sum = _t_increattention_bnsd(fa_param)
    y_all = torch.from_numpy(y_all)
    y_all = trans_bnsd_to_layout(y_all, out_shape, layoutQuery, fa_param['actualSeqLengths_q'])
    if fa_param['return_softmax']:
        softmax_max = torch.from_numpy(softmax_max)
        softmax_sum = torch.from_numpy(softmax_sum)
        return [y_all, softmax_max, softmax_sum]
    else:
        return y_all