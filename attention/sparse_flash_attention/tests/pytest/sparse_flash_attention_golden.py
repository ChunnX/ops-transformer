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


def _rand_tensor(shape, dtype, low, high):
    return torch.tensor(np.random.uniform(low, high, shape)).to(dtype)


def _build_query_tensors(B, S1, N1, D, rope_head_dim, q_type):
    query_base = torch.empty((B, S1, N1, D), dtype=torch.float32).uniform_(
        DATA_RANGE_QUERY_LEFT, DATA_RANGE_QUERY_RIGHT).to(q_type)
    query_rope = torch.empty((B, S1, N1, rope_head_dim), dtype=torch.float32).uniform_(
        DATA_RANGE_QUERY_LEFT, DATA_RANGE_QUERY_RIGHT).to(q_type)
    return torch.cat((query_base, query_rope), dim=3)


def _build_kv_tensors(kv_dim0, S2, N2, D, rope_head_dim, q_type):
    key_base = torch.empty((kv_dim0, S2, N2, D), dtype=torch.float32).uniform_(
        DATA_RANGE_KV_LEFT, DATA_RANGE_KV_RIGHT).to(q_type)
    key_rope = torch.empty((kv_dim0, S2, N2, rope_head_dim), dtype=torch.float32).uniform_(
        DATA_RANGE_ROPE_LEFT, DATA_RANGE_ROPE_RIGHT).to(q_type)
    return key_base, key_rope


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
    case_name = input_data["case_name"]
    filepath = os.path.join(output_dir, f"{case_name}.pt")
    torch.save(input_data, filepath)
    print(f"测试用例已保存到: {filepath}")
    return filepath


def generate_and_save_testdata(params, save_pt=False, save_path="", compute_golden=False):
    testcase_name = params["case_name"]
    layout_query = params["layout_query"]
    layout_kv = params["layout_kv"]
    q_type = params["q_type"]
    B = params["B"]
    S1 = params["S1"]
    S2 = params["S2"]
    N1 = params["N1"]
    N2 = params["N2"]
    D = params["D"]
    K = params["K"]
    scale_value = params["scalevalue"]
    sparse_block_size = params["sparse_blocksize"]
    rope_head_dim = params["rope_head_dim"]
    sparse_mode = params["sparsemode"]
    attention_mode = params["attention_mode"]
    return_softmax_lse = params.get("return_softmax_lse", False)
    block_size = params.get("block_size") or 256
    block_num = params.get("block_num")
    actual_seq_q = params.get("actual_seq_q")
    actual_seq_kv = params.get("actual_seq_kv")

    pa_mode = layout_kv == "PA_BSND"
    actual_seq_q_list = _resolve_actual_seq(actual_seq_q, S1, B, "actual_seq_q")
    actual_seq_kv_list = _resolve_actual_seq(actual_seq_kv, S2, B, "actual_seq_kv")

    if pa_mode:
        if block_num is None:
            raise ValueError("PA mode requires block_num")
        kv_dim0 = block_num
        S2_storage = block_size
    else:
        kv_dim0 = B
        S2_storage = S2

    query_bsnd = _build_query_tensors(B, S1, N1, D, rope_head_dim, q_type)
    key_bsnd, key_rope_bsnd = _build_kv_tensors(kv_dim0, S2_storage, N2, D, rope_head_dim, q_type)

    sparse_indices_bsnd = _build_sparse_indices(
        B, S1, N2, K, sparse_block_size,
        sparse_mode, actual_seq_q_list, actual_seq_kv_list,
    )

    if pa_mode:
        block_table = _build_pa_block_table(B, block_num, block_size, S2, actual_seq_kv_list)
        key_input = key_bsnd
        key_rope_input = key_rope_bsnd
        actual_seq_kv_tensor = torch.tensor(actual_seq_kv_list, dtype=torch.int32)
    else:
        block_table = None
        actual_seq_kv_tensor = _build_actual_seq_tensor(layout_kv, actual_seq_kv_list)
        if layout_kv == "TND":
            key_input = _to_tnd(key_bsnd, actual_seq_kv_list)
            key_rope_input = _to_tnd(key_rope_bsnd, actual_seq_kv_list)
        else:
            key_input = key_bsnd
            key_rope_input = key_rope_bsnd

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
        "case_name": testcase_name,
        "params": params,
        "input": {
            "query": query_input,
            "key": key_input,
            "value": key_input[..., :D].clone(),
            "sparse_indices": sparse_indices,
            "block_table": block_table,
            "actual_seq_lengths_query": actual_seq_q_tensor,
            "actual_seq_lengths_kv": actual_seq_kv_tensor,
            "query_rope": query_input[..., D:].clone(),
            "key_rope": key_rope_input,
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
        "cpu_output": cpu_result,
    }

    if save_pt:
        _save_test_case(input_data, save_path)
    return input_data


def concat_tensor(tensor1, shape1, tensor2, shape2, n, tnd_flag=False):
    if len(shape1) != len(shape2):
        return None, None

    t1 = tensor1
    t2 = tensor2

    if len(shape1) == 3:
        if shape1[0] != shape2[0] or shape1[1] != shape2[1]:
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
            return None, None
        concatenated_tensor = torch.cat((t1, t2), dim=3)
        concatenated_shape = [shape1[0], shape1[1], shape1[2], shape1[3] + shape2[3]]
    else:
        return None, None
    return concatenated_tensor, concatenated_shape


def preprocessing(fa_param):
    numHeads = fa_param['numHeads']
    numKeyValueHeads = fa_param['numKeyValueHeads']

    def _clone(t):
        if torch.is_tensor(t):
            return t.clone()
        return copy.deepcopy(t)

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

    k_new_tensor, k_new_shape = concat_tensor(fa_param['k_tensor'], fa_param['k_shape'],
                                               fa_param['k_rope_tensor'],
                                               fa_param['k_rope_shape'], numKeyValueHeads,
                                               fa_param['kv_tnd_flag'])
    if k_new_tensor is not None:
        fa_param['k_tensor'] = k_new_tensor
        fa_param['k_shape'] = k_new_shape

    return fa_param


def _n_trans_shape_to_bnsd(tensor, shape, layout, headnums=None, act_seq=None, tensor_name=None):
    t = tensor

    if layout in ["BSND", "PA_BSND"]:
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
    return list_new


def trans_bnsd_to_layout(tensor, shape, layout, act_q=None):
    if layout == "BSH":
        output = tensor.permute(0, 2, 1, 3).contiguous().view(shape)
        return output
    elif layout in ["BSND", "PA_BSND"]:
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


def get_param_fus(input_tensor_dict, params):
    fa_param = {}
    fa_param['normal_flag'] = True

    fa_param['actualSeqLengths_q_raw'] = params['actualseqlengths']
    fa_param['actualSeqLengths_raw'] = params['actualseqlengthskv']
    fa_param['scaleValue'] = params['scalevalue']
    fa_param['layout_query'] = params['layout_query']
    fa_param['layout_kv'] = params['layout_kv']
    fa_param['sparse_mode'] = params['sparsemode']
    fa_param['sparse_blocksize'] = params['sparse_blocksize']

    fa_param['q_shape'] = params['shape_input']['query']
    fa_param['q_dtype'] = params['dtype_input']['query']
    fa_param['q_tensor'] = input_tensor_dict['query']

    fa_param['k_shape'] = params['shape_input']['key']
    fa_param['k_dtype'] = params['dtype_input']['key']
    fa_param['k_tensor'] = input_tensor_dict['key']

    fa_param['v_shape'] = fa_param['k_shape']
    fa_param['v_dtype'] = fa_param['k_dtype']
    fa_param['v_tensor'] = fa_param['k_tensor']

    fa_param['sparse_indices_shape'] = params['shape_input']['sparse_indices']
    fa_param['sparse_indices_dtype'] = params['dtype_input']['sparse_indices']
    fa_param['sparse_indices_tensor'] = input_tensor_dict['sparse_indices']

    fa_param['bt_shape'] = params['shape_input']['block_table']
    fa_param['bt_dtype'] = params['dtype_input']['block_table']
    fa_param['block_table'] = input_tensor_dict['block_table']

    fa_param['q_rope_shape'] = params['shape_input']['query_rope']
    fa_param['q_rope_dtype'] = params['dtype_input']['query_rope']
    fa_param['q_rope_tensor'] = input_tensor_dict['query_rope']

    fa_param['k_rope_shape'] = params['shape_input']['key_rope']
    fa_param['k_rope_dtype'] = params['dtype_input']['key_rope']
    fa_param['k_rope_tensor'] = input_tensor_dict['key_rope']

    fa_param['out_shape'] = fa_param['q_shape']
    fa_param['out_dtype'] = params['dtype_output'][0]

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
        fa_param['blocksize'] = params['block_size']

    fa_param['pa_flag'] = True if fa_param['layout_kv'] in ["PA_BSND"] else False
    fa_param['tnd_flag'] = True if fa_param['layout_query'] in ["TND"] else False
    fa_param['kv_tnd_flag'] = True if fa_param['layout_kv'] in ["TND"] else False

    if 0 in fa_param['q_shape']:
        fa_param['normal_flag'] = False
    if len(fa_param['k_shape']) == 0 or 0 in fa_param['k_shape']:
        fa_param['normal_flag'] = False
    if len(fa_param['v_shape']) == 0 or 0 in fa_param['v_shape']:
        fa_param['normal_flag'] = False

    return fa_param


def generate_input_tensors(params):
    q_dtype_str = params['dtype_input'].get('query', 'bf16')
    dtype_map = {'bf16': torch.bfloat16, 'bfloat16': torch.bfloat16,
                 'fp16': torch.float16, 'float16': torch.float16,
                 'fp32': torch.float32, 'float32': torch.float32}
    q_type = dtype_map.get(q_dtype_str, torch.bfloat16)

    B = params.get("B", 1)
    S1 = params["shape_input"]["query"][1] if params["layout_query"] == "BSND" else params["shape_input"]["query"][0]
    S2 = params["shape_input"]["key"][1] if params["layout_kv"] == "BSND" else params["shape_input"]["key"][0]

    actual_seq_q_list = _resolve_actual_seq(params["actualseqlengths"], S1, B, "actual_seq_q")
    actual_seq_kv_list = _resolve_actual_seq(params["actualseqlengthskv"], S2, B, "actual_seq_kv")

    query_bsnd = _build_query_tensors(B, S1, params["N1"], params["D"], params["rope_head_dim"], q_type)

    if params["layout_kv"] == "PA_BSND":
        block_num_sum = sum(math.ceil(seq / params["block_size"]) for seq in actual_seq_kv_list)
        key_bsnd, key_rope_bsnd = _build_kv_tensors(
            block_num_sum, params["block_size"], params["N2"], params["D"], params["rope_head_dim"], q_type
        )
        block_table = _build_pa_block_table(B, params.get("block_num"), params["block_size"], S2, actual_seq_kv_list)
        actual_seq_kv_tensor = torch.tensor(actual_seq_kv_list, dtype=torch.int32)
    else:
        key_bsnd, key_rope_bsnd = _build_kv_tensors(B, S2, params["N2"], params["D"], params["rope_head_dim"], q_type)
        block_table = None
        actual_seq_kv_tensor = _build_actual_seq_tensor(params["layout_kv"], actual_seq_kv_list)

    sparse_indices_bsnd = _build_sparse_indices(
        B, S1, params["N2"], params["K"], params["sparse_blocksize"],
        params["sparsemode"], actual_seq_q_list, actual_seq_kv_list,
    )

    if params["layout_query"] == "TND":
        query_input = _to_tnd(query_bsnd, actual_seq_q_list)
        sparse_indices = _to_tnd_sparse_indices(sparse_indices_bsnd, actual_seq_q_list)
    else:
        query_input = query_bsnd
        sparse_indices = sparse_indices_bsnd

    if params["layout_kv"] == "TND":
        key_input = _to_tnd(key_bsnd, actual_seq_kv_list)
        key_rope_input = _to_tnd(key_rope_bsnd, actual_seq_kv_list)
    else:
        key_input = key_bsnd
        key_rope_input = key_rope_bsnd

    actual_seq_q_tensor = _build_actual_seq_tensor(params["layout_query"], actual_seq_q_list)

    return {
        "query": query_input,
        "key": key_input,
        "value": key_input[..., :params["D"]].clone(),
        "sparse_indices": sparse_indices,
        "block_table": block_table,
        "actual_seq_lengths_query": actual_seq_q_tensor,
        "actual_seq_lengths_kv": actual_seq_kv_tensor,
        "query_rope": query_input[..., params["D"]:].clone(),
        "key_rope": key_rope_input,
        "scale_value": params["scalevalue"],
        "sparse_block_size": params["sparse_blocksize"],
        "layout_query": params["layout_query"],
        "layout_kv": params["layout_kv"],
        "sparse_mode": params["sparsemode"],
        "attention_mode": params.get("attention_mode", 2),
        "return_softmax_lse": params.get("return_softmax_lse", False),
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


def compute_golden(input_tensor_dict, params):
    def _to_cpu_float32_tensor(val):
        if torch.is_tensor(val):
            return val.cpu().float()
        return val

    cpu_input = {key: _to_cpu_float32_tensor(val) for key, val in input_tensor_dict.items()}
    fa_param = _prepare_fa_param(cpu_input, params)
    if not fa_param['normal_flag']:
        return torch.zeros(fa_param['out_shape'])

    out_shape = fa_param['out_shape']
    layoutQuery = fa_param['layout_query']

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
        threshold = delta_s + s1_idx + 1
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
    out_shape_bnsd = fa_param['q_bnsd_shape'].copy()
    out_shape_bnsd[-1] = fa_param['v_bnsd_shape'][-1]
    sparse_mode = fa_param['sparse_mode']
    g = numheads // numKeyValueHeads

    q_bnsd_tensor = fa_param['q_bnsd_tensor'].float()
    k_bnsd_tensor = fa_param['k_bnsd_tensor'].float()
    v_bnsd_tensor = k_bnsd_tensor[..., :512].clone()

    y = torch.zeros(out_shape_bnsd, dtype=torch.float32)

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
                bmm2Res = torch.matmul(softmax_res, v_sparse.float())
                y[batch, n2Idx * g: (n2Idx + 1) * g, s1Idx, :] = bmm2Res
    return y