#!/usr/bin/python
# -*- coding: utf-8 -*-
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import os
from functools import partial
from qli_single import GeneralizedQLI
import pandas as pd 
import numpy as np
import torch
import torch_npu
import pytest
import random
import math
import ast
import custom_ops

def load_excel_test_cases(excel_file_path: str, sheetname: str):

    """
    从 Excel 文件加载测试用例。

    参数:
        excel_file_path (str): Excel 文件的路径。
        sheetname (str, optional): 工作表名称。若未提供，则默认 'Sheet1'。

    返回:
        list[tuple]: 测试用例元组列表，每个元组包含 20+ 个字段。
                       若失败或跳过，则返回空列表。
    """
    # 优先使用传入的 sheetname，否则尝试从环境变量获取
    if sheetname is None:
        sheetname = 'Sheet1'

    # 检查文件是否存在
    if not os.path.exists(excel_file_path):
        pytest.skip(f"Excel file not found: {excel_file_path}", allow_module_level=True)

    try:
        # 读取 Excel 文件的指定 sheet
        df = pd.read_excel(excel_file_path, sheet_name=sheetname)
        df = df.replace({np.nan: None, pd.NA: None})

        # 定义必需的列名
        required_columns = [
            'Testcase_Name', 'batch_size', 'q_seq','k_seq','q_t_size', 'k_t_size',
            'q_head_num', 'k_head_num', 'head_dim', 'block_size', 'block_num',
            'qk_dtype', 'dequant_dtype', 'actual_seq_dtype', 'act_seq_q', 'act_seq_k',
            'query_quant_mode', 'key_quant_mode', 'layout_query','layout_key','sparse_count',
            'sparse_mode', 'query_datarange','key_datarange','weights_datarange','q_scale_datarange',
            'k_scale_datarange','cmp_ratio'
        ]

        # 检查是否缺少必要列
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            pytest.skip(f"Missing required columns in Excel: {missing_cols}", allow_module_level=True)

        # 构建测试用例列表
        test_cases = []
        for _, row in df.iterrows():
            test_cases.append((
                row['Testcase_Name'],
                row['batch_size'],
                row['q_seq'],
                row['k_seq'],
                row['q_t_size'],
                row['k_t_size'],
                row['q_head_num'],
                row['k_head_num'],
                row['head_dim'],
                row['block_size'],
                row['block_num'],
                row['qk_dtype'],
                row['dequant_dtype'],
                row['actual_seq_dtype'],
                row['act_seq_q'],
                row['act_seq_k'],
                row['query_quant_mode'],
                row['key_quant_mode'],
                row['layout_query'],
                row['layout_key'],
                row['sparse_count'],
                row['sparse_mode'],
                row['query_datarange'],
                row['key_datarange'],
                row['weights_datarange'],
                row['q_scale_datarange'],
                row['k_scale_datarange'],
                row['cmp_ratio']
            ))

        return test_cases

    except Exception as e:
        pytest.skip(f"Failed to read Excel file: {e}", allow_module_level=True)
        return None
def _get_data_from_pa_cache(key, block_table, act_s2):
    block_num, block_size, n2, d = key.shape
    if n2 != 1:
        raise ValueError("n2 only support 1")
    need_blcok_num = (act_s2 + block_size - 1) // block_size
    act_s2_align = need_blcok_num * block_size
    out = torch.zeros((act_s2_align, d), dtype=key.dtype, device=key.device)
    for i in range(need_blcok_num):
        out[i * block_size:(i + 1) * block_size, :] = key[block_table[i], ...].reshape(block_size, d)

    return out[:act_s2, :]
def _get_k_scale(key_dequant_scale, block_table, act_s2):
    block_num, block_size, n2 = key_dequant_scale.shape
    if n2 != 1:
        raise ValueError("n2 only support 1")
    need_blcok_num = (act_s2 + block_size - 1) // block_size
    act_s2_align = need_blcok_num * block_size
    out = torch.zeros((act_s2_align), dtype=key_dequant_scale.dtype, device=key_dequant_scale.device)
    key_dequant_scale = key_dequant_scale.reshape(block_num, block_size)
    for i in range(need_blcok_num):
        out[i * block_size:(i + 1) * block_size] = key_dequant_scale[block_table[i], ...].reshape(block_size)

    return out[:act_s2]
def _lightning_indexer_quant(query, key, weights, query_dequant_scale, key_dequant_scale, actual_seq_lengths_query,
                             actual_seq_lengths_key, block_table,
                             layout_query="BSND", sparse_count=2048, sparse_mode=3, cmpRatio = 4):
    batch_size = query.shape[0]
    if layout_query == "TND":
        batch_size = actual_seq_lengths_query.shape[0]
	
    out_shape = list(query.shape)
    n2 = key.shape[2]
    d = query.shape[-1]
    n1 = query.shape[-2]
    out_shape[-1] = sparse_count
    out_shape[-2] = n2
    # 初始化为全-1
    out = torch.zeros(out_shape, dtype=torch.int32, device=query.device).reshape(-1, n2, sparse_count) - 1
    topk_value = torch.full(
        size=out_shape,
        fill_value=float('-inf'),
        dtype=torch.float32,  
        device=query.device
    ).reshape(-1, n2, sparse_count)

    act_s1 = 0
    act_s2 = 0
    process_q_len = 0
    for batch_id in range(batch_size):
        if actual_seq_lengths_query is None:
            # 只能为BSND格式
            act_s1 = query.shape[1]
        else:
            if layout_query == "TND": # TND格式时actual_seq_lengths_query为前缀和
                act_s1 = actual_seq_lengths_query[batch_id] - process_q_len
            else:
                act_s1 = actual_seq_lengths_query[batch_id]
        act_s2_origin = actual_seq_lengths_key[batch_id]
        act_s2 = math.floor(act_s2_origin / cmpRatio)

        # print(f"================ act_s2 : {act_s2}====================")
        # n1, s1, d
        now_q = query.reshape(-1, n1, d)[process_q_len:process_q_len + act_s1, :, :].transpose(0, 1).to(torch.float32)
        now_weights = weights.reshape(-1, n1, 1)[process_q_len:process_q_len + act_s1, :, :].to(torch.float32)
        now_query_scale = query_dequant_scale.reshape(-1, n1, 1)[process_q_len:process_q_len + act_s1, :, :]
        # s1, n1, 1
        weights_scale = now_weights * now_query_scale # float16 相乘
        process_q_len += act_s1
        now_block_table = block_table[batch_id, :]
        # d s2
        now_k = _get_data_from_pa_cache(key, now_block_table, act_s2).transpose(0, 1).to(torch.float32)
        # s2
        now_k_scale = _get_k_scale(key_dequant_scale, now_block_table, act_s2).to(torch.float32)
        # n1,s1,d @ d,s2 -> n1,s1,s2
        s_out = (torch.maximum(torch.matmul(now_q, now_k), torch.tensor(0)).to(torch.float32)) / 1024.0
        # n1,s1,s2 -> s1,n1,s2  to fp16 降精度与kernel保持一致
        s_out = s_out.transpose(0, 1)
        # s1,n1,1 -> s1,1,n1
        weights_scale = weights_scale.transpose(1, 2)
        # s1,1,n1 @ s1,n1,s2 -> s1,1,s2 -> s1,s2
        topk_in = torch.bmm(weights_scale, s_out).squeeze(1)
        # s1,s2 * s2
        topk_in = topk_in * now_k_scale
        # sparse场景下三角置为-inf
        tmp_s1 = topk_in.shape[0]
        tmp_s2 = topk_in.shape[1]
        # print(f"=================tmp_s2 : {tmp_s2}=====================\n")
        tmp_pos = tmp_s2*cmpRatio - tmp_s1
        tmp_pos_orig = act_s2_origin - tmp_s1
        if (sparse_mode == 3):
            for i in range(tmp_s1):
                # topk_in[-1 - i, tmp_s2 - i:] = float('-inf')
                topk_in[i, math.floor((tmp_pos_orig + i + 1)/cmpRatio):] = float('-inf')
        # print(f"=================================== topk_in: {topk_in[:,-20:]}==============================")
        sorted_value, sorted_indices = torch.sort(topk_in, dim=1, descending=True, stable=True)
        if (sparse_mode == 3):
            for i in range(tmp_s1):
                # sorted_indices[-1 - i, tmp_s2 - i:] = -1
                sorted_indices[i, math.floor((tmp_pos_orig + i + 1)/cmpRatio):] = -1
        # print(f"=================================== sorted_indices: {sorted_indices[:,-20:]}==============================")
        # print(f"=================================== sorted_indices shape : {sorted_indices.shape[0]}\{sorted_indices[1]} ")
        return_s2 = min(sparse_count, tmp_s2)
        out[process_q_len - act_s1:process_q_len, 0, :return_s2] = sorted_indices.to(torch.int32)[:, :return_s2]
        topk_value[process_q_len - act_s1:process_q_len, 0, :return_s2] = sorted_value.to(torch.float32)[:, :return_s2]

    out = out.reshape(out_shape)
    topk_value = topk_value.reshape(out.shape)
    return out, topk_value
def qli_output_single(data_case):
    casename = data_case[0]
    params = data_case[1:]
    print("params:", params)
    batch_size, q_seq, k_seq, q_t_size, k_t_size, q_head_num, k_head_num, head_dim, block_size, block_num, \
    qk_dtype, dequant_dtype, actual_seq_dtype, act_seq_q, act_seq_k, query_quant_mode,key_quant_mode, layout_query, \
    layout_key, sparse_count, sparse_mode, query_datarange, key_datarange, weights_datarange,q_scale_datarange, \
    k_scale_datarange, cmp_ratio = params

    if qk_dtype == 'INT8':
        qk_dtype = torch.int8
    elif qk_dtype == 'FLOAT8_E4M3FN':
        qk_dtype = torch.float8_e4m3fn
    
    if dequant_dtype == 'FP16':
        dequant_dtype = torch.float16
    elif dequant_dtype == 'FP32':
        dequant_dtype = torch.float32

    if actual_seq_dtype == 'INT32':
        actual_seq_dtype = torch.int32

    # 处理B+1
    # print(f"===== {act_seq_q} =====")
    if isinstance(act_seq_q, int):
        act_seq_q = [act_seq_q]
    elif isinstance(act_seq_q, list):
        act_seq_q = act_seq_q
    else:
        act_seq_q = [int(x.strip()) for x in act_seq_q.split(',')]
        if layout_query == 'TND':
            if len(act_seq_q) == batch_size + 1:
                act_seq_q = act_seq_q[1:]
    # print(f"===== {act_seq_q} =====")
    if isinstance(act_seq_k, int):
        act_seq_k = [act_seq_k]
    elif isinstance(act_seq_k, list):
        act_seq_k = act_seq_k
    else:
        act_seq_k = [int(x.strip()) for x in act_seq_k.split(',')]


    query_datarange = [float(x.strip()) for x in query_datarange.split(',')]
    key_datarange = [float(x.strip()) for x in key_datarange.split(',')]
    weights_datarange = [float(x.strip()) for x in weights_datarange.split(',')]
    q_scale_datarange = [float(x.strip()) for x in q_scale_datarange.split(',')]
    k_scale_datarange = [float(x.strip()) for x in k_scale_datarange.split(',')]

    if layout_query == "BSND":
        actual_seq_lengths_query = torch.tensor(np.random.uniform(q_seq, q_seq, batch_size)).to(actual_seq_dtype).npu()
    elif layout_query == "TND":
        actual_seq_lengths_query = torch.tensor(act_seq_q).to(actual_seq_dtype).npu()
    if layout_key == "BSND":
        actual_seq_lengths_key = torch.tensor(np.random.uniform(k_seq*cmp_ratio, k_seq*cmp_ratio, batch_size)).to(actual_seq_dtype).npu()
    elif layout_key == "TND" or layout_key == "PA_BSND":
        actual_seq_lengths_key = torch.tensor(act_seq_k).to(actual_seq_dtype).npu()

    if layout_query == "BSND":
        query = torch.tensor(np.random.uniform(query_datarange[0], query_datarange[1],(batch_size, q_seq, q_head_num, head_dim))).to(qk_dtype).npu()
        query_dequant_scale = torch.tensor(np.random.uniform(q_scale_datarange[0], q_scale_datarange[1], (batch_size, q_seq, q_head_num))).to(dequant_dtype).npu()
        weights = torch.tensor(np.random.uniform(weights_datarange[0], weights_datarange[1], (batch_size, q_seq, q_head_num))).to(dequant_dtype).npu()

    elif layout_query == "TND":
        query = torch.tensor(np.random.uniform(query_datarange[0], query_datarange[1], (q_t_size, q_head_num, head_dim))).to(qk_dtype).npu()
        query_dequant_scale = torch.tensor(np.random.uniform(q_scale_datarange[0], q_scale_datarange[1], (q_t_size, q_head_num))).to(dequant_dtype).npu()
        weights = torch.tensor(np.random.uniform(weights_datarange[0], weights_datarange[1], (q_t_size, q_head_num))).to(dequant_dtype).npu()

    if layout_key == "BSND":
        key = torch.tensor(np.random.uniform(key_datarange[0], key_datarange[1], (batch_size, k_seq, k_head_num, head_dim))).to(qk_dtype).npu()
        key_dequant_scale = torch.tensor(np.random.uniform(k_scale_datarange[0], k_scale_datarange[1], (batch_size, k_seq, k_head_num))).to(dequant_dtype).npu()
        block_table = None
        cpu_result, topk_value = _lightning_indexer_quant(query.cpu(), key.cpu(), weights.cpu(), query_dequant_scale.cpu(), key_dequant_scale.cpu(),
                                          actual_seq_lengths_query.cpu(), actual_seq_lengths_key.cpu(), block_table,
                                          layout_query, sparse_count, sparse_mode, cmp_ratio)

    elif layout_key == "TND":
        key = torch.tensor(np.random.uniform(k_datarange[0], k_datarange[1], (k_t_size, k_head_num, head_dim))).to(qk_dtype).npu()
        key_dequant_scale = torch.tensor(np.random.uniform(k_scale_datarange[0], k_scale_datarange[1], (k_t_size, k_head_num))).to(dequant_dtype).npu()
        block_table = None
        cpu_result, topk_value = _lightning_indexer_quant(query.cpu(), key.cpu(), weights.cpu(), query_dequant_scale.cpu(), key_dequant_scale.cpu(),
                                          actual_seq_lengths_query.cpu(), actual_seq_lengths_key.cpu(), block_table,
                                          layout_query, sparse_count, sparse_mode, cmp_ratio)

    elif layout_key == "PA_BSND":
        # 以不同batch中最大seq为标准初始化key(bnsd)和key_dequant_scale(bns)
        k_max_s2 = math.floor(max(act_seq_k)/cmp_ratio)
        k_max_block_num_per_batch = math.ceil(k_max_s2 / block_size) #遍历batch得到的最大的block num
        key_bnsd = torch.tensor(np.random.uniform(key_datarange[0], key_datarange[1],(batch_size, k_head_num, k_max_s2, head_dim))).to(qk_dtype)
        key_dequant_scale_bns = torch.tensor(np.random.uniform(k_scale_datarange[0], k_scale_datarange[1], (batch_size, k_head_num, k_max_s2))).to(dequant_dtype)

        key_block_num_per_batch = []
        key_block_num_sum = 0
        for cur_act_k in act_seq_k:
            cur_cmp_act_k = math.floor(cur_act_k / cmp_ratio)
            cur_key_block_num = math.ceil(cur_cmp_act_k / block_size)
            key_block_num_per_batch.append(cur_key_block_num)
            key_block_num_sum += cur_key_block_num
        if block_num < key_block_num_sum:
            raise ValueError(f"key actual block num < needed block num")

        # 构建block table
        block_id_list = np.arange(block_num)
        block_id_list = np.random.permutation(block_id_list).astype(np.int32)
        cur_block_id = 0
        block_table = np.full((batch_size, k_max_block_num_per_batch), fill_value = -1, dtype=np.int32)
        batch_idx = 0
        for cur_block_id_threshold in key_block_num_per_batch:
            for i_block_id in range(cur_block_id_threshold):
                block_table[batch_idx][i_block_id] = block_id_list[cur_block_id]
                cur_block_id += 1
            batch_idx += 1
        
        # 构建PA场景的key
        # [batch_size, s2, k_head_num, head_dim] expand to [batch_size, k_max_block_num_per_batch * block_size, k_head_num, head_dim]
        key_expand = torch.zeros((batch_size, k_head_num, k_max_block_num_per_batch * block_size, head_dim), dtype = qk_dtype)
        key_expand[:,:,:k_max_s2,:] = key_bnsd
        key = torch.zeros((block_num, block_size, k_head_num, head_dim), dtype = qk_dtype)
        for i_batch in range(batch_size):
            for  i_block, cur_block_id in enumerate(block_table[i_batch]):
                block_start_pos = i_block * block_size
                if cur_block_id == -1:
                    continue
                else:
                    for i_n in range(k_head_num):
                        key[cur_block_id, :, i_n, :] = key_expand[i_batch, i_n, block_start_pos:block_start_pos+block_size,:]
        key = key.npu()


        # 构建PA场景的key_dequant_scale
        key_dequant_scale_expand = torch.zeros((batch_size, k_head_num, k_max_block_num_per_batch * block_size), dtype= dequant_dtype)
        key_dequant_scale_expand[:,:,:k_max_s2] = key_dequant_scale_bns
        key_dequant_scale = torch.zeros((block_num, block_size, k_head_num), dtype = dequant_dtype)
        for i_batch in range(batch_size):
            for i_block, cur_block_id in enumerate(block_table[i_batch]):
                block_start_pos = i_block * block_size
                if cur_block_id == -1:
                    continue
                else:
                    for i_n in range(k_head_num):
                        key_dequant_scale[cur_block_id, :, i_n] = key_dequant_scale_expand[i_batch, i_n,block_start_pos:block_start_pos+block_size]
        key_dequant_scale = key_dequant_scale.npu()
        cpu_result, topk_value = _lightning_indexer_quant(query.cpu(), key.cpu(), weights.cpu(), query_dequant_scale.cpu(), key_dequant_scale.cpu(),
                                          actual_seq_lengths_query.cpu(), actual_seq_lengths_key.cpu(), block_table,
                                          layout_query, sparse_count, sparse_mode, cmp_ratio)
        block_table = torch.from_numpy(block_table).to(dtype=torch.int32).npu()

    #关于metadata的设置
    metadata = torch_npu.npu_quant_lightning_indexer_metadata(
                                      actual_seq_lengths_query=actual_seq_lengths_query, 
                                      actual_seq_lengths_key=actual_seq_lengths_key,
                                      batch_size=batch_size, 
                                      max_seqlen_q=q_seq,
                                      num_heads_q=q_head_num, 
                                      max_seqlen_k=k_seq,
                                      num_heads_k=k_head_num, 
                                      head_dim = head_dim, 
                                      query_quant_mode=query_quant_mode, 
                                      key_quant_mode=key_quant_mode,
                                      layout_query=layout_query, 
                                      layout_key=layout_key,
                                      sparse_count=sparse_count, 
                                      sparse_mode=sparse_mode,
                                      is_fd=False, 
                                      pre_tokens=(1<<63)-1, 
                                      next_tokens=(1<<63)-1, 
                                      cmp_ratio=cmp_ratio)

    metadata = metadata.npu()
    

    output_tensors = {
        "params":params,
        "cpu_result": cpu_result,
        "topk_value": topk_value,
        "query": query.to(dtype=torch.float16),
        "key":key.to(dtype=torch.float16),
        "weights": weights,
        "query_dequant_scale": query_dequant_scale,
        "key_dequant_scale": key_dequant_scale,
        "actual_seq_lengths_query": actual_seq_lengths_query,
        "actual_seq_lengths_key": actual_seq_lengths_key,
        "block_table": block_table,
        "metadata": metadata,
        "query_quant_mode":query_quant_mode,
        "key_quant_mode":key_quant_mode,
        "layout_query":layout_query,
        "layout_key":layout_key,
        "sparse_count":sparse_count,
        "sparse_mode":sparse_mode,
        "cmp_ratio":cmp_ratio
    }
    return  casename, output_tensors

def save_test_case(test_cases, file_path):
    print("正在保存pt文件...")
    # 创建输出目录
    os.makedirs(file_path, exist_ok=True)

    for idx, case in enumerate(test_cases):
        try:
            case_name, output_tensors = qli_output_single(case)
            # 生成文件名
            input_filename = f"{case_name}.pt"
            input_filepath = os.path.join(file_path, input_filename)

            # 保存数据
            torch.save(output_tensors, input_filepath)
            print(f"测试用例已保存到: {input_filepath}")  
                    
        except Exception as e:
            print(f"[失败] 生成 pt 文件失败: {case[0]} (索引: {idx})")
            print(f"错误详情: {e}")

if __name__ == "__main__":
    filepath_excel = "./excel/qli_rdv_A5.xlsx"
    testcase =  load_excel_test_cases(filepath_excel, "Sheet1")
    filepath = "./output"
    save_test_case(testcase, filepath)

