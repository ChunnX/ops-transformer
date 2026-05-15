#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import sys
import torch_npu
import math
import itertools


import torch

def convert_q_scale_tnd_to_n2gtd_layout(tensor_tnd: torch.Tensor, num_key_value_heads: int):
    T, N, D, SS = tensor_tnd.shape

    # 计算分组数，用 torch.div 更安全（整数除法）
    group_size = torch.div(N, num_key_value_heads, rounding_mode="floor")

    # 重塑维度 T, num_kv_heads, group_size, D, 2
    tensor_reshape = tensor_tnd.reshape(T, num_key_value_heads, group_size, D, SS)

    # 转置：把 num_kv_heads 换到最前面
    tensor_transpose = tensor_reshape.permute(1, 0, 2, 3, 4)

    tensor_transpose = tensor_transpose.reshape(T, N, D, SS)

    return tensor_transpose


def test_fia_mxfp8(batch_in, sparse_mode, kvs):
    print("[INFO] FA mxfp8 start")
    N_q = 64
    N_kv = 8
    D = 128
    batch_size = batch_in
    seq_len_q = 4
    seq_len_kv = kvs
    seq_len_per_batch_q = [seq_len_q] * batch_size
    accumSeqLengthqs = list(itertools.accumulate(seq_len_per_batch_q))
    T_q = accumSeqLengthqs[-1]

    seq_len_per_batch_kv = [seq_len_kv] * batch_size
    actualSeqLengthkvsNoPA = list(itertools.accumulate(seq_len_per_batch_kv))
    T_kv = actualSeqLengthkvsNoPA[-1]

    print(f"accumSeqLengthqs: {accumSeqLengthqs}")
    print(f"actualSeqLengthkvs: {actualSeqLengthkvsNoPA}")

    query = torch.tensor(np.random.uniform(-1, 1, (T_q, N_q, D))).to(torch.float8_e4m3fn).npu()
    key = torch.tensor(np.random.uniform(-1, 1, (T_kv, N_kv, D))).to(torch.float8_e4m3fn).npu()
    value = torch.tensor(np.random.uniform(-1, 1, (T_kv, N_kv, D))).to(torch.float8_e4m3fn).npu()

    dequant_scale_query = torch.tensor(np.random.uniform(0, 1, (T_q, N_q, D // 64, 2)), dtype=torch.uint8).npu()
    dequant_scale_query = convert_q_scale_tnd_to_n2gtd_layout(dequant_scale_query, N_kv)
    key_antiquant_scale = torch.tensor(np.random.uniform(0, 1, (T_kv, N_kv, D // 64, 2)), dtype=torch.uint8).npu()
    value_antiquant_scale = torch.tensor(np.random.uniform(0, 1, (T_kv // 64, N_kv, D, 2)), dtype=torch.uint8).npu()

    print(f"query: {query.shape}")
    print(f"key  : {key.shape}")
    print(f"value: {value.shape}")
    print(f"dequant_scale_query  : {dequant_scale_query.shape}")
    print(f"key_antiquant_scale  : {key_antiquant_scale.shape}")
    print(f"value_antiquant_scale: {value_antiquant_scale.shape}")

    query_dtype = torch.float8_e4m3fn
    key_dtype = torch.float8_e4m3fn
    value_dtype = torch.float8_e4m3fn
    softmax_scale = 1 / math.sqrt(D)

    if sparse_mode == 3:
        mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool)).npu()
        out = torch_npu.npu_fused_infer_attention_score_v2(query,
                                                            key,
                                                            value,
                                                            atten_mask=mask,
                                                            actual_seq_qlen=accumSeqLengthqs,
                                                            actual_seq_kvlen=actualSeqLengthkvsNoPA,
                                                            dequant_scale_query=dequant_scale_query,
                                                            dequant_scale_key=key_antiquant_scale,
                                                            dequant_scale_value=value_antiquant_scale,
                                                            num_query_heads=N_q,
                                                            num_key_value_heads=N_kv,
                                                            softmax_scale=softmax_scale,
                                                            input_layout="TND",
                                                            sparse_mode=3,
                                                            query_quant_mode=6,
                                                            key_quant_mode=6,
                                                            value_quant_mode=8,
                                                            query_dtype=query_dtype,
                                                            key_dtype=key_dtype,
                                                            value_dtype=value_dtype,
                                                            dequant_scale_query_dtype=torch_npu.float8_e8m0fnu,
                                                            dequant_scale_key_dtype=torch_npu.float8_e8m0fnu,
                                                            dequant_scale_value_dtype=torch_npu.float8_e8m0fnu,
                                                            )
    else:
        out = torch_npu.npu_fused_infer_attention_score_v2(query,
                                                    key,
                                                    value,
                                                    actual_seq_qlen=accumSeqLengthqs,
                                                    actual_seq_kvlen=actualSeqLengthkvsNoPA,
                                                    dequant_scale_query=dequant_scale_query,
                                                    dequant_scale_key=key_antiquant_scale,
                                                    dequant_scale_value=value_antiquant_scale,
                                                    num_query_heads=N_q,
                                                    num_key_value_heads=N_kv,
                                                    softmax_scale=softmax_scale,
                                                    input_layout="TND",
                                                    sparse_mode=0,
                                                    query_quant_mode=6,
                                                    key_quant_mode=6,
                                                    value_quant_mode=8,
                                                    query_dtype=query_dtype,
                                                    key_dtype=key_dtype,
                                                    value_dtype=value_dtype,
                                                    dequant_scale_query_dtype=torch_npu.float8_e8m0fnu,
                                                    dequant_scale_key_dtype=torch_npu.float8_e8m0fnu,
                                                    dequant_scale_value_dtype=torch_npu.float8_e8m0fnu,
                                                    )
    # print(out)
    print("[INFO] FA mxfp8 end")

def test_fia_mxfp8_pa(batch_in, sparse_mode, kvs):
    print("[INFO] FA mxfp8 start pa")
    N_q = 64
    N_kv = 8
    D = 128
    batch_size = batch_in
    block_size = 512
    seq_len_q = 4
    seq_len_kv = kvs
    seq_len_per_batch_q = [seq_len_q] * batch_size
    accumSeqLengthqs = list(itertools.accumulate(seq_len_per_batch_q))
    actualSeqLengthkvs = [seq_len_kv] * batch_size
    T_q = accumSeqLengthqs[-1]

    block_num = sum(seq_len // block_size for seq_len in actualSeqLengthkvs)
    block_table_shape = [batch_size, max(actualSeqLengthkvs) // block_size]
    block_idx_list = np.arange(block_num)
    block_idx_list = np.random.permutation(block_idx_list)
    block_idx = 0
    block_table = np.full(block_table_shape, -1).astype(np.int32)
    blockNumPerBlock = []
    for actual_seq in actualSeqLengthkvs:
        blockNumPerBlock.append(math.ceil(actual_seq / block_size))
    for idx, block_per_b in enumerate(blockNumPerBlock):
        for j in range(block_per_b):
            block_table[idx][j] = (block_idx_list[block_idx])
            block_idx += 1
    block_table = torch.from_numpy(block_table).npu()
    print(f"block_table: {block_table.shape}")
    print(f"accumSeqLengthqs: {accumSeqLengthqs}")
    print(f"actualSeqLengthkvs: {actualSeqLengthkvs}")

    # pa
    query = torch.tensor(np.random.uniform(-1, 1, (T_q, N_q, D))).to(torch.float8_e4m3fn).npu()
    key = torch.tensor(np.random.uniform(-1, 1, (block_num, N_kv, block_size, D))).to(torch.float8_e4m3fn).npu()
    value = torch.tensor(np.random.uniform(-1, 1, (block_num, N_kv, block_size, D))).to(torch.float8_e4m3fn).npu()

    dequant_scale_query = torch.tensor(np.random.uniform(0, 1, (T_q, N_q, D // 64, 2)), dtype=torch.uint8).npu()
    dequant_scale_query = convert_q_scale_tnd_to_n2gtd_layout(dequant_scale_query, N_kv)
    key_antiquant_scale = torch.tensor(np.random.uniform(0, 1, (block_num, N_kv, block_size, D // 64, 2)), dtype=torch.uint8).npu()
    value_antiquant_scale = torch.tensor(np.random.uniform(0, 1, (block_num, N_kv, block_size // 64, D, 2)), dtype=torch.uint8).npu()

    print(f"query: {query.shape}")
    print(f"key  : {key.shape}")
    print(f"value: {value.shape}")
    print(f"dequant_scale_query  : {dequant_scale_query.shape}")
    print(f"key_antiquant_scale  : {key_antiquant_scale.shape}")
    print(f"value_antiquant_scale: {value_antiquant_scale.shape}")

    query_dtype = torch.float8_e4m3fn
    key_dtype = torch.float8_e4m3fn
    value_dtype = torch.float8_e4m3fn
    softmax_scale = 1 / math.sqrt(D)

    if sparse_mode == 3:
        mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool)).npu()
        out = torch_npu.npu_fused_infer_attention_score_v2(query,
                                                            key,
                                                            value,
                                                            atten_mask=mask,
                                                            actual_seq_qlen=accumSeqLengthqs,
                                                            actual_seq_kvlen=actualSeqLengthkvs,
                                                            dequant_scale_query=dequant_scale_query,
                                                            dequant_scale_key=key_antiquant_scale,
                                                            dequant_scale_value=value_antiquant_scale,
                                                            num_query_heads=N_q,
                                                            num_key_value_heads=N_kv,
                                                            block_size=block_size,
                                                            block_table=block_table,
                                                            softmax_scale=softmax_scale,
                                                            input_layout="TND",
                                                            sparse_mode=3,
                                                            query_quant_mode=6,
                                                            key_quant_mode=6,
                                                            value_quant_mode=8,
                                                            query_dtype=query_dtype,
                                                            key_dtype=key_dtype,
                                                            value_dtype=value_dtype,
                                                            dequant_scale_query_dtype=torch_npu.float8_e8m0fnu,
                                                            dequant_scale_key_dtype=torch_npu.float8_e8m0fnu,
                                                            dequant_scale_value_dtype=torch_npu.float8_e8m0fnu,
                                                            )
    else:
        out = torch_npu.npu_fused_infer_attention_score_v2(query,
                                                    key,
                                                    value,
                                                    actual_seq_qlen=accumSeqLengthqs,
                                                    actual_seq_kvlen=actualSeqLengthkvs,
                                                    dequant_scale_query=dequant_scale_query,
                                                    dequant_scale_key=key_antiquant_scale,
                                                    dequant_scale_value=value_antiquant_scale,
                                                    num_query_heads=N_q,
                                                    num_key_value_heads=N_kv,
                                                    block_size=block_size,
                                                    block_table=block_table,
                                                    softmax_scale=softmax_scale,
                                                    input_layout="TND",
                                                    sparse_mode=0,
                                                    query_quant_mode=6,
                                                    key_quant_mode=6,
                                                    value_quant_mode=8,
                                                    query_dtype=query_dtype,
                                                    key_dtype=key_dtype,
                                                    value_dtype=value_dtype,
                                                    dequant_scale_query_dtype=torch_npu.float8_e8m0fnu,
                                                    dequant_scale_key_dtype=torch_npu.float8_e8m0fnu,
                                                    dequant_scale_value_dtype=torch_npu.float8_e8m0fnu,
                                                    )
    # print(out)
    print("[INFO] FA mxfp8 pa end")


def test_fia_mxfp8_pa_nz(batch_in, sparse_mode, kvs):
    print("[INFO] FA mxfp8 start pa")
    N_q = 64
    N_kv = 8
    D = 128
    batch_size = batch_in
    block_size = 512
    seq_len_q = 4
    seq_len_kv = kvs
    seq_len_per_batch_q = [seq_len_q] * batch_size
    accumSeqLengthqs = list(itertools.accumulate(seq_len_per_batch_q))
    actualSeqLengthkvs = [seq_len_kv] * batch_size
    T_q = accumSeqLengthqs[-1]

    block_num = sum(seq_len // block_size for seq_len in actualSeqLengthkvs)
    block_table_shape = [batch_size, max(actualSeqLengthkvs) // block_size]
    block_idx_list = np.arange(block_num)
    block_idx_list = np.random.permutation(block_idx_list)
    block_idx = 0
    block_table = np.full(block_table_shape, -1).astype(np.int32)
    blockNumPerBlock = []
    for actual_seq in actualSeqLengthkvs:
        blockNumPerBlock.append(math.ceil(actual_seq / block_size))
    for idx, block_per_b in enumerate(blockNumPerBlock):
        for j in range(block_per_b):
            block_table[idx][j] = (block_idx_list[block_idx])
            block_idx += 1
    block_table = torch.from_numpy(block_table).npu()
    print(f"block_table: {block_table}")
    print(f"accumSeqLengthqs: {accumSeqLengthqs}")
    print(f"actualSeqLengthkvs: {actualSeqLengthkvs}")

    # pa
    query = torch.tensor(np.random.uniform(-1, 1, (T_q, N_q, D))).to(torch.float8_e4m3fn).npu()
    key = torch.tensor(np.random.uniform(-1, 1, (block_num, N_kv, D // 32, block_size, 32))).to(torch.float8_e4m3fn).npu()
    value = torch.tensor(np.random.uniform(-1, 1, (block_num, N_kv, D // 32, block_size, 32))).to(torch.float8_e4m3fn).npu()

    dequant_scale_query = torch.tensor(np.random.uniform(0, 1, (T_q, N_q, D // 64, 2)), dtype=torch.uint8).npu()
    dequant_scale_query = convert_q_scale_tnd_to_n2gtd_layout(dequant_scale_query, N_kv)
    key_antiquant_scale = torch.tensor(np.random.uniform(0, 1, (block_num, N_kv, block_size // 16, D // 64, 16, 2)), dtype=torch.uint8).npu()
    value_antiquant_scale = torch.tensor(np.random.uniform(0, 1, (block_num, N_kv, D//16, block_size // 64, 16, 2)), dtype=torch.uint8).npu()

    print(f"query: {query.shape}")
    print(f"key  : {key.shape}")
    print(f"value: {value.shape}")
    print(f"dequant_scale_query  : {dequant_scale_query.shape}")
    print(f"key_antiquant_scale  : {key_antiquant_scale.shape}")
    print(f"value_antiquant_scale: {value_antiquant_scale.shape}")

    query_dtype = torch.float8_e4m3fn
    key_dtype = torch.float8_e4m3fn
    value_dtype = torch.float8_e4m3fn
    softmax_scale = 1 / math.sqrt(D)

    if sparse_mode == 3:
        mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool)).npu()
        out = torch_npu.npu_fused_infer_attention_score_v2(query,
                                                            key,
                                                            value,
                                                            atten_mask=mask,
                                                            actual_seq_qlen=accumSeqLengthqs,
                                                            actual_seq_kvlen=actualSeqLengthkvs,
                                                            dequant_scale_query=dequant_scale_query,
                                                            dequant_scale_key=key_antiquant_scale,
                                                            dequant_scale_value=value_antiquant_scale,
                                                            num_query_heads=N_q,
                                                            num_key_value_heads=N_kv,
                                                            block_size=block_size,
                                                            block_table=block_table,
                                                            softmax_scale=softmax_scale,
                                                            input_layout="TND",
                                                            sparse_mode=3,
                                                            query_quant_mode=6,
                                                            key_quant_mode=6,
                                                            value_quant_mode=8,
                                                            query_dtype=query_dtype,
                                                            key_dtype=key_dtype,
                                                            value_dtype=value_dtype,
                                                            dequant_scale_query_dtype=torch_npu.float8_e8m0fnu,
                                                            dequant_scale_key_dtype=torch_npu.float8_e8m0fnu,
                                                            dequant_scale_value_dtype=torch_npu.float8_e8m0fnu,
                                                            )
    else:
        out = torch_npu.npu_fused_infer_attention_score_v2(query,
                                                    key,
                                                    value,
                                                    actual_seq_qlen=accumSeqLengthqs,
                                                    actual_seq_kvlen=actualSeqLengthkvs,
                                                    dequant_scale_query=dequant_scale_query,
                                                    dequant_scale_key=key_antiquant_scale,
                                                    dequant_scale_value=value_antiquant_scale,
                                                    num_query_heads=N_q,
                                                    num_key_value_heads=N_kv,
                                                    block_size=block_size,
                                                    block_table=block_table,
                                                    softmax_scale=softmax_scale,
                                                    input_layout="TND",
                                                    sparse_mode=0,
                                                    query_quant_mode=6,
                                                    key_quant_mode=6,
                                                    value_quant_mode=8,
                                                    query_dtype=query_dtype,
                                                    key_dtype=key_dtype,
                                                    value_dtype=value_dtype,
                                                    dequant_scale_query_dtype=torch_npu.float8_e8m0fnu,
                                                    dequant_scale_key_dtype=torch_npu.float8_e8m0fnu,
                                                    dequant_scale_value_dtype=torch_npu.float8_e8m0fnu,
                                                    )
    # print(out)
    print("[INFO] FA mxfp8 pa end")

if __name__ == '__main__':
    # sparse_mode = int(sys.argv[1])
    # kvs = int(sys.argv[2])
    # batch_in = int(sys.argv[3])
    # print(f"[INFO] sparse_mode: {sparse_mode}, batch_in: {batch_in}, kvs: {kvs}")
    # test_fia_mxfp8_pa(32, 3, 10240)
    # test_fia_mxfp8(32, 3, 10240)
    # test_fia_mxfp8_pa_nz(1, 3, 512)
    test_fia_mxfp8_pa_nz(8, 3, 65536)

    # sparse_mode = 3
    # kvs = 65536
    # for b in [1, 2, 4, 8, 16, 17, 32, 63, 64, 127, 128, 256]:
    #     test_fia_mxfp8_pa_nz(b, sparse_mode, kvs)

    # sparse_mode = 3
    # kvs = 10240
    # for b in [1, 2, 4, 8, 16, 17, 32, 63, 64, 127, 128, 256, 512]:
    #     test_fia_mxfp8_pa_nz(b, sparse_mode, kvs)