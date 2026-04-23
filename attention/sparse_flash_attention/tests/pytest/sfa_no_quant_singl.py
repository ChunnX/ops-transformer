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

import test
import torch
import torch_npu
import check_valid_param
import pytest
import random
import numpy as np
import math


class GeneralizedSFA:
    def __init__(self, batch_size, q_t_size, k_t_size, block_size, q_seq, kv_seq, q_head_num, kv_head_num, head_dim, rope_dim, 
                    q_dtype, idx_dtype, sparse_block_size, sparse_block_count, kv_seq_act, layout_kv, layout_query):
        q_t_size = 10
        k_t_size = 8192*4 - 10
        block_size = 25
        self.batch_size = batch_size
        self.q_t_size = q_t_size
        self.k_t_size = k_t_size
        self.q_seq = q_seq
        self.kv_seq = kv_seq
        self.q_head_num = q_head_num
        self.kv_head_num = kv_head_num
        self.group_size = q_head_num // kv_head_num
        self.head_dim = head_dim
        self.block_size = block_size
        self.q_dtype = q_dtype
        self.k_dtype = q_dtype
        self.v_dtype = q_dtype
        self.sparse_block_count = sparse_block_count
        self.sparse_block_size = sparse_block_size
        self.layout_kv = layout_kv
        self.layout_query = layout_query

        if layout_query == "BSND":
            self.q_shape = [batch_size, q_seq, q_head_num, head_dim]
            self.sparse_indices_shape = [batch_size, q_seq, kv_head_num, sparse_block_count]
            self.q_rope_shape = [batch_size, q_seq, q_head_num, rope_dim] 
            self.tnd_flag = 0

        elif layout_query == "TND":
            self.q_shape = [q_t_size, q_head_num, head_dim]
            self.sparse_indices_shape = [q_t_size, kv_head_num, sparse_block_count]
            self.q_rope_shape = [q_t_size, q_head_num, rope_dim]
            self.tnd_flag = 1
        
        if (layout_kv == "BSND") | (layout_kv == "PA_BSND"):
            self.k_shape = [batch_size, kv_seq, kv_head_num, head_dim]
            self.k_rope_shape = [batch_size, kv_seq, kv_head_num, rope_dim]
            self.kv_tnd_flag = 0
        
        elif layout_kv == "TND":
            self.k_shape = [k_t_size, kv_head_num, head_dim]
            self.k_rope_shape = [k_t_size, kv_head_num, rope_dim]
            self.kv_tnd_flag = 1

        self.v_shape = self.k_shape
        self.out_shape = self.q_shape
        if layout_kv == "PA_BSND":
            self.bt_shape = [batch_size, kv_head_num // block_size]

        self.scaleValue = 1 / ((head_dim + rope_dim)**0.5)
        self.sparse_mode = 3


    def _t_increattention_bnsd(self):
        batch_size = self.batch_size
        numheads = self.q_head_num
        numKeyValueHeads = self.kv_head_num
        actualSeqLengths_q = self.actualSeqLengths_q_get
        actualSeqLengths_kv = self.actualSeqLengths_kv_get
        scaleValue = self.scaleValue
        sparse_blocksize = self.sparse_block_size
        sparse_blockcount = self.sparse_block_count
        sparseIndicesIndices = self.sparse_indices_bnsd_tensor
        out_shape_bnsd = self.q_bnsd_shape
        out_shape_bnsd[-1] = out_shape_bnsd[-1] - self.q_rope_shape[-1]
        sparse_mode = self.sparse_mode
        g = numheads // numKeyValueHeads

        q_bnsd_tensor = self.q_bnsd_tensor
        k_bnsd_tensor = self.k_bnsd_tensor
        v_bnsd_tensor = self.v_bnsd_tensor
        matmul_dtype = np.float32
        y = np.zeros(out_shape_bnsd, dtype=np.float32)
        q_dtype = self.q_dtype

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

                    emptyFlag, k_sparse, v_sparse = self.gatherKV(k_bnsd_tensor, v_bnsd_tensor, sparse_indices_Indices,
                                                            sparse_blocksize, sparse_blockcount, batch, n2Idx, s1Idx,
                                                            curr_actualSeq, curr_actualSeq_q, sparse_mode)
                    if emptyFlag:
                        continue
                    bmm1Res = np.matmul(q_curr, k_sparse.T, dtype=matmul_dtype)
                    scaleRes = bmm1Res * scaleValue
                    softmax_res = self.softmax(scaleRes)
                    if q_dtype == "float16":
                        bmm2Res = np.matmul(softmax_res.astype(np.float16).astype(np.float32), v_sparse, dtype=matmul_dtype)
                    elif q_dtype == "bfloat16":
                        bmm2Res = np.matmul(softmax_res.astype(tf.bfloat16.as_numpy_dtype).astype(np.float32), v_sparse,
                                            dtype=matmul_dtype)
                    else:
                        bmm2Res = np.matmul(softmax_res, v_sparse, dtype=matmul_dtype)
                    y[batch, n2Idx * g: (n2Idx + 1) * g, s1Idx, :] = bmm2Res
        return y
    
    def preprocessing(self, query, key, value, query_rope, key_rope):
        numHeads = self.q_head_num
        numKeyValueHeads = self.kv_head_num
        ## 备份拼接前的信息
        q_shape = self.q_shape
        k_shape = self.k_shape
        q_rope_shape = self.q_rope_shape
        k_rope_shape = self.k_rope_shape
        tnd_flag = self.tnd_flag
        kv_tnd_flag = self.kv_tnd_flag
        q_new_tensor, q_new_shape = self.concat_tensor(query, q_shape, query_rope,
                                                q_rope_shape, numHeads, tnd_flag)
        if q_new_tensor is not None:
            self.query = q_new_tensor
            self.q_shape = q_new_shape
        else:
            print("[ERROR]q tensor的预处理异常，输出空tensor！")

        k_new_tensor, k_new_shape = self.concat_tensor(key, k_shape, key_rope,
                                                k_rope_shape, numKeyValueHeads,
                                                kv_tnd_flag)
        if k_new_tensor is not None:
            self.key = k_new_tensor
            self.k_shape = k_new_shape
        else:
            print("[ERROR]k tensor的预处理异常，输出空tensor！")

        # return fa_param
    def gatherKV(self, k_tensor, v_tensor, sparse_indices_Indices, sparse_blocksize, sparse_blockcount, batch, n2Idx, s1Idx,curr_actualSeq, curr_actualSeq_q, sparse_mode):
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

    def softmax(self, x):
        x = x.astype(np.float32)
        x_max = x.max(axis=-1, keepdims=True)
        x_sub = x - x_max
        y = np.exp(x_sub)
        x_sum = y.sum(axis=-1, keepdims=True)
        ans = y / x_sum
        return ans

    def _n_trans_shape_to_bnsd(self, tensor, shape, layout, headnums=None, act_seq=None, tensor_name=None):
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
                if act_s == 0:
                    continue
                for n_index in range(N):
                    new_tensor[b_index, n_index, 0:act_s, :] = tensor[t_start:t_end, n_index, :]
                t_start += act_s
            return new_tensor, [B, N, S, D]
        else:
            return tensor, shape

    def pa_to_bsnd(slef, pa_in, block_table, actual_seq_lengths, pa_dtype):
        block_num, block_size, n, d = pa_in.shape
        b = len(actual_seq_lengths)
        output = torch.zeros([b, block_num * block_size // b, n, d], dtype = pa_dtype)
        output = output.numpy()
        for i in range(b):
            for j in range(actual_seq_lengths[i] // block_size):
                output[i, j * block_size: (j + 1) * block_size, 0, :] = \
                    pa_in[block_table[i][j], :, 0, :].reshape(block_size, d)
        return output

    def trans_bnsd_to_layout(self, tensor, shape, layout, act_q=None):
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
    
    def trans_tnd_actseq(self, list):
        list_len = len(list)
        list_new = []
        list_new.append(list[0])
        for i in range(list_len - 1):
            new_item = list[i + 1] - list[i]
            if new_item >= 0:
                list_new.append(new_item)
            else:
                print(f"[ERROR]trans_tnd_actseq: Wrong input actseq:{list}, in loop {i}, item {new_item} < 0")
        return list_new

    def concat_tensor(self, tensor1, shape1, tensor2, shape2, n, tnd_flag=False):
        tensor1 = tensor1.cpu()
        tensor2 = tensor2.cpu()
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

    def forward(self, query, key, value, sparse_indices, query_rope, key_rope, act_seq_q, act_seq_kv, block_table):
        print("cpu执行中...")
        numHeads = self.q_head_num
        numKeyValueHeads = self.kv_head_num
        out_shape = self.out_shape
        k_dtype = self.k_dtype
        v_dtype = self.v_dtype
        sparse_mode = self.sparse_mode
        q_shape = self.q_shape
        k_shape = self.k_shape
        v_shape = self.v_shape
        

        # >> 2.将q与qRope拼接 将k与kRope拼接
        self.preprocessing(query, key, value, query_rope, key_rope)
        query = self.query # 拼接后的query
        key = self.key
        q_shape = self.q_shape
        k_shape = self.k_shape

        value = value.cpu()
        value = value.numpy()
        sparse_indices = sparse_indices.cpu()
        sparse_indices = sparse_indices.numpy()

        # >> 3.将q、k、v、topkIndices转换为bnsd(TND时只有Q和TopK写成TND格式，其余均写成BSND格式)
        layout_query = self.layout_query
        layout_kv = self.layout_kv


        actualSeqLengths_q = act_seq_q
        actualSeqLengths_kv = act_seq_kv
        if (layout_query == "TND"):
            actualSeqLengths_q_tnd = self.trans_tnd_actseq(actualSeqLengths_q)
            q_bnsd_tensor, q_bnsd_shape = self._n_trans_shape_to_bnsd(query, q_shape, layout_query,
                                                                numHeads, actualSeqLengths_q_tnd)
        else:
            q_bnsd_tensor, q_bnsd_shape = self._n_trans_shape_to_bnsd(query, q_shape, layout_query,
                                                                numHeads, actualSeqLengths_q)
        # import pdb; pdb.set_trace()
        if (layout_kv == "TND"):
            actualSeqLengths_kv_tnd = self.trans_tnd_actseq(actualSeqLengths_kv)
            k_bnsd_tensor, k_bnsd_shape = self._n_trans_shape_to_bnsd(key, k_shape,
                                                                layout_kv,
                                                                numKeyValueHeads, actualSeqLengths_kv_tnd)
            v_bnsd_tensor, v_bnsd_shape = self._n_trans_shape_to_bnsd(value, v_shape,
                                                                layout_kv,
                                                                numKeyValueHeads, actualSeqLengths_kv_tnd)
        else:
            if (layout_kv == "PA_BSND"):
                key = self.pa_to_bsnd(key, block_table, actualSeqLengths_kv, k_dtype)
                value = self.pa_to_bsnd(value, block_table, actualSeqLengths_kv, v_dtype)
            k_bnsd_tensor, k_bnsd_shape = self._n_trans_shape_to_bnsd(key, k_shape,
                                                                layout_kv,
                                                                numKeyValueHeads, actualSeqLengths_kv)
            v_bnsd_tensor, v_bnsd_shape = self._n_trans_shape_to_bnsd(value, v_shape,
                                                                layout_kv,
                                                                numKeyValueHeads, actualSeqLengths_kv)
        if (layout_query == "TND"):
            sparse_indices_bnsd_tensor, sparse_indices_bnsd_shape = self._n_trans_shape_to_bnsd(
                sparse_indices, self.sparse_indices_shape, layout_query,
                numKeyValueHeads, actualSeqLengths_q_tnd, 'sparseIndices')
        else:
            sparse_indices_bnsd_tensor, sparse_indices_bnsd_shape = self._n_trans_shape_to_bnsd(
                sparse_indices, self.sparse_indices_shape, layout_query,
                numKeyValueHeads, actualSeqLengths_q, 'sparseIndices')

        self.q_bnsd_tensor = q_bnsd_tensor
        self.q_bnsd_shape = q_bnsd_shape
        self.k_bnsd_tensor = k_bnsd_tensor
        self.k_bnsd_shape = k_bnsd_shape
        self.v_bnsd_tensor = v_bnsd_tensor
        self.v_bnsd_shape = v_bnsd_shape
        self.sparse_indices_bnsd_tensor = sparse_indices_bnsd_tensor
        self.sparse_indices_bnsd_shape = sparse_indices_bnsd_shape
        self.ks_max = 0

        # >> 4.act seq处理
        tnd_flag = (layout_query == "TND")
        kv_tnd_flag = (layout_kv == "TND")
        self.tnd_flag = tnd_flag
        self.kv_tnd_flag = kv_tnd_flag
        if not tnd_flag:
            if len(actualSeqLengths_q) == 0:
                actualSeqLengths_q_get = [q_bnsd_shape[2]] * q_bnsd_shape[0]
            elif len(actualSeqLengths_q) == 1:
                actualSeqLengths_q_get = [actualSeqLengths_q[0]] * q_bnsd_shape[0]
            else:
                actualSeqLengths_q_get = actualSeqLengths_q
        else:
            actualSeqLengths_q_get = actualSeqLengths_q_tnd
        if not kv_tnd_flag:
            if len(actualSeqLengths_kv) == 0:
                actualSeqLengths_kv_get = [k_bnsd_shape[2]] * k_bnsd_shape[0]
            elif len(actualSeqLengths_kv) == 1:
                actualSeqLengths_kv_get = [actualSeqLengths_kv[0]] * k_bnsd_shape[0]
            else:
                actualSeqLengths_kv_get = actualSeqLengths_kv
        else:
            actualSeqLengths_kv_get = actualSeqLengths_kv_tnd
        self.actualSeqLengths_q_get = actualSeqLengths_q_get
        self.actualSeqLengths_kv_get = actualSeqLengths_kv_get
        # ===计算入口===
        y_all = self._t_increattention_bnsd()
        y_all = torch.from_numpy(y_all)
        y_all = self.trans_bnsd_to_layout(y_all, out_shape, layout_query, actualSeqLengths_q_get)
        return y_all


def output_sfa_no_quant(params):
    batch_size, q_t_size, k_t_size, block_size, q_seq, kv_seq, q_head_num, kv_head_num, head_dim, rope_dim, \
    q_dtype, idx_dtype, sparse_block_size, sparse_block_count, kv_seq_act, layout_kv, layout_query = params
    
    if layout_query == "BSND":
        query = torch.tensor(np.random.uniform(-10, 10, (batch_size, q_seq, q_head_num, head_dim))).to(q_dtype).npu()
    elif layout_query == "TND":
        query = torch.tensor(np.random.uniform(-10, 10, (t, q_head_num, head_dim))).to(q_dtype).npu()
    
    if layout_kv == "BSND":
        key = torch.tensor(np.random.uniform(-5, 10, (batch_size, kv_seq, kv_head_num, head_dim))).to(q_dtype).npu()
    elif layout_kv == "TND":
        key = torch.tensor(np.random.uniform(-10, 10, (t_k, kv_head_num, head_dim))).to(q_dtype).npu()
    elif layout_kv == "PA_BSND":
        key = torch.tensor(np.random.uniform(-5, 10, (batch_size * (kv_seq // block_size), block_size, kv_head_num, head_dim))).to(q_dtype).npu()

    value = key.clone()
    idxs = random.sample(range(kv_seq_act - q_seq + 1), sparse_block_count)
    sparse_indices = torch.tensor([idxs for _ in range(batch_size * q_seq * kv_head_num)]).\
                    reshape(batch_size, q_seq, kv_head_num, sparse_block_count).to(idx_dtype).npu()
    
    query_rope = torch.tensor(np.random.uniform(-10, 10, (batch_size, q_seq, q_head_num, rope_dim))).to(q_dtype).npu()
    key_rope = torch.tensor(np.random.uniform(-10, 10, (batch_size, kv_seq, kv_head_num, rope_dim))).to(q_dtype).npu()

    act_seq_q = [q_seq] * batch_size
    act_seq_kv = [kv_seq_act] * batch_size   
    act_seq_q = torch.tensor(act_seq_q).to(idx_dtype).npu()
    act_seq_kv = torch.tensor(act_seq_kv).to(idx_dtype).npu()

    block_table = torch.tensor([range(batch_size * kv_seq // block_size)], dtype=torch.int32).reshape(batch_size, -1)
    scale_value = 1 / ((head_dim + rope_dim)**0.5)
    block_table = None
    attention_mode = 2
    return_softmax_lse = False


    test_sfa = GeneralizedSFA(
        batch_size, q_t_size, k_t_size, block_size, q_seq, kv_seq, q_head_num, kv_head_num, head_dim, rope_dim, 
        q_dtype, idx_dtype, sparse_block_size, sparse_block_count, kv_seq_act, layout_kv, layout_query)
    

    cpu_result = test_sfa.forward(query, key, value, sparse_indices, query_rope, key_rope, act_seq_q, act_seq_kv, block_table)

    
    #调用SFA算子
    npu_result, _ , _ = torch_npu.npu_sparse_flash_attention(query, key, value,
                                                            sparse_indices, scale_value, 
                                                            block_table=block_table, 
                                                            actual_seq_lengths_query=act_seq_q, 
                                                            actual_seq_lengths_kv=act_seq_kv,
                                                            query_rope=query_rope,  
                                                            key_rope=key_rope, 
                                                            sparse_block_size=sparse_block_size,
                                                            layout_query=layout_query,
                                                            layout_kv=layout_kv, 
                                                            sparse_mode=3, 
                                                            pre_tokens=(1<<63)-1, 
                                                            next_tokens=(1<<63)-1,
                                                            attention_mode = 2, 
                                                            return_softmax_lse = False)
    
    torch.npu.synchronize()
    
    return npu_result, cpu_result

