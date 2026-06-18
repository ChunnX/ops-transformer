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

import test
import torch
import torch_npu
import pytest
import random
import numpy as np
import math
import ctypes
import copy
import torchair
import torch.nn as nn
from torchair.configs.compiler_config import CompilerConfig

class GeneralizedQLI:
    def __init__(self, batch_size, q_seq, k_seq, q_t_size, k_t_size, q_head_num, k_head_num, head_dim, block_size, block_num, qk_dtype, weight_dtype, dequant_dtype, actual_seq_dtype, act_seq_q, act_seq_k, query_quant_mode, key_quant_mode, layout_query, layout_key, sparse_count, sparse_mode, cmp_ratio):
        self.batch_size = batch_size
        self.q_seq = q_seq
        self.k_seq = k_seq
        self.q_t_size = q_t_size
        self.k_t_size = k_t_size
        self.q_head_num = q_head_num
        self.k_head_num = k_head_num
        self.group_size = q_head_num // k_head_num
        self.head_dim = head_dim
        self.block_size = block_size
        self.block_num = block_num
        self.qk_dtype = qk_dtype
        self.weight_dtype = weight_dtype
        self.dequant_dtype = dequant_dtype
        self.actual_seq_dtype = actual_seq_dtype
        self.act_seq_q = act_seq_q
        self.act_seq_k = act_seq_k
        self.query_quant_mode = query_quant_mode
        self.key_quant_mode = key_quant_mode
        self.layout_query = layout_query
        self.layout_key = layout_key
        self.sparse_count = sparse_count
        self.sparse_mode = sparse_mode
        self.cmp_ratio = cmp_ratio
        self.w_dtype = torch.bfloat16
        self.mxfp8_flag = 0
        
        if layout_query == "BSND":
            self.out_shape = [batch_size, q_seq, k_head_num, sparse_count]
        elif layout_query == "TND":
            self.out_shape = [q_t_size, k_head_num, sparse_count]

    def cal_atten_bnsd(self):
        batch_size = self.batch_size
        qs = self.q_seq
        ks = self.k_seq
        n1 = self.q_head_num
        n2 = self.k_head_num
        actualSeqLengths_q = self.actual_seq_lengths_query
        actualSeqLengths_k = self.actual_seq_lengths_key
        q_bnsd_tensor = self.q_bnsd_tensor
        k_bnsd_tensor = self.k_bnsd_tensor
        wt_bnsd_tensor = self.wt_bnsd_tensor
        mask_tensor = self.m_tensor
        q_scale_bnsd_tensor = self.q_scale_bnsd_tensor
        k_scale_bnsd_tensor = self.k_scale_bnsd_tensor
        cmp_ratio = self.cmp_ratio
        out_shape_bnsd = list(q_bnsd_tensor.shape)
        out_shape_bnsd[1] = n2
        out_shape_bnsd[-1] = self.sparse_count

        out_shape_bnss = list(q_bnsd_tensor.shape)
        out_shape_bnss[1] = n2
        out_shape_bnss[-1] = math.floor(max(actualSeqLengths_k)/cmp_ratio)

        y = torch.full(out_shape_bnsd,-1,dtype = torch.int32)
        y_value = torch.full(out_shape_bnss,-float('inf'), dtype=torch.float32)

        for b_idx in range(batch_size):
            curr_actualSeq_q = actualSeqLengths_q[b_idx]
            curr_actualSeq_k = math.floor(actualSeqLengths_k[b_idx] /cmp_ratio)
            self.cur_actseq_q = curr_actualSeq_q
            self.cur_actseq_k = curr_actualSeq_k
            self.cur_q = q_bnsd_tensor[b_idx:(b_idx + 1), :, :curr_actualSeq_q, :]
            self.cur_k = k_bnsd_tensor[b_idx:(b_idx + 1), :, :curr_actualSeq_k, :]
            self.cur_wt = wt_bnsd_tensor[b_idx:(b_idx + 1), :, :curr_actualSeq_q, :]
            self.cur_q_scale = q_scale_bnsd_tensor[b_idx:(b_idx + 1), :, :curr_actualSeq_q, :]
            self.cur_k_scale = k_scale_bnsd_tensor[b_idx:(b_idx + 1), :, :curr_actualSeq_k]
            if self.sparse_mode != 0:
                self.cur_m = mask_tensor[b_idx:(b_idx + 1), :curr_actualSeq_q, :curr_actualSeq_k]
            self.cur_b_idx = b_idx

            if curr_actualSeq_q != 0:
                actual_selected_count = min(curr_actualSeq_k, self.sparse_count)
                if self.qk_dtype == torch.int8:
                    y[b_idx:(b_idx + 1), :, :curr_actualSeq_q, :actual_selected_count], y_value[b_idx:(b_idx + 1), :,
                                                                                        :curr_actualSeq_q,
                                                                                        :curr_actualSeq_k] = self.cal_atten_per_batch_int8(b_idx)
                elif self.qk_dtype == torch.float8_e4m3fn:
                    if self.mxfp8_flag == 0:
                        y[b_idx:(b_idx + 1), :, :curr_actualSeq_q, :actual_selected_count], y_value[b_idx:(b_idx + 1), :,
                                                                                        :curr_actualSeq_q,
                                                                                        :curr_actualSeq_k] = self.cal_atten_per_batch_fp8(b_idx)
                    else:
                        y[b_idx:(b_idx + 1), :, :curr_actualSeq_q, :actual_selected_count], y_value[b_idx:(b_idx + 1), :,
                                                                                        :curr_actualSeq_q,
                                                                                        :curr_actualSeq_k] = self.cal_atten_per_batch_mxfp8(b_idx)
            else:
                pass
        return y, y_value

    def trans_shape_to_bnsd(self, tensor, layout, headnums=None, act_seq=None, tensor_name=None):
        if layout == "BSND":
            B = tensor.shape[0]
            S = tensor.shape[1]
            N = tensor.shape[2]
            D = 1
            if len(tensor.shape) == 3:
                tensor = torch.unsqueeze(tensor, dim=-1)
            else:
                D = tensor.shape[3]
            tensor = tensor.reshape(B, S, N, D).permute(0, 2, 1, 3)
            return tensor
        elif layout == "TND":
            T = tensor.shape[0]
            N = tensor.shape[1]
            D = 1
            if len(tensor.shape) == 2:
                tensor = torch.unsqueeze(tensor, dim=-1)
            else:
                D = tensor.shape[2]
            B = len(act_seq)
            S = max(act_seq)
            new_tensor = torch.zeros((B, N, S, D), dtype=tensor.dtype)
            t_start = 0
            for b_index in range(B):
                act_s = act_seq[b_index]
                t_end = t_start + act_s
                if act_s == 0:
                    continue
                for n_index in range(N):
                    new_tensor[b_index, n_index, 0:act_s, :] = tensor[t_start:t_end, n_index, :]
                t_start += act_s
            return new_tensor
        elif layout == "PA_BSND":
            if len(tensor.shape) == 3:
                tensor = torch.unsqueeze(tensor, dim=-1)
            return tensor
        else:
            return tensor

    def trans_tnd_actseq(self,list):
        list_len = len(list)
        if list_len == 0:
            raise ValueError(f'TND情况下 act_seq需要必传')
        list_new = []
        list_new.append(list[0])
        for i in range(list_len - 1):
            new_item = list[i + 1] - list[i]
            if new_item >= 0:
                list_new.append(new_item)
            else:
                raise ValueError(f'TND情况下 act_seq_len 为非递减数列 act_seq_len={list}')
        return list_new

    def cal_atten_per_batch_mxfp8(self,b_idx):
        cur_q = self.cur_q.to(dtype = torch.float32)
        cur_k = self.cur_k.to(dtype = torch.float32)
        cur_wt = self.cur_wt.to(dtype=torch.float32)
        cur_q_scale = e8m0_to_float(self.cur_q_scale)
        cur_k_scale = e8m0_to_float(self.cur_k_scale)

        # print("cur_q.shape", cur_q.shape)
        # print("cur_q", cur_q)
        # print("cur_k.shape", cur_k.shape)
        # print("cur_k", cur_k)
        # print("cur_q_scale.shape", cur_q_scale.shape)
        # print("cur_q_scale", cur_q_scale)
        # print("cur_k_scale.shape", cur_k_scale.shape)
        # print("cur_k_scale", cur_k_scale)

        cur_q = cur_q * cur_q_scale
        cur_k = cur_k * cur_k_scale

        sparse_count = self.sparse_count
        sparse_mode = self.sparse_mode
        cmp_ratio = self.cmp_ratio

        # print("cur_q", cur_q)
        # print("cur_k", cur_k)

        qk_bmm_res = torch.bmm(
            cur_q.squeeze(0),
            cur_k.permute(0, 1, 3, 2).squeeze(0)
        ).unsqueeze(0)

        print("qk_bmm_res", qk_bmm_res)
        
        qk_relu_out = (qk_bmm_res.to(dtype=torch.float32)).clamp_min(0.0)

        brc_vmul = torch.bmm(
            cur_wt.permute(0,2,3,1).to(dtype=torch.float32).squeeze(0),
            qk_relu_out.permute(0,2,1,3).to(dtype = torch.float32).squeeze(0)
        ).unsqueeze(0)

        temp_b, temp_s1, temp_n1, temp_s2 = brc_vmul.shape
        temp_g = self.group_size
        temp_n2 = self.k_head_num
        temp_b_idx = self.cur_b_idx
        actual_selected_count = min(temp_s2, sparse_count)

        reduce_sum = brc_vmul.reshape(temp_b, temp_n2, temp_s1, temp_s2)
        reduce_sum[0, :, :, :] = reduce_sum[0, :, :, :]

        if sparse_mode == 3:
            cur_m = self.cur_m
            cur_m_broadcasted = cur_m.reshape(1, 1, temp_s1, temp_s2)
            cur_m_broadcasted = torch.broadcast_to(cur_m_broadcasted, (1, temp_n2, temp_s1, temp_s2))
            # 根据布尔矩阵置-inf
            reduce_sum[cur_m_broadcasted.to(dtype = torch.bool)] = -torch.inf
        to_be_sort_ele = reduce_sum.clone()
        to_be_sort_ele = to_be_sort_ele.to(torch.bfloat16) # 

        # 稳定排序
        b_sorted_indices = torch.full(to_be_sort_ele.shape, -1, dtype=torch.int32)

        if sparse_mode == 3:
            for i in range(temp_s1):
                row_mask = cur_m_broadcasted[0, 0, i, :].to(dtype = torch.bool)
                true_indices = torch.where(~row_mask)[0]
                row_ele = to_be_sort_ele[0, 0, i, true_indices]
                indices = torch.arange(len(row_ele), device = row_ele.device)

                sorted_vals, sorted_idx = torch.sort(
                    torch.stack([-row_ele, indices],dim=1),
                    dim=0,
                    stable=True
                )
                b_sorted_indices[0, 0, i, true_indices] = true_indices[sorted_idx[:, 0]].to(torch.int32)
        else:
            for i in range(temp_s1):
                row_ele = to_be_sort_ele[0, 0, i, :]
                indices = torch.arange(len(row_ele),device = row_ele.device)
                sorted_vals, sorted_idx = torch.sort(
                    torch.stack([-row_ele, indices],dim=1),
                    dim=0,
                    stable=True
                )
                b_sorted_indices[0, 0, i, :] = sorted_idx[:,0]
        topk_indices = b_sorted_indices[..., :actual_selected_count]
        return topk_indices, to_be_sort_ele

    def cal_atten_per_batch_fp8(self,b_idx):
        cur_q = self.cur_q
        cur_k = self.cur_k

        cur_wt = self.cur_wt.to(dtype=torch.float32)

        cur_q_scale = self.cur_q_scale.to(dtype=torch.float32)
        cur_k_scale = self.cur_k_scale.to(dtype=torch.float32).squeeze(-1)

        sparse_count = self.sparse_count
        sparse_mode = self.sparse_mode
        cmp_ratio = self.cmp_ratio
        qk_bmm_res = torch.bmm(
            cur_q.to(dtype = torch.float32).squeeze(0),
            cur_k.to(dtype = torch.float32).permute(0, 1, 3, 2).squeeze(0)
        ).unsqueeze(0)

        cur_w = cur_wt * cur_q_scale
        
        qk_relu_out = (qk_bmm_res.to(dtype=torch.float32)).clamp_min(0.0)

        brc_vmul = torch.bmm(
            cur_w.permute(0,2,3,1).to(dtype=torch.float32).squeeze(0),
            qk_relu_out.permute(0,2,1,3).to(dtype = torch.float32).squeeze(0)
        ).unsqueeze(0)

        temp_b, temp_s1, temp_n1, temp_s2 = brc_vmul.shape
        temp_g = self.group_size
        temp_n2 = self.k_head_num
        temp_b_idx = self.cur_b_idx
        actual_selected_count = min(temp_s2, sparse_count)

        reduce_sum = brc_vmul.reshape(temp_b, temp_n2, temp_s1, temp_s2)
        reduce_sum[0, :, :, :] = reduce_sum[0, :, :, :] * cur_k_scale
        


        if sparse_mode == 3:
            cur_m = self.cur_m
            cur_m_broadcasted = cur_m.reshape(1, 1, temp_s1, temp_s2)
            cur_m_broadcasted = torch.broadcast_to(cur_m_broadcasted, (1, temp_n2, temp_s1, temp_s2))
            # 根据布尔矩阵置-inf
            reduce_sum[cur_m_broadcasted.to(dtype = torch.bool)] = -torch.inf
        to_be_sort_ele = reduce_sum.clone()
        to_be_sort_ele = to_be_sort_ele.to(torch.bfloat16) # 

        # 稳定排序
        b_sorted_indices = torch.full(to_be_sort_ele.shape, -1, dtype=torch.int32)

        if sparse_mode == 3:
            for i in range(temp_s1):
                row_mask = cur_m_broadcasted[0, 0, i, :].to(dtype = torch.bool)
                true_indices = torch.where(~row_mask)[0]
                row_ele = to_be_sort_ele[0, 0, i, true_indices]
                indices = torch.arange(len(row_ele), device = row_ele.device)

                sorted_vals, sorted_idx = torch.sort(
                    torch.stack([-row_ele, indices],dim=1),
                    dim=0,
                    stable=True
                )
                b_sorted_indices[0, 0, i, true_indices] = true_indices[sorted_idx[:, 0]].to(torch.int32)
        else:
            for i in range(temp_s1):
                row_ele = to_be_sort_ele[0, 0, i, :]
                indices = torch.arange(len(row_ele),device = row_ele.device)
                sorted_vals, sorted_idx = torch.sort(
                    torch.stack([-row_ele, indices],dim=1),
                    dim=0,
                    stable=True
                )
                b_sorted_indices[0, 0, i, :] = sorted_idx[:,0]
        topk_indices = b_sorted_indices[..., :actual_selected_count]
        return topk_indices, to_be_sort_ele

    def cal_atten_per_batch_int8(self,b_idx):
        cur_q = self.cur_q
        cur_k = self.cur_k
        cur_wt = self.cur_wt.to(dtype=torch.float16)
        cur_q_scale = self.cur_q_scale.to(dtype=torch.float16)
        cur_k_scale = self.cur_k_scale.to(dtype=torch.float16).unsqueeze(-1)
        sparse_count = self.sparse_count
        sparse_mode = self.sparse_mode
        cmp_ratio = self.cmp_ratio
        qk_bmm_res = torch.bmm(
            cur_q.to(dtype = torch.int32).squeeze(0),
            cur_k.to(dtype = torch.int32).permute(0, 1, 3, 2).squeeze(0)
        ).unsqueeze(0)
        cur_w = cur_wt * cur_q_scale
        qk_relu_out = (qk_bmm_res.to(dtype=torch.float32) / 1024.0).clamp_min(0.0).to(torch.float16)
        brc_vmul = torch.bmm(
            cur_w.permute(0,2,3,1).to(dtype=torch.float32).squeeze(0),
            qk_relu_out.permute(0,2,1,3).to(dtype = torch.float32).squeeze(0)
        ).unsqueeze(0)
        temp_b, temp_s1, temp_n1, temp_s2 = brc_vmul.shape
        temp_g = self.group_size
        temp_n2 = self.k_head_num
        temp_b_idx = self.cur_b_idx
        actual_selected_count = min(temp_s2, sparse_count)
        reduce_sum = brc_vmul.reshape(temp_b, temp_n2, temp_s1, temp_s2)
        reduce_sum[0, :, :, :] = reduce_sum[0, :, :, :] * cur_k_scale

        if sparse_mode == 3:
            cur_m = self.cur_m
            cur_m_broadcasted = cur_m.reshape(1, 1, temp_s1, temp_s2)
            cur_m_broadcasted = torch.broadcast_to(cur_m_broadcasted, (1, temp_n2, temp_s1, temp_s2))
            # 根据布尔矩阵置-inf
            reduce_sum[cur_m_broadcasted.to(dtype = torch.bool)] = -torch.inf


        to_be_sort_ele = reduce_sum.clone()
        # 稳定排序
        b_sorted_indices = torch.full(to_be_sort_ele.shape, -1, dtype=torch.int32)
        if sparse_mode == 3:
            for i in range(temp_s1):
                row_mask = cur_m_broadcasted[0, 0, i, :].to(dtype = torch.bool)
                true_indices = torch.where(~row_mask)[0]
                row_ele = to_be_sort_ele[0, 0, i, true_indices]
                indices = torch.arange(len(row_ele), device = row_ele.device)

                sorted_vals, sorted_idx = torch.sort(
                    torch.stack([-row_ele, indices],dim=1),
                    dim=0,
                    stable=True
                )
                b_sorted_indices[0, 0, i, true_indices] = true_indices[sorted_idx[:, 0]].to(torch.int32)
        else:
            for i in range(temp_s1):
                row_ele = to_be_sort_ele[0, 0, i, :]
                indices = torch.arange(len(row_ele),device = row_ele.device)
                sorted_vals, sorted_idx = torch.sort(
                    torch.stack([-row_ele, indices],dim=1),
                    dim=0,
                    stable=True
                )
                b_sorted_indices[0, 0, i, :] = sorted_idx[:,0]
        topk_indices = b_sorted_indices[..., :actual_selected_count]
        return topk_indices, to_be_sort_ele

    def trans_bnsd_to_layout(self,tensor, shape, layout, act_q=None):
        # 此时的输出D轴是K轴
        if layout == "BSH":
            output = tensor.permute(0, 2, 1, 3).contiguous().view(shape)
            return output
        elif layout == "BSND":
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
            output = torch.full(size=(T, N, D), fill_value=-1, dtype=tensor.dtype)
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

    def broadcast_n_axis(self, n1, n2, temp_tensor):
        g = n1 // n2
        temp_shape = temp_tensor.shape
        B = temp_shape[0]
        S = temp_shape[2]
        D = temp_shape[3]
        modify_tensor = torch.zeros([B, n1, S, D], dtype=temp_tensor.dtype)
        for i in range(n1):
            j = i // g
            modify_tensor[:, i:i + 1, :, :] = temp_tensor[:, j:j + 1, :, :]
        return modify_tensor

    def broadcast_d_axis(self, temp_tensor):
        dg = 32
        return temp_tensor.repeat_interleave(dg, dim=-1)

    def create_mask(self, m_shape, act_k, S1):
        atten_masks = torch.zeros(tuple(m_shape), dtype=torch.uint8)
        cmp_ratio = self.cmp_ratio
        tmp_pos_orig = act_k - S1

        for i in range(S1):
            if(((tmp_pos_orig+i+1)/cmp_ratio) < 0):
               atten_masks[i,:] = 1
            else:
               atten_masks[i, math.floor((tmp_pos_orig+i+1)/cmp_ratio):] = 1
        return atten_masks

    def create_mask_right_down(self, m_shape, actualSeqLengthsQ, actualSeqLengthsK, batch):
        mask_s_q = m_shape[0]
        mask_s_kv = m_shape[1]
        cmp_ratio = self.cmp_ratio
        next_tokens_list = []
        re_mask_batch = []
        pre_tokens = 214748647
        for i in range(batch):
            if len(actualSeqLengthsQ) == 0:
                S1 = mask_s_q
            else:
                S1 = actualSeqLengthsQ[i]

            if len(actualSeqLengthsK) == 0:
                S2 = mask_s_kv
            else:
                S2 = math.floor(actualSeqLengthsK[i]/cmp_ratio)
            next_tokens = S2-S1
            next_tokens_list.append(next_tokens)
            act_k = actualSeqLengthsK[i]
            atten_masks = self.create_mask(m_shape, act_k, S1)
            re_mask_batch.append(atten_masks)
        re_mask_batch = [np.array(x) for x in re_mask_batch]
        re_mask_np = np.array(re_mask_batch, dtype=np.bool_)
        cpu_mask = torch.from_numpy(re_mask_np)
        return cpu_mask, next_tokens_list

    def forward(self, query, key, weights, query_dequant_scale, key_dequant_scale, actual_seq_lengths_query, actual_seq_lengths_key, block_table, mxfp8_flag):
        print("cpu执行中...")

        # 参数的初始化
        batch_size = self.batch_size
        q_seq = self.q_seq
        k_seq = self.k_seq
        layout_query = self.layout_query
        layout_key = self.layout_key
        sparse_count = self.sparse_count
        sparse_mode = self.sparse_mode
        out_shape = self.out_shape
        head_dim = self.head_dim
        q_head_num = self.q_head_num
        k_head_num = self.k_head_num
        q_t_size = self.q_t_size
        k_t_size = self.k_t_size
        block_size = self.block_size
        block_num = self.block_num
        q_dtype = self.qk_dtype
        weight_dtype = self.weight_dtype
        k_dtype = self.qk_dtype
        w_dtype = self.w_dtype
        dequant_dtype = self.dequant_dtype
        actual_seq_dtype = self.actual_seq_dtype
        cmp_ratio = self.cmp_ratio
        self.mxfp8_flag = mxfp8_flag

        print('cmp_ratio:', cmp_ratio)

        if layout_query == "TND":
            actual_seq_lengths_query = self.trans_tnd_actseq(actual_seq_lengths_query)
            self.actual_seq_lengths_query = torch.tensor(actual_seq_lengths_query)
            actualSeqLengths_q = self.actual_seq_lengths_query

        elif layout_query == "BSND":
            self.actual_seq_lengths_query = actual_seq_lengths_query
            actualSeqLengths_q = self.actual_seq_lengths_query

        if layout_key == "TND":
            actual_seq_lengths_key = self.trans_tnd_actseq(actual_seq_lengths_key)
            self.actual_seq_lengths_key = torch.tensor(actual_seq_lengths_key)
            actualSeqLengths_k = self.actual_seq_lengths_key

        elif layout_key == "BSND":
            self.actual_seq_lengths_key = actual_seq_lengths_key
            actualSeqLengths_k = self.actual_seq_lengths_key

        elif layout_key == "PA_BSND":
            self.actual_seq_lengths_key = actual_seq_lengths_key
            actualSeqLengths_k = self.actual_seq_lengths_key
            k_max_s2 = math.floor(max(actualSeqLengths_k)/self.cmp_ratio)
        query = query.cpu()
        key = key.cpu()
        weights = weights.cpu()
        query_dequant_scale = query_dequant_scale.cpu()
        key_dequant_scale = key_dequant_scale.cpu()

        # 将输入转化为BNSD
        ## BSND / TND -> BNSD
        q_bnsd_tensor = self.trans_shape_to_bnsd(query,
            layout_query, q_head_num, actualSeqLengths_q)

        # BSN1 -> BNS1
        q_scale_bnsd_tensor = self.trans_shape_to_bnsd(query_dequant_scale,
            layout_query, q_head_num, actualSeqLengths_q)

        ## BSND/TND/ -> BNSD
        k_bnsd_tensor = self.trans_shape_to_bnsd(key,
            layout_key, k_head_num, torch.floor(actualSeqLengths_k/cmp_ratio).to(actual_seq_dtype))

        k_scale_bnsd_tensor = self.trans_shape_to_bnsd(key_dequant_scale,
            layout_key, k_head_num, torch.floor(actualSeqLengths_k/cmp_ratio).to(actual_seq_dtype))

        ## BSN1 -> BNS1   TN1 -> BNS1
        wt_bnsd_tensor = self.trans_shape_to_bnsd(weights,
            layout_query, q_head_num, actualSeqLengths_q)

        # 将 k n2轴 广播为 n1
        if q_head_num != k_head_num:
            k_bnsd_tensor = self.broadcast_n_axis(q_head_num, k_head_num, k_bnsd_tensor)
            # mxfp8场景流程改回k*kscale，kscale也要广播
            if mxfp8_flag == 1:
                k_scale_bnsd_tensor = self.broadcast_n_axis(q_head_num, k_head_num, k_scale_bnsd_tensor)

        # mxfp8场景scale的d轴为d/32，无法做乘法，所以要广播
        if mxfp8_flag == 1:
            q_scale_bnsd_tensor = self.broadcast_d_axis(q_scale_bnsd_tensor)
            k_scale_bnsd_tensor = self.broadcast_d_axis(k_scale_bnsd_tensor)

        self.q_bnsd_tensor = q_bnsd_tensor
        self.k_bnsd_tensor = k_bnsd_tensor
        self.wt_bnsd_tensor = wt_bnsd_tensor
        self.q_scale_bnsd_tensor = q_scale_bnsd_tensor
        self.k_scale_bnsd_tensor = k_scale_bnsd_tensor
        # 生成mask, sparse_mode=3时使能
        m_shape_std = [q_bnsd_tensor.shape[2], k_bnsd_tensor.shape[2]] #m_shape应该是[s1,s2]
        batch = q_bnsd_tensor.shape[0]
        m_tensor = []
        if sparse_mode == 3:
            m_tensor, next_tokens_list = self.create_mask_right_down(m_shape_std, actualSeqLengths_q, actualSeqLengths_k, batch)
        elif sparse_mode == 0:
            pass
        else:
            raise ValueError("unsupported sparse_mode!")
        self.m_tensor = m_tensor
        y, y_value = self.cal_atten_bnsd()
        # TND & PA 需要传入out_shape为BNSD
        out_shape_bnsd = list(q_bnsd_tensor.shape)
        out_shape_bnsd[1] = k_head_num
        out_shape_bnsd[-1] = sparse_count
        y = self.trans_bnsd_to_layout(y, out_shape_bnsd, layout_query, actualSeqLengths_q)
        return y, y_value

def float_to_e8m0(x: torch.Tensor) -> torch.Tensor:
    """float -> E8M0: E = clamp(floor(log2(|x|)) + 127, 1, 254), 零值 E=0"""
    x_abs = x.float().abs()
    zero_mask = (x_abs == 0)
    x_safe = torch.where(zero_mask, torch.ones_like(x_abs), x_abs)
    biased_exp = torch.clamp(torch.floor(torch.log2(x_safe)) + 127.0, 1.0, 254.0)
    biased_exp = torch.where(zero_mask, torch.zeros_like(biased_exp), biased_exp)
    return biased_exp.to(torch.uint8).view(torch.float8_e8m0fnu)


def e8m0_to_float(x: torch.Tensor) -> torch.Tensor:
    """E8M0 -> float: E=0 -> 0.0, E=1~254 -> 2^(E-127), E=255 -> NaN"""
    raw = x.view(torch.uint8).float()
    result = torch.pow(2.0, raw - 127.0)
    result = torch.where(raw == 0, torch.zeros_like(result), result)
    result = torch.where(raw == 255, torch.full_like(result, float('nan')), result)
    return result

# class QLINetwork(nn.Module):
#     def __init__(self):
#         super(QLINetwork, self).__init__()

#     def forward(self, query, key, weights, query_dequant_scale, key_dequant_scale, query_quant_mode, key_quant_mode,
#                 actual_seq_lengths_query, actual_seq_lengths_key, block_table, layout_query, layout_key, sparse_count,
#                 sparse_mode, pre_tokens, next_tokens, query_dtype, key_dtype):
#         return torch_npu.npu_quant_lightning_indexer(query, key, weights, query_dequant_scale, key_dequant_scale,
#                                                     query_quant_mode, key_quant_mode,
#                                                     actual_seq_lengths_query=actual_seq_lengths_query,
#                                                     actual_seq_lengths_key=actual_seq_lengths_key,
#                                                     block_table=block_table,
#                                                     layout_query=layout_query,
#                                                     layout_key=layout_key,
#                                                     sparse_count=sparse_count,
#                                                     sparse_mode=sparse_mode,
#                                                     pre_tokens=pre_tokens,
#                                                     next_tokens=next_tokens,
#                                                     query_dtype=query_dtype,
#                                                     key_dtype=key_dtype)

def qli_output_single(params): #TODO:
    batch_size, q_seq, k_seq, q_t_size, k_t_size, q_head_num, k_head_num, head_dim, block_size, block_num, \
    qk_dtype, weight_dtype, dequant_dtype, actual_seq_dtype, act_seq_q, act_seq_k, query_quant_mode,key_quant_mode, \
    layout_query, layout_key, sparse_count, sparse_mode, query_datarange, key_datarange, weights_datarange,\
    q_scale_datarange, k_scale_datarange, cmp_ratio = params
    import ml_dtypes
    torch.set_printoptions(threshold=float('inf'))

    test_qli = GeneralizedQLI(batch_size, q_seq, k_seq, q_t_size, k_t_size, q_head_num, k_head_num, head_dim, block_size, block_num,
                              qk_dtype, weight_dtype, dequant_dtype, actual_seq_dtype, act_seq_q, act_seq_k, query_quant_mode,
                              key_quant_mode, layout_query, layout_key, sparse_count, sparse_mode, cmp_ratio)

    actual_seq_lengths_query = torch.tensor(np.random.uniform(q_seq, q_seq, batch_size)).to(actual_seq_dtype).cpu() \
                            if act_seq_q is None else torch.tensor(act_seq_q).to(actual_seq_dtype).cpu()
    actual_seq_lengths_key = torch.tensor(np.random.uniform(k_seq*cmp_ratio, k_seq*cmp_ratio, batch_size)).to(actual_seq_dtype).cpu() \
                            if act_seq_k is None else torch.tensor(act_seq_k).to(actual_seq_dtype).cpu()     

    # shape推导
    if layout_query == "BSND":
        query_shape = [batch_size, q_seq, q_head_num, head_dim]
        query_dequant_scale_shape = [batch_size, q_seq, q_head_num]
        weights_shape = [batch_size, q_seq, q_head_num]
    elif layout_query == "TND":
        query_shape = [q_t_size, q_head_num, head_dim]
        query_dequant_scale_shape = [q_t_size, q_head_num]
        weights_shape = [q_t_size, q_head_num]

    if layout_key == "BSND":
        key_shape = [batch_size, k_seq, k_head_num, head_dim]
        key_dequant_scale_shape = [batch_size, k_seq, k_head_num]
    elif layout_key == "TND":
        key_shape = [k_t_size, k_head_num, head_dim]
        key_dequant_scale_shape = [k_t_size, k_head_num]
    elif layout_key == "PA_BSND":
        # 以不同batch中最大seq为标准初始化key(bnsd)和key_dequant_scale(bns)
        k_max_s2 = math.floor(max(act_seq_k)/cmp_ratio)
        key_shape = [batch_size, k_head_num, k_max_s2, head_dim]
        key_dequant_scale_shape = [batch_size, k_head_num, k_max_s2]

    mxfp8_flag = 0
    if dequant_dtype == torch.float8_e8m0fnu:
        mxfp8_flag = 1
        query_dequant_scale_shape.append(head_dim // 32)
        key_dequant_scale_shape.append(head_dim // 32)

    # 随机数据生成
    query = torch.tensor(np.random.uniform(query_datarange[0], query_datarange[1], query_shape)).to(qk_dtype).cpu()
    weights = torch.tensor(np.random.uniform(weights_datarange[0], weights_datarange[1], weights_shape)).to(weight_dtype).cpu()
    key = torch.tensor(np.random.uniform(key_datarange[0], key_datarange[1], key_shape)).to(qk_dtype).cpu()
    if mxfp8_flag == 1:
        query_dequant_scale = float_to_e8m0(torch.tensor(np.random.uniform(q_scale_datarange[0], q_scale_datarange[1], query_dequant_scale_shape))).cpu()
        key_dequant_scale = float_to_e8m0(torch.tensor(np.random.uniform(k_scale_datarange[0], k_scale_datarange[1], key_dequant_scale_shape))).cpu()
    else:
        query_dequant_scale = torch.tensor(np.random.uniform(q_scale_datarange[0], q_scale_datarange[1], query_dequant_scale_shape)).to(dequant_dtype).cpu()
        key_dequant_scale = torch.tensor(np.random.uniform(k_scale_datarange[0], k_scale_datarange[1], key_dequant_scale_shape)).to(dequant_dtype).cpu()

    block_table = None
    key_cpu = key
    key_dequant_scale_cpu = key_dequant_scale

    if layout_key == "PA_BSND":
        k_max_block_num_per_batch = math.ceil(k_max_s2 / block_size) #遍历batch得到的最大的block num
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
        print('block table size:', block_table.shape)
        batch_idx = 0
        for cur_block_id_threshold in key_block_num_per_batch:
            for i_block_id in range(cur_block_id_threshold):
                block_table[batch_idx][i_block_id] = block_id_list[cur_block_id]
                cur_block_id += 1
            batch_idx += 1
        # 构建PA场景的key和key_dequant_scale
        # [batch_size, k_head_num, s2, head_dim] expand to [batch_size, k_head_num, k_max_block_num_per_batch * block_size, D head_dim]
        # s2按block_size向上对齐后
        key_expand = torch.zeros((batch_size, k_head_num, k_max_block_num_per_batch * block_size, head_dim), dtype = qk_dtype)
        key_expand[:,:,:k_max_s2,:] = key_cpu
        key = torch.zeros((block_num, block_size, k_head_num, head_dim), dtype = qk_dtype)
        key_dequant_scale_expand_shape = [batch_size, k_head_num, k_max_block_num_per_batch * block_size]
        key_dequant_scale_shape = [block_num, block_size, k_head_num]
        if mxfp8_flag == 1:
            key_dequant_scale_expand_shape.append(head_dim // 32)
            key_dequant_scale_shape.append(head_dim // 32)
        key_dequant_scale_expand = torch.zeros(key_dequant_scale_expand_shape, dtype= dequant_dtype)
        if mxfp8_flag == 1:
            key_dequant_scale_expand[:,:,:k_max_s2,:] = key_dequant_scale_cpu
        else:
            key_dequant_scale_expand[:,:,:k_max_s2] = key_dequant_scale_cpu
        key_dequant_scale = torch.zeros(key_dequant_scale_shape, dtype = dequant_dtype)
        for i_batch in range(batch_size):
            for i_block, cur_block_id in enumerate(block_table[i_batch]): # batch [i_block -> cur_block_id]
                block_start_pos = i_block * block_size
                if cur_block_id == -1:
                    continue
                else:
                    for i_n in range(k_head_num):
                        key[cur_block_id, :, i_n, :] = key_expand[i_batch, i_n, block_start_pos:block_start_pos+block_size,:]
                        if mxfp8_flag == 1:
                            key_dequant_scale[cur_block_id, :, i_n, :] = key_dequant_scale_expand[i_batch, i_n,block_start_pos:block_start_pos+block_size, :]
                        else:
                            key_dequant_scale[cur_block_id, :, i_n] = key_dequant_scale_expand[i_batch, i_n,block_start_pos:block_start_pos+block_size]
        key = key.cpu()
        key_dequant_scale = key_dequant_scale.cpu() 
        block_table = torch.from_numpy(block_table).to(dtype=torch.int32).cpu()

    # cpu golden生成
    cpu_result, topk_value = test_qli.forward(query, key_cpu, weights, query_dequant_scale, key_dequant_scale_cpu, actual_seq_lengths_query, actual_seq_lengths_key, block_table, mxfp8_flag)
    cpu_result, _ = torch.sort(cpu_result)

    # 调试日志
    def log(tensor, name):
        if tensor != None:
            print(name, tensor.shape)
            return tensor.npu()
        return None

    query = log(query, "query")
    key = log(key, "key")
    weights = log(weights, "weights")
    query_dequant_scale = log(query_dequant_scale, "query_dequant_scale")
    key_dequant_scale = log(key_dequant_scale, "key_dequant_scale")
    actual_seq_lengths_query = log(actual_seq_lengths_query, "actual_seq_lengths_query")
    actual_seq_lengths_key = log(actual_seq_lengths_key, "actual_seq_lengths_key")
    block_table = log(block_table, "block_table")

    print("query_quant_mode", query_quant_mode)
    print("key_quant_mode", key_quant_mode)
    print("layout_query", layout_query)
    print("layout_key", layout_key)
    print("sparse_count", sparse_count)
    print("sparse_mode", sparse_mode)

    # 特战队acl_graph
    # model = QLINetwork().to("npu")
    # model_compiled = torch.compile(
    #     model,
    #     fullgraph=True,
    #     backend="aot_eager",
    #     dynamic=False
    # )
    # # 录制ACLGraph（NPUGraph）
    # g = torch_npu.npu.NPUGraph()

    # with torch.npu.graph(g):
    #     npu_result = model_compiled(query, key, weights, query_dequant_scale, key_dequant_scale, query_quant_mode, key_quant_mode,
    #                                 actual_seq_lengths_query=actual_seq_lengths_query,
    #                                 actual_seq_lengths_key=actual_seq_lengths_key,
    #                                 block_table=block_table,
    #                                 layout_query=layout_query,
    #                                 layout_key=layout_key,
    #                                 sparse_count=sparse_count,
    #                                 sparse_mode=sparse_mode,
    #                                 pre_tokens=9223372036854775807,
    #                                 next_tokens=9223372036854775807,
    #                                 query_dtype=None,
    #                                 key_dtype=None)
    # # 重放ACLGraph
    # g.replay()
    # print("g.replay() end")

    # # 鹏辉acl_graph
    # config = CompilerConfig()
    # npu_backend = torchair.get_npu_backend(compiler_config=config)
    # torch._dynamo.reset()
    # config.mode = "reduce-overhead"
    # npu_mode = torch.compile(QLINetwork().npu(), fullgraph=True, backend=npu_backend, dynamic=False)
    # npu_result = npu_mode(query, key, weights, query_dequant_scale, key_dequant_scale, query_quant_mode, key_quant_mode,
    #                         actual_seq_lengths_query=actual_seq_lengths_query,
    #                         actual_seq_lengths_key=actual_seq_lengths_key,
    #                         block_table=block_table,
    #                         layout_query=layout_query,
    #                         layout_key=layout_key,
    #                         sparse_count=sparse_count,
    #                         sparse_mode=sparse_mode,
    #                         pre_tokens=9223372036854775807,
    #                         next_tokens=9223372036854775807,
    #                         query_dtype=None,
    #                         key_dtype=None)

    # pta直调
    npu_result = torch_npu.npu_quant_lightning_indexer(query, key, weights,
                                                    query_dequant_scale,
                                                    key_dequant_scale,
                                                    actual_seq_lengths_query = actual_seq_lengths_query,
                                                    actual_seq_lengths_key = actual_seq_lengths_key,
                                                    block_table = block_table,
                                                    query_quant_mode = query_quant_mode,
                                                    key_quant_mode = key_quant_mode,
                                                    layout_query = layout_query,
                                                    layout_key = layout_key,
                                                    sparse_count = sparse_count,
                                                    sparse_mode = sparse_mode)
    # npu_result = 1
    torch.npu.synchronize()
    
    torch.sort(npu_result)
    npu_result, _ = torch.sort(npu_result)
    cpu_result, _ = torch.sort(cpu_result)
    
    return cpu_result, npu_result, topk_value