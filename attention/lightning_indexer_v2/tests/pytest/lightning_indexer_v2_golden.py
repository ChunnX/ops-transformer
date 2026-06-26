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
import cann_ops_transformer
from cann_ops_transformer.ops import lightning_indexer, lightning_indexer_metadata

class GeneralizedLIV2:
    def __init__(self, batch_size, q_seq, k_seq, q_t_size, k_t_size, q_head_num, k_head_num,
                 head_dim, block_size, block_num, qk_dtype, cu_seqlens_q, cu_seqlens_k, 
                 seqused_q, seqused_k, cmp_residual_k, layout_query, layout_key, topk, max_seqlen_q, mask_mode, cmp_ratio, return_value):
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
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        self.seqused_q = seqused_q
        self.seqused_k = seqused_k
        self.cmp_residual_k = cmp_residual_k
        self.layout_query = layout_query
        self.layout_key = layout_key
        self.topk = topk
        self.max_seqlen_q = max_seqlen_q
        self.mask_mode = mask_mode
        self.cmp_ratio = cmp_ratio
        self.return_value = return_value

        if layout_query == "BSND":
            self.q_shape = [batch_size, q_seq, q_head_num, head_dim]
            self.w_shape = [batch_size, q_seq, q_head_num]
            self.q_tnd_flag = 0
        elif layout_query == "TND":
            self.q_shape = [q_t_size, q_head_num, head_dim]
            self.w_shape = [q_t_size, q_head_num]
            self.q_tnd_flag = 1

        if layout_key == "BSND":
            self.k_shape = [batch_size, k_seq, k_head_num, head_dim]
        elif layout_key == "TND":
            self.k_shape = [k_t_size, k_head_num, head_dim]

        if layout_query == "BSND":
            self.out_shape = [batch_size, q_seq, k_head_num, topk]
            self.output_idx_offset_shape = [batch_size, q_seq, k_head_num]
        elif layout_query == "TND":
            self.out_shape = [q_t_size, k_head_num, topk]
            self.output_idx_offset_shape = [q_t_size, k_head_num]

    def cal_atten_bnsd(self, output_idx_offset):
        batch_size = self.batch_size
        qs = self.q_seq
        ks = self.k_seq
        n1 = self.q_head_num
        n2 = self.k_head_num
        cu_seqlens_q = self.cu_seqlens_q
        cu_seqlens_k = self.cu_seqlens_k
        seqused_q = self.seqused_q
        seqused_k = self.seqused_k
        cmp_residual_k = self.cmp_residual_k
        q_bnsd_tensor = self.q_bnsd_tensor
        k_bnsd_tensor = self.k_bnsd_tensor
        wt_bnsd_tensor = self.wt_bnsd_tensor
        mask_tensor = self.m_tensor
        cmp_ratio = self.cmp_ratio

        out_shape_bnsd = copy.deepcopy(self.q_bnsd_shape)
        out_shape_bnsd[1] = n2
        out_shape_bnsd[-1] = self.topk

        out_shape_bnss = copy.deepcopy(self.q_bnsd_shape)
        out_shape_bnss[1] = n2
        out_shape_bnss[-1] = math.floor(max(seqused_k)) if seqused_k is not None else ks

        y = torch.full(out_shape_bnsd, -1 , dtype = torch.int32)
        y_value = torch.full(out_shape_bnss,-float('inf'), dtype=torch.float32)
        y_value_np = np.full(out_shape_bnsd, -np.inf, dtype=np.float32)

        prefix = 0
        for b_idx in range(batch_size):
            if self.layout_query == "TND":
                if seqused_q is not None:
                    curr_actualSeq_q = seqused_q[b_idx]
                else:
                    # 已被处理为shape为(B,)的tensor
                    curr_actualSeq_q = cu_seqlens_q[b_idx]
            elif self.layout_query == "BSND":
                if seqused_q is not None:
                    curr_actualSeq_q = seqused_q[b_idx]
                else:
                    curr_actualSeq_q = qs

            if self.layout_key == "TND":
                if seqused_k is not None:
                    curr_actualSeq_k = seqused_k[b_idx]
                else:
                    curr_actualSeq_k = cu_seqlens_k[b_idx]
            elif self.layout_key == "PA_BBND":
                curr_actualSeq_k = seqused_k[b_idx]
            elif self.layout_key == "BSND":
                if seqused_k is not None:
                    curr_actualSeq_k = seqused_k[b_idx]
                else:
                    curr_actualSeq_k = ks
            self.cur_actseq_q = curr_actualSeq_q
            self.cur_actseq_k = curr_actualSeq_k
            self.cur_q = q_bnsd_tensor[b_idx:(b_idx + 1), :, :curr_actualSeq_q, :]
            self.cur_k = k_bnsd_tensor[b_idx:(b_idx + 1), :, :curr_actualSeq_k, :]
            self.cur_wt = wt_bnsd_tensor[b_idx:(b_idx + 1), :, :curr_actualSeq_q, :]
            if self.mask_mode != 0:
                self.cur_m = mask_tensor[b_idx:(b_idx + 1), :curr_actualSeq_q, :curr_actualSeq_k]
            self.cur_b_idx = b_idx

            if curr_actualSeq_q != 0:
                actual_selected_count = min(curr_actualSeq_k, self.topk)
                if self.qk_dtype == torch.bfloat16:
                    y[b_idx:(b_idx + 1), :, :curr_actualSeq_q, :actual_selected_count], y_value[b_idx:(b_idx + 1), :,
                                                                                        :curr_actualSeq_q,
                                                                                        :curr_actualSeq_k] = self.cal_atten_per_batch_bf16(b_idx)
                elif self.qk_dtype == torch.float16:
                    y[b_idx:(b_idx + 1), :, :curr_actualSeq_q, :actual_selected_count], y_value[b_idx:(b_idx + 1), :,
                                                                                        :curr_actualSeq_q,
                                                                                        :curr_actualSeq_k] = self.cal_atten_per_batch_fp16(b_idx)
                y[b_idx: (b_idx + 1), :, curr_actualSeq_q:, :actual_selected_count] = -1
                if self.layout_query == "TND":
                    offset = output_idx_offset.flatten()[prefix : prefix + curr_actualSeq_q].reshape(1, -1, 1)
                else:
                    offset = output_idx_offset.flatten()[b_idx * qs : b_idx * qs + curr_actualSeq_q].reshape(1, -1, 1)
                offset_mask = (y[b_idx:(b_idx + 1), :, :curr_actualSeq_q, :actual_selected_count] != -1)
                y[b_idx:(b_idx + 1), :, :curr_actualSeq_q, :actual_selected_count] += offset * offset_mask
                y_value_np[b_idx:(b_idx + 1), :, :curr_actualSeq_q, :actual_selected_count] = -np.sort(-y_value.numpy())[b_idx:(b_idx + 1), :, :curr_actualSeq_q, :actual_selected_count]
            else:
                pass
            if self.layout_query == "TND":
                prefix += cu_seqlens_q[b_idx]
        return y, y_value, y_value_np

    def trans_shape_to_bnsd(self, tensor, shape, layout, headnums=None, act_seq=None, is_weights=False, tensor_name=None):
        if layout in ["BSND"]:
            B = shape[0]
            S = shape[1]
            N = shape[2]
            D = 1
            if is_weights:
                tensor = torch.unsqueeze(tensor, dim=-1)
            else:
                D = shape[3]
            tensor = tensor.reshape(B, S, N, D).permute(0, 2, 1, 3)
            return tensor, [B, N, S, D]
        elif layout == "BSN":
            print("shape", shape)
            B = shape[0]
            S = shape[1]
            N = shape[2]
            if is_weights:
                D = 1
                tensor = torch.unsqueeze(tensor, dim=-1)  # 补D轴
                tensor = tensor.reshape(B, S, N, D).permute(0, 2, 1, 3)
                return tensor, [B, N, S, D]
            else:
                tensor = tensor.reshape(B, S, N).permute(0, 2, 1)
                return tensor, [B, N, S]
        elif layout in ["TND"]:
            T = shape[0]
            N = shape[1]
            D = 1
            if is_weights:
                tensor = torch.unsqueeze(tensor, dim=-1)
            else:
                D = shape[2]
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
            return new_tensor, [B, N, S, D]
        elif layout == "TN":
            T = shape[0]
            N = shape[1]
            D = 1
            B = len(act_seq)
            S = max(act_seq)
            new_tensor = torch.zeros((B, N, S), dtype=tensor.dtype)
            t_start = 0
            for b_index in range(B):
                act_s = act_seq[b_index]
                t_end = t_start + act_s
                if act_s == 0:
                    continue
                for n_index in range(N):
                    new_tensor[b_index, n_index, 0:act_s] = tensor[t_start:t_end, n_index]
                t_start += act_s
            return new_tensor, [B, N, S]
        else:
            return tensor, shape

    def trans_tnd_actseq(self,list):
        list_len = len(list)
        if list_len == 0:
            raise ValueError(f'TND情况下 cu_seqlens需要必传')
        list_new = []
        list_new.append(list[0])
        for i in range(list_len - 1):
            new_item = list[i + 1] - list[i]
            if new_item >= 0:
                list_new.append(new_item)
            else:
                raise ValueError(f'TND情况下 cu_seqlens 为非递减数列 cu_seqlens={list}')
        return list_new

    def cal_atten_per_batch_fp16(self,b_idx):
        cur_q = self.cur_q
        cur_k = self.cur_k
        cur_w = self.cur_wt.to(dtype=torch.float32)
        sparse_count = self.topk
        sparse_mode = self.mask_mode
        cmp_ratio = self.cmp_ratio
        qk_bmm_res = torch.bmm(
            cur_q.to(dtype = torch.float32).squeeze(0),
            cur_k.to(dtype = torch.float32).permute(0, 1, 3, 2).squeeze(0)
        ).unsqueeze(0)
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

        if sparse_mode == 3:
            cur_m = self.cur_m
            cur_m_broadcasted = cur_m.reshape(1, 1, temp_s1, temp_s2)
            cur_m_broadcasted = torch.broadcast_to(cur_m_broadcasted, (1, temp_n2, temp_s1, temp_s2))
            # 根据布尔矩阵置-inf
            reduce_sum[cur_m_broadcasted.to(dtype = torch.bool)] = -torch.inf
        to_be_sort_ele = reduce_sum.clone()
        to_be_sort_ele = to_be_sort_ele.to(torch.float32)
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

    def cal_atten_per_batch_bf16(self,b_idx):
        cur_q = self.cur_q
        cur_k = self.cur_k
        cur_w = self.cur_wt.to(dtype=torch.float32)
        sparse_count = self.topk
        sparse_mode = self.mask_mode
        cmp_ratio = self.cmp_ratio
        qk_bmm_res = torch.bmm(
            cur_q.to(dtype = torch.float32).squeeze(0),
            cur_k.to(dtype = torch.float32).permute(0, 1, 3, 2).squeeze(0)
        ).unsqueeze(0)
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

        if sparse_mode == 3:
            cur_m = self.cur_m
            cur_m_broadcasted = cur_m.reshape(1, 1, temp_s1, temp_s2)
            cur_m_broadcasted = torch.broadcast_to(cur_m_broadcasted, (1, temp_n2, temp_s1, temp_s2))
            # 根据布尔矩阵置-inf
            reduce_sum[cur_m_broadcasted.to(dtype = torch.bool)] = -torch.inf

        to_be_sort_ele = reduce_sum.clone()
        to_be_sort_ele = to_be_sort_ele.to(torch.float32)
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

    def broadcast_n_axis(self,n1, n2, temp_tensor, input_dtype):
        g = n1 // n2
        temp_shape = temp_tensor.shape
        B = temp_shape[0]
        S = temp_shape[2]
        D = temp_shape[3]
        modify_tensor = torch.zeros([B, n1, S, D], dtype=temp_tensor.dtype)
        for i in range(n1):
            j = i // g
            modify_tensor[:, i:i + 1, :, :] = temp_tensor[:, j:j + 1, :, :]
        return modify_tensor, modify_tensor.shape

    def create_mask(self, m_shape, act_k, S1, residual):
        atten_masks = torch.zeros(tuple(m_shape), dtype=torch.uint8)
        cmp_ratio = self.cmp_ratio
        tmp_pos_orig = act_k * cmp_ratio + residual - S1

        for i in range(S1):
            if(((tmp_pos_orig+i + 1) / cmp_ratio) < 0):
               atten_masks[i,:] = 1
            else:
               atten_masks[i, math.floor((tmp_pos_orig + i + 1) / cmp_ratio):] = 1
        return atten_masks

    def create_mask_right_down(self, m_shape, actualSeqLengthsQ, actualSeqLengthsK, cmpResidualK, batch):
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
                S2 = math.floor(actualSeqLengthsK[i])
            next_tokens = S2 - S1
            next_tokens_list.append(next_tokens)
            act_k = actualSeqLengthsK[i]
            atten_masks = self.create_mask(m_shape, act_k, S1, cmpResidualK[i])
            re_mask_batch.append(np.array(atten_masks, dtype=np.bool_))
        re_mask_np = np.array(re_mask_batch, dtype=np.bool_)
        cpu_mask = torch.from_numpy(re_mask_np)
        return cpu_mask, next_tokens_list


    def forward(self, query, key, weights, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k, block_table, output_idx_offset):
        print("cpu执行中...")

        # 参数的初始化
        batch_size = self.batch_size
        q_seq = self.q_seq
        k_seq = self.k_seq
        layout_query = self.layout_query
        layout_key = self.layout_key
        topk = self.topk
        sparse_mode = self.mask_mode
        out_shape = self.out_shape
        q_shape = self.q_shape
        head_dim = self.head_dim
        q_head_num = self.q_head_num
        k_head_num = self.k_head_num
        q_t_size = self.q_t_size
        k_t_size = self.k_t_size
        block_size = self.block_size
        block_num = self.block_num
        q_dtype = self.qk_dtype
        k_dtype = self.qk_dtype
        w_shape = self.w_shape
        cmp_ratio = self.cmp_ratio
        return_value = self.return_value

        if layout_query == "TND":
            self.cu_seqlens_q = self.trans_tnd_actseq(cu_seqlens_q[1:])
            actualSeqLengths_q = self.cu_seqlens_q
            if seqused_q is not None:
                self.seqused_q = seqused_q
                self.has_seqused_q = True
                actualSeqLengths_q = self.seqused_q
        elif layout_query == "BSND":
            if seqused_q is not None:
                self.seqused_q = seqused_q
                actual_seq_lengths_query = seqused_q
                self.has_seqused_q = True
            else:
                actual_seq_lengths_query = torch.tensor(np.random.uniform(q_seq, q_seq, batch_size).to(torch.int32))
            actualSeqLengths_q = actual_seq_lengths_query

        if layout_key == "TND":
            self.cu_seqlens_k = self.trans_tnd_actseq(cu_seqlens_k[1:])
            actualSeqLengths_k = self.cu_seqlens_k
            k_shape = self.k_shape
            if seqused_k is not None:
                self.seqused_k = seqused_k
                self.has_seqused_k = True
                actualSeqLengths_k = self.seqused_k
        elif layout_key == "BSND":
            if seqused_k is not None:
                self.seqused_k = seqused_k
                actual_seq_lengths_key = seqused_k
                self.has_seqused_k = True
            else:
                actual_seq_lengths_key = torch.tensor(np.random.uniform(k_seq, k_seq, batch_size).to(torch.int32))
            actualSeqLengths_k = actual_seq_lengths_key
            k_shape = self.k_shape

        elif layout_key == "PA_BBND":
            self.actual_seq_lengths_key = seqused_k
            actualSeqLengths_k = self.actual_seq_lengths_key
            k_max_s2 = math.floor(max(actualSeqLengths_k))
            k_shape = [batch_size, k_head_num, k_max_s2, head_dim]
        query = query.cpu()
        key = key.cpu()
        weights = weights.cpu()
        output_idx_offset = output_idx_offset.cpu()
        # 将输入转化为BNSD
        ## BSND / TND -> BNSD
        if self.layout_query == "TND":
            q_bnsd_tensor, q_bnsd_shape = self.trans_shape_to_bnsd(query, q_shape, layout_query,
                                                            q_head_num, self.cu_seqlens_q)
        else:
            q_bnsd_tensor, q_bnsd_shape = self.trans_shape_to_bnsd(query, q_shape, layout_query,
                                                            q_head_num, actualSeqLengths_q)

        ## BSND/TND/ -> BNSD
        if self.layout_key == "TND":
            k_bnsd_tensor, k_bnsd_shape = self.trans_shape_to_bnsd(key, k_shape, layout_key,
                                                            k_head_num, self.cu_seqlens_k)
        else:
            k_bnsd_tensor, k_bnsd_shape = self.trans_shape_to_bnsd(key, k_shape, layout_key,
                                                            k_head_num, actualSeqLengths_k)

        ## BSN1 -> BNS1   TN1 -> BNS1
        is_weights = True
        if self.layout_query == "TND":
            wt_bnsd_tensor, wt_bnsd_shape = self.trans_shape_to_bnsd(weights, w_shape, layout_query,
                                                                q_head_num, self.cu_seqlens_q, is_weights)
        else:
            wt_bnsd_tensor, wt_bnsd_shape = self.trans_shape_to_bnsd(weights, w_shape, layout_query,
                                                                q_head_num, actualSeqLengths_q, is_weights)

        # 将 k n2轴 广播为 n1
        if q_head_num != k_head_num:
            k_bnsd_tensor, k_bnsd_shape = self.broadcast_n_axis(q_head_num, k_head_num, k_bnsd_tensor, k_dtype)
        self.q_bnsd_tensor = q_bnsd_tensor
        self.q_bnsd_shape = q_bnsd_shape
        self.k_bnsd_tensor = k_bnsd_tensor
        self.k_bnsd_shape = k_bnsd_shape
        self.wt_bnsd_tensor = wt_bnsd_tensor
        self.wt_bnsd_shape = wt_bnsd_shape
        # 生成mask, sparse_mode=3时使能
        m_shape_std = [q_bnsd_shape[2], k_bnsd_shape[2]] # m_shape应该是[s1,s2]
        batch = q_bnsd_shape[0]
        m_tensor = []
        if sparse_mode == 3:
            m_tensor, next_tokens_list = self.create_mask_right_down(m_shape_std, actualSeqLengths_q, actualSeqLengths_k, cmp_residual_k, batch)
        elif sparse_mode == 0:
            pass
        else:
            raise ValueError("unsupported sparse_mode!")
        self.m_tensor = m_tensor
        y, y_value, y_value_np = self.cal_atten_bnsd(output_idx_offset)
        sparse_value = torch.from_numpy(y_value_np)

        # TND & PA 需要传入out_shape为BNSD
        out_shape_bnsd = copy.deepcopy(self.q_bnsd_shape)
        out_shape_bnsd[1] = k_head_num
        out_shape_bnsd[-1] = topk
        if self.layout_query == "TND":
            y = self.trans_bnsd_to_layout(y, out_shape_bnsd, layout_query, self.cu_seqlens_q)
            if return_value:
                sparse_value = self.trans_bnsd_to_layout(sparse_value, out_shape_bnsd, layout_query, self.cu_seqlens_q)
        else:
            y = self.trans_bnsd_to_layout(y, out_shape_bnsd, layout_query, q_seq)
            if return_value:
                sparse_value = self.trans_bnsd_to_layout(sparse_value, out_shape_bnsd, layout_query, q_seq)

        return y, y_value, sparse_value

def trans_prefix_actseq(self,list):
        list_len = len(list)
        if list_len == 0:
            raise ValueError(f'PA场景下 act_seq需要必传')
        list_new = []
        list_new.append(list[0])
        for i in range(list_len - 1):
            new_item = list[i + 1] - list[i]
            if new_item >= 0:
                list_new.append(new_item)
            else:
                raise ValueError(f'PA场景下act seq 为非递减数列 act_seq ={list}')
        return list_new

def liv2_output_single(params):
    batch_size, q_seq, k_seq, q_t_size, k_t_size, q_head_num, k_head_num, head_dim, block_size, block_num, \
    qk_dtype, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k, output_idx_offset, \
    layout_query, layout_key, topk, mask_mode, query_datarange, key_datarange, weights_datarange, \
    cmp_ratio, return_value, max_seqlen_q = params

    test_liv2 = GeneralizedLIV2(batch_size, q_seq, k_seq, q_t_size, k_t_size, q_head_num, k_head_num,
                                head_dim, block_size, block_num, qk_dtype, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k,
                                cmp_residual_k, layout_query, layout_key, topk, max_seqlen_q, mask_mode, cmp_ratio, return_value)

    if cu_seqlens_q is not None:
        cu_seqlens_q = torch.tensor(cu_seqlens_q).to(torch.int32).npu()
    if cu_seqlens_k is not None:
        cu_seqlens_k = torch.tensor(cu_seqlens_k).to(torch.int32).npu()
    if seqused_q is not None:
        seqused_q = torch.tensor(seqused_q).to(torch.int32).npu()
    if seqused_k is not None:
        seqused_k = torch.tensor(seqused_k).to(torch.int32).npu()
    if cmp_residual_k is not None:
        cmp_residual_k = torch.tensor(cmp_residual_k).to(torch.int32).npu()

    if layout_query == "BSND":
        query = torch.tensor(np.random.uniform(query_datarange[0], query_datarange[1], (batch_size, q_seq, q_head_num, head_dim))).to(qk_dtype).npu()
        weights = torch.tensor(np.random.uniform(weights_datarange[0], weights_datarange[1], (batch_size, q_seq, q_head_num))).to(torch.float32).npu()
        if output_idx_offset is not None:
            output_idx_offset = torch.tensor(output_idx_offset).reshape(batch_size, q_seq, 1).to(torch.int32).npu()
    elif layout_query == "TND":
        query = torch.tensor(np.random.uniform(query_datarange[0], query_datarange[1], (q_t_size, q_head_num, head_dim))).to(qk_dtype).npu()
        weights = torch.tensor(np.random.uniform(weights_datarange[0], weights_datarange[1], (q_t_size, q_head_num))).to(torch.float32).npu()
        if output_idx_offset is not None:
            output_idx_offset = torch.tensor(output_idx_offset).reshape(q_t_size, 1).to(torch.int32).npu()

    if layout_key == "BSND":
        key = torch.tensor(np.random.uniform(key_datarange[0], key_datarange[1], (batch_size, k_seq, k_head_num, head_dim))).to(qk_dtype).npu()
        block_table = None
        cpu_result, topk_value, cpu_topk_value = test_liv2.forward(query, key, weights, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k, block_table, output_idx_offset)

    elif layout_key == "TND":
        key = torch.tensor(np.random.uniform(key_datarange[0], key_datarange[1], (k_t_size, k_head_num, head_dim))).to(qk_dtype).npu()
        block_table = None
        cpu_result, topk_value, cpu_topk_value = test_liv2.forward(query, key, weights, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k, block_table, output_idx_offset)

    elif layout_key == "PA_BBND":
        # 以不同batch中最大seq为标准初始化key(bnsd)
        k_max_s2 = math.floor(max(seqused_k))
        k_max_block_num_per_batch = math.ceil(k_max_s2 / block_size) #遍历batch得到的最大的block num

        key_bnsd = torch.tensor(np.random.uniform(key_datarange[0], key_datarange[1],(batch_size, k_head_num, k_max_s2, head_dim))).to(qk_dtype)
        key_block_num_per_batch = []
        key_block_num_sum = 0
        for cur_act_k in seqused_k:
            cur_cmp_act_k = math.floor(cur_act_k)
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
        # kv_cache 0轴非连续：将key和key_dequant_scale融合到blockFusion (ref v1 commit keyStride0)
        properties = torch.npu.get_device_properties()
        if "Ascend950" in properties.name:
            key_stride = 10  # 0轴非连续增加stride
            bytes_per_token = head_dim + key_stride # 整个非连续的长度
            blockFusion = torch.zeros((block_num, block_size * k_head_num * bytes_per_token), dtype=qk_dtype)
            key_flat = key.view(block_num, block_size * k_head_num * head_dim)
            blockFusion[:, :block_size * k_head_num * head_dim] = key_flat
            blockFusion = blockFusion.npu()
            key = blockFusion[:, :block_size * k_head_num * head_dim].view(block_num, block_size, k_head_num, head_dim)

        key = key.npu()
        cpu_result, topk_value, cpu_topk_value = test_liv2.forward(query, key_bnsd, weights, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k, block_table, output_idx_offset)
        block_table = torch.from_numpy(block_table).to(dtype=torch.int32).npu()
    
    if layout_query == "TND":
        if seqused_q is not None:
            max_seqlen_q = max(seqused_q).item()
        else:
            seqlen = []
            for b_idx in range(batch_size):
                seqlen.append(cu_seqlens_q[b_idx + 1] - cu_seqlens_q[b_idx])
            max_seqlen_q = max(seqlen).item()
    else:
        if seqused_q is not None:
            max_seqlen_q = max(seqused_q).item()
        else:
            max_seqlen_q = q_seq
    
    if layout_key == "TND":
        if seqused_k is not None:
            max_seqlen_k = max(seqused_k).item()
        else:
            seqlen = []
            for b_idx in range(batch_size):
                seqlen.append(cu_seqlens_q[b_idx + 1] - cu_seqlens_k[b_idx])
            max_seqlen_k = max(seqlen).item()
    elif layout_key == "PA_BBND":
        max_seqlen_k = max(seqused_k).item()
    else:
        if seqused_k is not None:
            max_seqlen_k = max(seqused_k).item()
        else:
            max_seqlen_k = k_seq
    # metadata = lightning_indexer_metadata(
    #                             num_heads_q = q_head_num,
    #                             num_heads_k = k_head_num,
    #                             head_dim = head_dim,
    #                             topk = topk,
    #                             cu_seqlens_q = cu_seqlens_q,
    #                             cu_seqlens_k = cu_seqlens_k,
    #                             seqused_q = seqused_q,
    #                             seqused_k = seqused_k,
    #                             cmp_residual_k = cmp_residual_k,
    #                             batch_size = batch_size,
    #                             max_seqlen_q = max_seqlen_q,
    #                             max_seqlen_k = max_seqlen_k,
    #                             layout_q = layout_query,
    #                             layout_k = layout_key,
    #                             mask_mode = mask_mode,
    #                             cmp_ratio = cmp_ratio)

    # metadata = metadata.npu()
    metadata = None
    npu_result, npu_topk_value = lightning_indexer(query, key, weights,
                                             cu_seqlens_q = cu_seqlens_q,
                                             cu_seqlens_k = cu_seqlens_k,
                                             seqused_q = seqused_q,
                                             seqused_k = seqused_k,
                                             cmp_residual_k = cmp_residual_k,
                                             block_table = block_table,
                                             output_idx_offset = output_idx_offset,
                                             metadata = metadata,
                                             topk = topk,
                                             max_seqlen_q = max_seqlen_q,
                                             layout_q = layout_query,
                                             layout_k = layout_key,
                                             mask_mode = mask_mode,
                                             cmp_ratio = cmp_ratio,
                                             return_value = return_value)
    torch.npu.synchronize()
    npu_topk_value, _ = npu_topk_value.sort(dim=-1, descending=True)
    return cpu_result, npu_result, topk_value, cpu_topk_value, npu_topk_value