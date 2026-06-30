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

# Standalone assets golden: TTK uses the CPU reference below directly and does
# not depend on tests/pytest modules being present in source distributions.
import torch
import random
import numpy as np
import math
import ctypes
import copy

class GeneralizedLI:
    def __init__(self, batch_size, q_seq, k_seq, q_t_size, k_t_size, q_head_num, k_head_num, head_dim, block_size, block_num, qk_dtype, weight_dtype, actual_seq_dtype, act_seq_q, act_seq_k, layout_query, layout_key, sparse_count, sparse_mode):
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
        self.actual_seq_dtype = actual_seq_dtype
        self.act_seq_q = act_seq_q
        self.act_seq_k = act_seq_k
        self.layout_query = layout_query
        self.layout_key = layout_key
        self.sparse_count = sparse_count
        self.sparse_mode = sparse_mode

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
        out_shape_bnsd = list(q_bnsd_tensor.shape)
        out_shape_bnsd[1] = n2
        out_shape_bnsd[-1] = self.sparse_count

        out_shape_bnss = list(q_bnsd_tensor.shape)
        out_shape_bnss[1] = n2
        out_shape_bnss[-1] = math.floor(max(actualSeqLengths_k))

        y = torch.full(out_shape_bnsd,-1,dtype = torch.int32)
        y_value = torch.full(out_shape_bnss,-float('inf'), dtype=torch.float32)

        for b_idx in range(batch_size):
            curr_actualSeq_q = actualSeqLengths_q[b_idx]
            curr_actualSeq_k = math.floor(actualSeqLengths_k[b_idx])
            self.cur_actseq_q = curr_actualSeq_q
            self.cur_actseq_k = curr_actualSeq_k
            self.cur_q = q_bnsd_tensor[b_idx:(b_idx + 1), :, :curr_actualSeq_q, :]
            self.cur_k = k_bnsd_tensor[b_idx:(b_idx + 1), :, :curr_actualSeq_k, :]
            self.cur_wt = wt_bnsd_tensor[b_idx:(b_idx + 1), :, :curr_actualSeq_q, :]
            if self.sparse_mode != 0:
                self.cur_m = mask_tensor[b_idx:(b_idx + 1), :curr_actualSeq_q, :curr_actualSeq_k]
            self.cur_b_idx = b_idx

            if curr_actualSeq_q != 0:
                actual_selected_count = min(curr_actualSeq_k, self.sparse_count)
                y[b_idx:(b_idx + 1), :, :curr_actualSeq_q, :actual_selected_count], y_value[b_idx:(b_idx + 1),
                    :, :curr_actualSeq_q, :curr_actualSeq_k] = self.cal_atten_per_batch_b16(b_idx)
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

    def cal_atten_per_batch_b16(self,b_idx):
        cur_q = self.cur_q.to(dtype=torch.float32)
        cur_k = self.cur_k.to(dtype=torch.float32)
        cur_wt = self.cur_wt.to(dtype=torch.float32)

        sparse_count = self.sparse_count
        sparse_mode = self.sparse_mode
        
        qk_bmm_res = torch.bmm(
            cur_q.squeeze(0),
            cur_k.permute(0, 1, 3, 2).squeeze(0)
        ).unsqueeze(0)
        
        qk_relu_out = (qk_bmm_res.to(dtype=torch.float32)).clamp_min(0.0)

        brc_vmul = torch.bmm(
            cur_wt.permute(0, 2, 3, 1).to(dtype=torch.float32).squeeze(0),
            qk_relu_out.permute(0, 2, 1, 3).to(dtype=torch.float32).squeeze(0)
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
        to_be_sort_ele = to_be_sort_ele.to(torch.bfloat16)
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

    def create_mask(self, m_shape, act_k, S1):
        atten_masks = torch.zeros(tuple(m_shape), dtype=torch.uint8)
        tmp_pos_orig = act_k - S1

        for i in range(S1):
            if((tmp_pos_orig+i+1) < 0):
               atten_masks[i,:] = 1
            else:
               atten_masks[i, math.floor(tmp_pos_orig+i+1):] = 1
        return atten_masks

    def create_mask_right_down(self, m_shape, actualSeqLengthsQ, actualSeqLengthsK, batch):
        mask_s_q = m_shape[0]
        mask_s_kv = m_shape[1]
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
            next_tokens = S2-S1
            next_tokens_list.append(next_tokens)
            act_k = actualSeqLengthsK[i]
            atten_masks = self.create_mask(m_shape, act_k, S1)
            re_mask_batch.append(atten_masks)
        re_mask_np = torch.stack(re_mask_batch, dim=0).numpy().astype(np.bool_)
        cpu_mask = torch.from_numpy(re_mask_np)
        return cpu_mask, next_tokens_list

    def forward(self, query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key, block_table):
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
        actual_seq_dtype = self.actual_seq_dtype

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
            k_max_s2 = math.floor(max(actualSeqLengths_k))
        query = query.cpu()
        key = key.cpu()
        weights = weights.cpu()

        # 将输入转化为BNSD
        ## BSND / TND -> BNSD
        q_bnsd_tensor = self.trans_shape_to_bnsd(query,
            layout_query, q_head_num, actualSeqLengths_q)

        ## BSND/TND/ -> BNSD
        k_bnsd_tensor = self.trans_shape_to_bnsd(key,
            layout_key, k_head_num, torch.floor(actualSeqLengths_k).to(actual_seq_dtype))

        ## BSN1 -> BNS1   TN1 -> BNS1
        wt_bnsd_tensor = self.trans_shape_to_bnsd(weights,
            layout_query, q_head_num, actualSeqLengths_q)

        # 将 k n2轴 广播为 n1
        if q_head_num != k_head_num:
            k_bnsd_tensor = self.broadcast_n_axis(q_head_num, k_head_num, k_bnsd_tensor)
        self.q_bnsd_tensor = q_bnsd_tensor
        self.k_bnsd_tensor = k_bnsd_tensor
        self.wt_bnsd_tensor = wt_bnsd_tensor
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

def li_output_single(params):
    import torch_npu
    batch_size, q_seq, k_seq, q_t_size, k_t_size, q_head_num, k_head_num, head_dim, block_size, block_num, \
    qk_dtype, weight_dtype, actual_seq_dtype, act_seq_q, act_seq_k, layout_query, layout_key, sparse_count, \
    sparse_mode, query_datarange, key_datarange, weights_datarange = params
    import ml_dtypes

    test_li = GeneralizedLI(batch_size, q_seq, k_seq, q_t_size, k_t_size, q_head_num, k_head_num, head_dim, block_size, block_num,
                              qk_dtype, weight_dtype, actual_seq_dtype, act_seq_q, act_seq_k, layout_query, layout_key, sparse_count, sparse_mode)

    actual_seq_lengths_query = torch.tensor(np.random.uniform(q_seq, q_seq, batch_size)).to(actual_seq_dtype).npu() \
                            if act_seq_q is None else torch.tensor(act_seq_q).to(actual_seq_dtype).npu()
    actual_seq_lengths_key = torch.tensor(np.random.uniform(k_seq, k_seq, batch_size)).to(actual_seq_dtype).npu() \
                            if act_seq_k is None else torch.tensor(act_seq_k).to(actual_seq_dtype).npu()

    # shape推导
    if layout_query == "BSND":
        query_shape = [batch_size, q_seq, q_head_num, head_dim]
        weights_shape = [batch_size, q_seq, q_head_num]
    elif layout_query == "TND":
        query_shape = [q_t_size, q_head_num, head_dim]
        weights_shape = [q_t_size, q_head_num]

    if layout_key == "BSND":
        key_shape = [batch_size, k_seq, k_head_num, head_dim]
    elif layout_key == "TND":
        key_shape = [k_t_size, k_head_num, head_dim]
    elif layout_key == "PA_BSND":
        # 以不同batch中最大seq为标准初始化key(bnsd)
        k_max_s2 = math.floor(max(act_seq_k))
        key_shape = [batch_size, k_head_num, k_max_s2, head_dim]

    # 随机数据生成
    query = torch.tensor(np.random.uniform(query_datarange[0], query_datarange[1], query_shape)).to(qk_dtype).npu()
    weights = torch.tensor(np.random.uniform(weights_datarange[0], weights_datarange[1], weights_shape)).to(weight_dtype).npu()
    key = torch.tensor(np.random.uniform(key_datarange[0], key_datarange[1], key_shape)).to(qk_dtype).npu()
    block_table = None
    key_cpu = key

    if layout_key == "PA_BSND":
        k_max_block_num_per_batch = math.ceil(k_max_s2 / block_size) #遍历batch得到的最大的block num
        key_block_num_per_batch = []
        key_block_num_sum = 0
        for cur_act_k in act_seq_k:
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
        key_expand = torch.zeros((batch_size, k_head_num, k_max_block_num_per_batch * block_size, head_dim), dtype = qk_dtype)
        key_expand[:,:,:k_max_s2,:] = key_cpu
        key = torch.zeros((block_num, block_size, k_head_num, head_dim), dtype = qk_dtype)
        for i_batch in range(batch_size):
            for i_block, cur_block_id in enumerate(block_table[i_batch]):
                block_start_pos = i_block * block_size
                if cur_block_id == -1:
                    continue
                else:
                    for i_n in range(k_head_num):
                        key[cur_block_id, :, i_n, :] = key_expand[i_batch, i_n, block_start_pos:block_start_pos+block_size,:]
        key = key.npu()
        block_table = torch.from_numpy(block_table).to(dtype=torch.int32).npu()

    # cpu golden生成
    cpu_result, topk_value = test_li.forward(query, key_cpu, weights, actual_seq_lengths_query, actual_seq_lengths_key, block_table)
    cpu_result, _ = torch.sort(cpu_result)

    npu_result, _ = torch_npu.npu_lightning_indexer(query, key, weights,
                                                    actual_seq_lengths_query = actual_seq_lengths_query,
                                                    actual_seq_lengths_key = actual_seq_lengths_key,
                                                    block_table = block_table,
                                                    layout_query = layout_query,
                                                    layout_key = layout_key,
                                                    sparse_count = sparse_count,
                                                    sparse_mode = sparse_mode)
    torch.npu.synchronize()
    npu_result, _ = torch.sort(npu_result)
    return cpu_result, npu_result, topk_value

# ========================= TTK e2e adapter =========================
# The CPU reference above is bundled in assets so TTK e2e does not depend on tests/pytest.
# The adapter below maps torch_npu API-style tensors to GeneralizedLI and normalizes return_value semantics.
_TTK_V3_TOPK_SCORES = None


def clear_topk_scores():
    global _TTK_V3_TOPK_SCORES
    _TTK_V3_TOPK_SCORES = None


def set_topk_scores(scores):
    global _TTK_V3_TOPK_SCORES
    _TTK_V3_TOPK_SCORES = scores


def get_topk_scores():
    return _TTK_V3_TOPK_SCORES


__golden__ = {
    "e2e": {
        "torch_npu.npu_lightning_indexer": "npu_lightning_indexer"
    }
}


def _ttk_tensor_to_list(value):
    if value is None:
        return []
    if torch.is_tensor(value):
        return [int(x) for x in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return [int(value)]


def _ttk_per_batch_from_prefix(prefix):
    prefix = _ttk_tensor_to_list(prefix)
    if not prefix:
        return []
    out = [prefix[0]]
    for i in range(len(prefix) - 1):
        out.append(prefix[i + 1] - prefix[i])
    return out


def _ttk_scores_to_layout(golden_obj, scores_bnsd, layout_query, act_q_per_batch):
    shape = list(scores_bnsd.shape)
    return golden_obj.trans_bnsd_to_layout(scores_bnsd, shape, layout_query, act_q_per_batch)


def _ttk_restore_pa_bsnd(paged_tensor, block_table, act_k, block_size):
    if paged_tensor is None or block_table is None:
        return paged_tensor
    act_k = [int(x) for x in act_k]
    page_table = block_table.detach().cpu() if torch.is_tensor(block_table) else torch.tensor(block_table)
    src = paged_tensor.detach().cpu()
    batch = len(act_k)
    num_heads = src.shape[2]
    tail_shape = src.shape[3:]
    max_k = max(act_k) if act_k else 0
    restored = torch.zeros((batch, num_heads, max_k) + tail_shape, dtype=src.dtype)
    for b_idx, cur_k in enumerate(act_k):
        if cur_k <= 0:
            continue
        pages = int(math.ceil(cur_k / block_size))
        copied = 0
        for page_idx in range(min(pages, page_table.shape[1])):
            block_id = int(page_table[b_idx, page_idx].item())
            if block_id < 0:
                continue
            take = min(block_size, cur_k - copied)
            restored[b_idx, :, copied:copied + take, ...] = src[
                block_id, :take, :, ...
            ].permute(1, 0, *range(2, src.dim() - 1))
            copied += take
            if copied >= cur_k:
                break
    return restored


def npu_lightning_indexer(query, key, weights, *, actual_seq_lengths_query=None,
                          actual_seq_lengths_key=None, block_table=None,
                          layout_query="BSND", layout_key="BSND", sparse_count=2048,
                          sparse_mode=3, pre_tokens=(1 << 63) - 1,
                          next_tokens=(1 << 63) - 1, return_value=False, **kwargs):
    """Golden reference implementation for LightningIndexer attention with TopK scoring."""
    clear_topk_scores()
    q_shape = list(query.shape)
    k_shape = list(key.shape)
    act_q = _ttk_tensor_to_list(actual_seq_lengths_query)
    act_k = _ttk_tensor_to_list(actual_seq_lengths_key)

    if layout_query == "TND":
        batch_size = len(act_q) if act_q else 1
        q_t_size, q_head_num, head_dim = q_shape
        q_seq = max(_ttk_per_batch_from_prefix(act_q) or [q_t_size])
        act_q_forward = act_q
    else:
        batch_size, q_seq, q_head_num, head_dim = q_shape
        act_q_forward = act_q or [q_seq] * batch_size
        q_t_size = sum(act_q_forward)

    if layout_key == "TND":
        k_t_size, k_head_num, _ = k_shape
        k_seq = max(_ttk_per_batch_from_prefix(act_k) or [k_t_size])
        act_k_forward = act_k
    elif layout_key == "PA_BSND":
        block_num, block_size, k_head_num, _ = k_shape
        k_t_size = sum(act_k) if act_k else block_num * block_size
        k_seq = max(act_k or [block_size])
        act_k_forward = act_k or [k_seq] * batch_size
    else:
        _, k_seq, k_head_num, _ = k_shape
        k_t_size = sum(act_k) if act_k else k_seq * batch_size
        act_k_forward = act_k or [k_seq] * batch_size
        block_size = int(kwargs.get("block_size", 16))
        block_num = int(kwargs.get("block_num", 0))

    block_size = int(kwargs.get("block_size", locals().get("block_size", 16)))
    block_num = int(kwargs.get("block_num", locals().get("block_num", 0)))

    golden_key = key
    if layout_key == "PA_BSND":
        golden_key = _ttk_restore_pa_bsnd(key, block_table, act_k_forward, block_size)

    golden = GeneralizedLI(batch_size, q_seq, k_seq, q_t_size, k_t_size,
                           q_head_num, k_head_num, head_dim, block_size, block_num,
                           query.dtype, weights.dtype, torch.int32,
                           act_q_forward, act_k_forward, layout_query, layout_key,
                           int(sparse_count), int(sparse_mode))
    fwd_act_q = act_q_forward if layout_query == "TND" else torch.tensor(act_q_forward, dtype=torch.int32)
    fwd_act_k = act_k_forward if layout_key == "TND" else torch.tensor(act_k_forward, dtype=torch.int32)
    indices, full_scores_bnsd = golden.forward(query, golden_key, weights,
                                               fwd_act_q, fwd_act_k, block_table)

    act_q_per_batch = _ttk_per_batch_from_prefix(act_q_forward) if layout_query == "TND" else act_q_forward
    scores_layout = _ttk_scores_to_layout(golden, full_scores_bnsd, layout_query, act_q_per_batch)
    set_topk_scores(scores_layout)

    return_value = bool(return_value or kwargs.get("return_values", False))
    if return_value:
        gather_idx = indices.clamp_min(0).to(torch.long)
        values = torch.gather(scores_layout, -1, gather_idx).to(query.dtype)
        values = torch.where(indices < 0, torch.full_like(values, -float('inf')), values)
        return indices, values
    return indices, torch.zeros(0, dtype=scores_layout.dtype)
