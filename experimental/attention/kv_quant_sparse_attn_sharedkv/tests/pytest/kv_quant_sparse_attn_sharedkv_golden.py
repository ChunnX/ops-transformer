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

import os
import torch
import torch_npu
import check_valid_param
import pytest
import random
import numpy as np
import math
import custom_ops as ops

DATA_RANGE_LEFT = -2
DATA_RANGE_RIGHT = 2
FP8_DATA_RANGE_LEFT = -1
FP8_DATA_RANGE_RIGHT = 1

class GeneralizedSFAQuant:
    def __init__(self, layout_q, layout_kv, q_type, ori_kv_type, cmp_kv_type, B, S1, T1, T2, T3, N1, N2, D, K, block_num1, block_num2,
                 block_size1, block_size2, cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_kv, softmax_scale, cmp_ratio, ori_mask_mode, cmp_mask_mode,
                 ori_win_left, ori_win_right, kv_quant_mode, topk_value_mode, tile_size, rope_head_dim, ori_topk_length, cmp_topk_length, template_run_mode):
        self.layout_q = layout_q
        self.layout_kv = layout_kv
        self.q_type = q_type
        self.ori_kv_type = ori_kv_type
        self.cmp_kv_type = cmp_kv_type
        self.B = B
        self.S1 = S1
        self.T1 = T1
        self.T2 = T2
        self.T3 = T3
        self.N1 = N1
        self.N2 = N2
        self.D = D
        self.K = K
        self.block_num1 = block_num1
        self.block_num2 = block_num2
        self.block_size1 = block_size1
        self.block_size2 = block_size2
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_ori_kv = cu_seqlens_ori_kv
        self.cu_seqlens_cmp_kv = cu_seqlens_cmp_kv
        self.seqused_kv = seqused_kv
        self.softmax_scale = softmax_scale
        self.cmp_ratio = cmp_ratio
        self.ori_mask_mode = ori_mask_mode
        self.cmp_mask_mode = cmp_mask_mode
        self.ori_win_left = ori_win_left
        self.ori_win_right = ori_win_right
        self.kv_quant_mode = kv_quant_mode
        self.tile_size = tile_size
        self.rope_head_dim = rope_head_dim
        self.ori_topk_length = ori_topk_length
        self.cmp_topk_length = cmp_topk_length
        self.template_run_mode = template_run_mode

    def calculate_by_bnsd(self, q_bnsd, ori_k_bnsd, cmp_k_bnsd, ori_sparse_indices_bnsd, cmp_sparse_indices_bnsd,
        cu_seqlens_q, cu_seqlens_kv, seqused_kv, sinks, ori_topk_length_bnsd, cmp_topk_length_bnsd):
        attn_out = torch.zeros(q_bnsd.shape, dtype=q_bnsd.dtype)
        B = q_bnsd.shape[0]
        act_q = prefix_sum_to_original(cu_seqlens_q)
        G = int(self.N1 / self.N2)

        for i_B in range(B):
            print(f"i_B = {i_B}/{B}")
            cur_act_q = act_q[i_B]
            if self.layout_kv == "PA_ND":
                cur_ori_act_kv = seqused_kv[i_B]
            else: # self.layout_kv == "TND"
                cur_ori_act_kv = cu_seqlens_kv[i_B + 1] - cu_seqlens_kv[i_B]
            for i_N2 in range(self.N2):
                print(f"    i_N2 = {i_N2}/{self.N2}")
                cur_sinks = sinks[i_N2 * G:(i_N2 + 1) * G]
                cur_sinks_expand = cur_sinks.unsqueeze(1)
                for i_S1 in range(cur_act_q):
                    milestones = [int(cur_act_q * pct / 100) for pct in range(10, 101, 10)]
                    milestones = list(dict.fromkeys(milestones))
                    if i_S1 in milestones:
                        current_pct = (i_S1 / cur_act_q) * 100
                        print(f"      进度：{current_pct:.1f}% | 步数：{i_S1:>{len(str(cur_act_q))}}/{cur_act_q}")
                    if self.ori_mask_mode != 0 and i_S1 < cur_act_q - cur_ori_act_kv:  # 根据 ori_kv 判断行无效
                        attn_out[i_B, i_N2 * G: (i_N2 + 1) * G, i_S1, :] = torch.zeros([G, self.D], dtype=torch.float)
                        continue

                    if self.ori_mask_mode == 4:
                        ori_threshold = cur_ori_act_kv - cur_act_q + i_S1 + 1
                        ori_win_end = ori_threshold + self.ori_win_right
                        if self.ori_win_left == -1:
                            ori_win_start = 0
                        else:
                            ori_win_start = max(ori_threshold - self.ori_win_left - 1, 0)
                    elif self.ori_mask_mode == 3:
                        ori_threshold = cur_ori_act_kv - cur_act_q + i_S1 + 1
                        ori_win_end = ori_threshold + self.ori_win_right
                        ori_win_start = 0
                    elif self.ori_mask_mode == 0:
                        ori_win_start = 0
                        ori_win_end = cur_ori_act_kv

                    cur_ori_k_bnsd = ori_k_bnsd[i_B, i_N2, ori_win_start:ori_win_end, :]
                    k_concat = cur_ori_k_bnsd

                    if self.template_run_mode == "SCFA" and cmp_sparse_indices_bnsd is not None:
                        topk_id = cmp_sparse_indices_bnsd[i_B, i_N2, i_S1, :]

                        empty_flag, k_sparse = self.gather_cmp_kv(cmp_k_bnsd, topk_id, i_B, i_N2, i_S1, cur_ori_act_kv, cur_act_q, self.cmp_mask_mode, cmp_topk_length_bnsd, self.cmp_ratio)
                        if empty_flag != True:
                            k_concat = torch.concat([cur_ori_k_bnsd, k_sparse], dim=0)
                    elif self.template_run_mode == "CFA":
                        empty_flag, k_sparse = self.mask_cmp_kv(cmp_k_bnsd, i_B, i_N2, i_S1, cur_ori_act_kv, cur_act_q)
                        if empty_flag != True:
                            k_concat = torch.concat([cur_ori_k_bnsd, k_sparse], dim=0)
                    elif self.template_run_mode == "ORI_SCFA" and ori_sparse_indices_bnsd is not None:
                        topk_id = ori_sparse_indices_bnsd[i_B, i_N2, i_S1, :]

                        empty_flag, k_sparse = self.gather_cmp_kv(ori_k_bnsd, topk_id, i_B, i_N2, i_S1, cur_ori_act_kv, cur_act_q, self.ori_mask_mode, ori_topk_length_bnsd, 1)
                        if empty_flag != True:
                            k_concat = k_sparse
                    elif self.template_run_mode == "ALL_SCFA" and ori_sparse_indices_bnsd is not None and cmp_sparse_indices_bnsd is not None:
                        ori_topk_id = ori_sparse_indices_bnsd[i_B, i_N2, i_S1, :]
                        ori_empty_flag, ori_k_sparse = self.gather_cmp_kv(ori_k_bnsd, ori_topk_id, i_B, i_N2, i_S1, cur_ori_act_kv, cur_act_q, self.ori_mask_mode, ori_topk_length_bnsd, 1)

                        cmp_topk_id = cmp_sparse_indices_bnsd[i_B, i_N2, i_S1, :]
                        cmp_empty_flag, cmp_k_sparse = self.gather_cmp_kv(cmp_k_bnsd, cmp_topk_id, i_B, i_N2, i_S1, cur_ori_act_kv, cur_act_q, self.cmp_mask_mode, cmp_topk_length_bnsd, self.cmp_ratio)
                        if ori_empty_flag != True and cmp_empty_flag != True:
                            k_concat = torch.concat([ori_k_sparse, cmp_k_sparse], dim=0)

                    q_curr = q_bnsd[i_B, i_N2 * G: (i_N2 + 1) * G, i_S1, :]
                    q_curr_fp32 = q_curr.to(dtype=torch.float32)
                    k_concat_fp32 = k_concat.to(dtype=torch.float32)

                    v_concat_fp32 = k_concat_fp32.clone()

                    mm1_res = torch.matmul(q_curr_fp32, k_concat_fp32.T)
                    scale_res = mm1_res * self.softmax_scale
                    softmax_res, softmax_sum = self.sinks_softmax(scale_res, cur_sinks_expand)
                    softmax_res = softmax_res.to(q_bnsd.dtype).to(torch.float32)
                    mm2_res = torch.matmul(softmax_res, v_concat_fp32)
                    v2_res = torch.div(mm2_res, softmax_sum)
                    attn_out[i_B, i_N2 * G: (i_N2 + 1) * G, i_S1, :] = v2_res
        return attn_out

    def gather_cmp_kv(self, k_tensor, topk_id, i_B, i_N2, i_S1, cur_act_kv, cur_act_q, mask_mode, topk_length_bnsd, cmp_ratio, sparse_block_size = 1):
        s2_sparse = list()
        cur_cmp_act_kv = math.floor(cur_act_kv / cmp_ratio)
        threshold = 0
        if mask_mode == 3:
            threshold = math.floor((cur_act_kv - cur_act_q + i_S1 + 1) / cmp_ratio)
        elif mask_mode == 0:
            threshold = math.floor(cur_act_kv / cmp_ratio)
        if topk_length_bnsd != None:
            valid_count = min(topk_length_bnsd[i_B, 0, 0, 0], math.ceil(threshold / sparse_block_size))
        else:
            valid_count = min(self.K, math.ceil(threshold / sparse_block_size))
        for i_valid in range(valid_count):
            cur_topk_id = topk_id[i_valid]

            if cur_topk_id == -1:
                break
            begin_idx = cur_topk_id * sparse_block_size
            end_idx = begin_idx + sparse_block_size if begin_idx + sparse_block_size <= cur_cmp_act_kv else cur_cmp_act_kv
            if begin_idx >= threshold:
                continue
            if end_idx <= threshold:
                s2_sparse.extend(np.arange(begin_idx, end_idx))
            else:
                s2_sparse.extend(np.arange(begin_idx, threshold))

        empty_flag = False
        if len(s2_sparse) == 0:
            k_sparse = []
            empty_flag = True
        else:
            k_sparse = k_tensor[i_B, i_N2, s2_sparse, :]
        return empty_flag, k_sparse

    def mask_cmp_kv(self, k_tensor, i_B, i_N2, i_S1, cur_act_kv, cur_act_q):
        threshold = 0
        if self.cmp_mask_mode == 3:
            threshold = (cur_act_kv - cur_act_q + i_S1 + 1) // self.cmp_ratio
        elif self.cmp_mask_mode == 0:
            threshold = cur_act_kv // self.cmp_ratio
        empty_flag = True
        k_sparse = None
        if threshold > 0:
            empty_flag = False
            k_sparse = k_tensor[i_B, i_N2, :threshold, :]
        return empty_flag, k_sparse

    def sinks_softmax(self, x, sinks):  # [G, S2] [G, 1]
        x = x.to(dtype=torch.float)
        x_concat = torch.cat([x, sinks], dim=1)
        x_max = x_concat.max(dim=-1, keepdims=True)[0]
        x_sub = x - x_max
        y = torch.exp(x_sub)
        x_sum = y.sum(dim=-1, keepdims=True) + torch.exp(sinks - x_max)
        return y, x_sum

    def trans_shape_to_bnsd(self, tensor, shape, layout, act_seq=None):
        if layout in ["BSND"]:
            B = shape[0]
            S = shape[1]
            N = shape[2]
            D = shape[3]
            tensor = tensor.permute(0, 2, 1, 3)
            return tensor, [B, N, S, D]
        elif layout in ["TND"]:
            T = shape[0]
            N = shape[1]
            D = shape[2]
            B = len(act_seq) - 1  # TND act_q is cumulative
            max_s1 = get_max_adjacent_diff(act_seq)
            act_seq_per_batch = prefix_sum_to_original(act_seq)
            new_tensor = torch.zeros((B, N, max_s1, D), dtype=tensor.dtype)
            t_start = 0
            for b_index in range(B):
                cur_act_seq = act_seq_per_batch[b_index]
                t_end = t_start + cur_act_seq
                if cur_act_seq == 0:
                    continue
                for n_index in range(N):
                    new_tensor[b_index, n_index, 0:cur_act_seq, :] = tensor[t_start:t_end, n_index, :]
                t_start += cur_act_seq
            return new_tensor, [B, N, max_s1, D]
        else:
            return tensor, shape

    def trans_topk_length_shape_to_bnsd(self, tensor, shape, layout, act_seq=None):
        if layout in ["BSND"]:
            B = shape[0]
            # S = shape[1]
            # tensor = tensor.reshape(B, 1, S, 1)
            # return tensor, [B, 1, S, 1]
            tensor = tensor.reshape(B, 1, 1, 1)
            return tensor, [B, 1, 1, 1]
        elif layout in ["TND"]:
            T = shape[0]
            B = len(act_seq) - 1  # TND act_q is cumulative
            tensor = tensor.reshape(B, 1, 1, 1)
            return tensor, [B, 1, 1, 1]
        # elif layout in ["TND"]:
        #     T = shape[0]
        #     B = len(act_seq) - 1  # TND act_q is cumulative
        #     max_s1 = get_max_adjacent_diff(act_seq)
        #     act_seq_per_batch = prefix_sum_to_original(act_seq)
        #     new_tensor = torch.zeros((B, 1, max_s1, 1), dtype=tensor.dtype)
        #     t_start = 0
        #     for b_index in range(B):
        #         cur_act_seq = act_seq_per_batch[b_index]
        #         t_end = t_start + cur_act_seq
        #         if cur_act_seq == 0:
        #             continue
        #         for n_index in range(1):
        #             new_tensor[b_index, 0, 0:cur_act_seq, :] = tensor[t_start:t_end, :]
        #         t_start += cur_act_seq
        #    return new_tensor, [B, 1, max_s1, 1]
        else:
            return tensor, shape

    def trans_bnsd_to_target_layout(self, tensor, layout, act_seq=None):
        if layout in ["BSND"]:
            output = tensor.permute(0, 2, 1, 3).contiguous()
            return output
        elif layout in ["TND"]:
            T = act_seq[-1]
            B = tensor.shape[0]
            N = tensor.shape[1]
            D = tensor.shape[3]
            output = torch.zeros((T, N, D), dtype=torch.float)
            t_start = 0
            act_seq_per_batch = prefix_sum_to_original(act_seq)
            for b_index in range(B):
                cur_act_seq = act_seq_per_batch[b_index]
                t_end = t_start + cur_act_seq
                if cur_act_seq == 0:
                    continue
                for n_index in range(N):
                    output[t_start:t_end, n_index, :] = tensor[b_index, n_index, :cur_act_seq, :]
                t_start += cur_act_seq
            return output
        else:
            return tensor

    def forward(self, q, ori_k_cal, cmp_k_cal, ori_sparse_indices, cmp_sparse_indices, cu_seqlens_q, cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv, seqused_kv, ori_topk_length, cmp_topk_length, sinks):
        print("cpu执行中...")
        print(f"template_run_mode = {self.template_run_mode}")

        q_bnsd, q_bnsd_shape = self.trans_shape_to_bnsd(q, q.shape, self.layout_q, cu_seqlens_q)

        ori_sparse_indices_bnsd = None
        ori_sparse_indices_bnsd_shape = None
        if (self.template_run_mode == "ORI_SCFA" or self.template_run_mode == "ALL_SCFA") and ori_sparse_indices is not None:
            ori_sparse_indices_bnsd, ori_sparse_indices_bnsd_shape = self.trans_shape_to_bnsd(ori_sparse_indices,
                ori_sparse_indices.shape, self.layout_q, cu_seqlens_q)
        ori_k_bnsd, ori_k_bnsd_shape = self.trans_shape_to_bnsd(ori_k_cal, ori_k_cal.shape, self.layout_kv, cu_seqlens_ori_kv)
        if self.template_run_mode != "SWA":
            cmp_k_bnsd, cmp_k_bnsd_shape = self.trans_shape_to_bnsd(cmp_k_cal, cmp_k_cal.shape, self.layout_kv, cu_seqlens_cmp_kv)
        else:
            cmp_k_bnsd, cmp_k_bnsd_shape = cmp_k_cal, None
        cmp_sparse_indices_bnsd = None
        cmp_sparse_indices_bnsd_shape = None
        if (self.template_run_mode == "SCFA" or self.template_run_mode == "ALL_SCFA") and cmp_sparse_indices is not None:
            cmp_sparse_indices_bnsd, cmp_sparse_indices_bnsd_shape = self.trans_shape_to_bnsd(cmp_sparse_indices,
                cmp_sparse_indices.shape, self.layout_q, cu_seqlens_q)

        ori_topk_length_bnsd = None
        ori_topk_length_bnsd_shape = None
        if ori_topk_length != None:
            ori_topk_length_bnsd, ori_topk_length_bnsd_shape = self.trans_topk_length_shape_to_bnsd(ori_topk_length,
                ori_topk_length.shape, self.layout_q, cu_seqlens_q)

        cmp_topk_length_bnsd = None
        cmp_topk_length_bnsd_shape = None
        if cmp_topk_length != None:
            cmp_topk_length_bnsd, cmp_topk_length_bnsd_shape = self.trans_topk_length_shape_to_bnsd(cmp_topk_length,
                cmp_topk_length.shape, self.layout_q, cu_seqlens_q)
        attn_out = self.calculate_by_bnsd(q_bnsd, ori_k_bnsd, cmp_k_bnsd, ori_sparse_indices_bnsd, cmp_sparse_indices_bnsd, cu_seqlens_q,
                                          cu_seqlens_ori_kv, seqused_kv, sinks, ori_topk_length_bnsd, cmp_topk_length_bnsd)

        attn_out = self.trans_bnsd_to_target_layout(attn_out, self.layout_q, cu_seqlens_q)
        return attn_out

def prefix_sum_to_original(cu_seqlens):
    """
    从前缀和张量反向计算出原始的非前缀和张量（替代原列表逻辑）

    Args:
        cu_seqlens (torch.Tensor): 形状为 [B+1] 的一维前缀和张量（元素为数字类型，如int/float）

    Returns:
        torch.Tensor: 原始的非前缀和张量，形状为 [B]（与原列表长度一致）

    Raises:
        TypeError: 输入非tensor/非一维tensor
        ValueError: tensor长度<2（无法计算差值）
    """
    # 1. 基础类型校验：必须是torch.Tensor
    if not isinstance(cu_seqlens, torch.Tensor):
        raise TypeError(f"输入必须是torch.Tensor，当前类型：{type(cu_seqlens)}")

    # 2. 维度校验：必须是一维tensor（原列表对应一维）
    if cu_seqlens.ndim != 1:
        raise TypeError(f"输入必须是一维tensor，当前维度：{cu_seqlens.ndim}，形状：{cu_seqlens.shape}")

    # 3. 长度校验（前缀和tensor至少需2个元素才能反向计算）
    if len(cu_seqlens) < 2:
        raise ValueError(f"前缀和tensor长度需≥2，当前长度：{len(cu_seqlens)}")

    # 4. 核心逻辑：计算相邻元素差值（用tensor向量化运算替代循环，效率更高）
    # 原理：original_val[i] = cu_seqlens[i+1] - cu_seqlens[i]
    # 切片实现：cu_seqlens[1:] 取第2个到最后一个元素，cu_seqlens[:-1] 取第1个到倒数第2个元素
    original_tensor = cu_seqlens[1:] - cu_seqlens[:-1]

    return original_tensor

def get_max_adjacent_diff(cu_seqlens):
    """
    计算前缀和列表中相邻元素（后-前）的最大差值

    Args:
        cu_seqlens (list): 长度为 B+1 的前缀和列表

    Returns:
        float/int: 相邻元素的最大差值；若列表长度<2，返回 None
    """
    # 边界检查：列表长度不足2时无相邻元素
    if len(cu_seqlens) < 2:
        return None

    # 初始化最大差值为第一个相邻对的差值
    max_diff = cu_seqlens[1] - cu_seqlens[0]

    # 遍历所有相邻元素对（从第2对开始）
    for i in range(1, len(cu_seqlens)-1):
        current_diff = cu_seqlens[i+1] - cu_seqlens[i]
        # 更新最大差值
        if current_diff > max_diff:
            max_diff = current_diff

    return max_diff

def gen_sparse_indices_bsnd(cmp_ratio, B, S1, N2, K, seqused_kv, mask_mode, topk_length):
    # 有效索引在叠加了causal后有效tokens中选取，不足sparse_block_count，尾部填充-1
    sparse_data = torch.full((B, S1, N2, K), fill_value=-1, dtype=torch.int32)
    for i_B in range(B):
        cur_act_kv = seqused_kv[i_B]
        for i_N2 in range(N2):
            for i_S1 in range(S1):
                if mask_mode == 3:
                    cur_valid_s2_max = math.floor((cur_act_kv - S1 + i_S1 + 1) / cmp_ratio)
                elif mask_mode == 0:
                    cur_valid_s2_max = math.floor(cur_act_kv / cmp_ratio)
                else:
                    raise ValueError(f"mask_mode only support 1 and 3, which is {mask_mode}")

                valid_blocks_max = max(0, cur_valid_s2_max)
                valid_blocks_max = min(valid_blocks_max, topk_length[i_B]) if topk_length != None else valid_blocks_max
                block_indices = torch.randperm(valid_blocks_max).to(torch.int32)
                valid_blocks_topk = min(valid_blocks_max, K)
                sparse_data[i_B, i_S1, i_N2, :valid_blocks_topk] = block_indices[0:valid_blocks_topk]
    return sparse_data, sparse_data

def gen_sparse_offset_bsnd(cmp_ratio, B, S1, N2, K, D, seqused_kv, mask_mode, topk_length, block_table, max_block_num_per_batch, block_size):
    # 有效索引在叠加了causal后有效tokens中选取，不足sparse_block_count，尾部填充-1
    sparse_data = torch.full((B, S1, N2, K), fill_value=-1, dtype=torch.int32)
    sparse_indices = torch.full((B, S1, N2, K), fill_value=-1, dtype=torch.int32)
    for i_B in range(B):
        cur_act_kv = seqused_kv[i_B]
        for i_N2 in range(N2):
            for i_S1 in range(S1):
                if mask_mode == 3:
                    cur_valid_s2_max = math.floor((cur_act_kv - S1 + i_S1 + 1) / cmp_ratio)
                elif mask_mode == 0:
                    cur_valid_s2_max = math.floor(cur_act_kv / cmp_ratio)
                else:
                    raise ValueError(f"mask_mode only support 1 and 3, which is {mask_mode}")

                valid_blocks_max = max(0, cur_valid_s2_max)
                valid_blocks_max = min(valid_blocks_max, topk_length[i_B]) if topk_length != None else valid_blocks_max
                block_indices = torch.randperm(valid_blocks_max).to(torch.int32)
                block_offset = torch.zeros(valid_blocks_max).to(torch.int32)
                i = 0
                for k in block_indices:
                    blkTableIdx = k // block_size
                    blkTableOffset = k % block_size
                    block_offset[i] = block_table[i_B, blkTableIdx] * block_size + blkTableOffset
                    i = i + 1
                valid_blocks_topk = min(valid_blocks_max, K)
                sparse_indices[i_B, i_S1, i_N2, :valid_blocks_topk] = block_indices[0:valid_blocks_topk]
                sparse_data[i_B, i_S1, i_N2, :valid_blocks_topk] = block_offset[0:valid_blocks_topk]
    return sparse_data, sparse_indices

def gen_sparse_indices_tnd(cmp_ratio, B, T1, N2, K, cu_seqlens_q, seqused_kv, mask_mode, topk_length):
    # 有效索引在叠加了causal后有效tokens中选取，不足sparse_block_count，尾部填充-1
    sparse_data = torch.full((T1, N2, K), fill_value=-1, dtype=torch.int32)
    for i_B in range(B):
        cur_act_q = cu_seqlens_q[i_B + 1] - cu_seqlens_q[i_B]
        s1_prefix = cu_seqlens_q[i_B]
        cur_act_kv = seqused_kv[i_B]
        for i_N2 in range(N2):
            for i_S1 in range(cur_act_q):
                if mask_mode == 3:
                    cur_valid_s2_max = math.floor((cur_act_kv - cur_act_q + i_S1 + 1) / cmp_ratio)
                elif mask_mode == 0:
                    cur_valid_s2_max = math.floor(cur_act_kv / cmp_ratio)
                else:
                    raise ValueError(f"mask_mode only support 1 and 3, which is {mask_mode}")

                valid_blocks_max = max(0, cur_valid_s2_max)
                valid_blocks_max = min(valid_blocks_max, topk_length[i_B]) if topk_length != None else valid_blocks_max
                block_indices = torch.randperm(valid_blocks_max).to(torch.int32)
                valid_blocks_topk = min(valid_blocks_max, K)
                sparse_data[s1_prefix + i_S1, i_N2, :valid_blocks_topk] = block_indices[0:valid_blocks_topk]
    return sparse_data, sparse_data

def gen_sparse_offset_tnd(cmp_ratio, B, T1, N2, K, D, cu_seqlens_q, seqused_kv, mask_mode, topk_length, block_table, max_block_num_per_batch, block_size):
    # 有效索引在叠加了causal后有效tokens中选取，不足sparse_block_count，尾部填充-1
    sparse_data = torch.full((T1, N2, K), fill_value=-1, dtype=torch.int32)
    sparse_indices = torch.full((T1, N2, K), fill_value=-1, dtype=torch.int32)
    for i_B in range(B):
        cur_act_q = cu_seqlens_q[i_B + 1] - cu_seqlens_q[i_B]
        s1_prefix = cu_seqlens_q[i_B]
        cur_act_kv = seqused_kv[i_B]
        for i_N2 in range(N2):
            for i_S1 in range(cur_act_q):
                if mask_mode == 3:
                    cur_valid_s2_max = math.floor((cur_act_kv - cur_act_q + i_S1 + 1) / cmp_ratio)
                elif mask_mode == 0:
                    cur_valid_s2_max = math.floor(cur_act_kv / cmp_ratio)
                else:
                    raise ValueError(f"mask_mode only support 1 and 3, which is {mask_mode}")

                valid_blocks_max = max(0, cur_valid_s2_max)
                valid_blocks_max = min(valid_blocks_max, topk_length[i_B]) if topk_length != None else valid_blocks_max
                block_indices = torch.randperm(valid_blocks_max).to(torch.int32)
                block_offset = torch.zeros(valid_blocks_max).to(torch.int32)
                i = 0
                for k in block_indices:
                    blkTableIdx = k // block_size
                    blkTableOffset = k % block_size
                    block_offset[i] = block_table[i_B, blkTableIdx] * block_size + blkTableOffset
                    i = i + 1
                valid_blocks_topk = min(valid_blocks_max, K)
                sparse_indices[s1_prefix + i_S1, i_N2, :valid_blocks_topk] = block_indices[0:valid_blocks_topk]
                sparse_data[s1_prefix + i_S1, i_N2, :valid_blocks_topk] = block_offset[0:valid_blocks_topk]
    return sparse_data, sparse_indices

def gen_ori_kv_quant_1_tnd(q_type, B, T1, T2, N2, K, rope_head_dim, nope_head_dim, tile_size, quant_scale_head_dim,
        pad_d, cu_seqlens_q, cu_seqlens_ori_kv, ori_mask_mode, ori_topk_length, quant_param_range_left, quant_param_range_right):
    print("generating tnd ori kv.......")
    # 1. 生成并处理 Nope (448) 和 Rope (64) -> Feature (512)
    ori_k_nope_tnd_npu = torch.tensor(np.random.uniform(DATA_RANGE_LEFT, DATA_RANGE_RIGHT,
        (T2, N2, nope_head_dim))).to(torch.float8_e4m3fn)
    ori_k_nope_tnd = ori_k_nope_tnd_npu.to(q_type)
    ori_k_rope_tnd = torch.tensor(np.random.uniform(DATA_RANGE_LEFT, DATA_RANGE_RIGHT,
        (T2, N2, rope_head_dim))).to(q_type)

    # 2. 生成 Scale (7) 和 Padding (1) -> Metadata (8)
    ori_kv_quant_param_tensor_npu = torch.tensor(np.random.uniform(quant_param_range_left, quant_param_range_right,
        (T2, N2, quant_scale_head_dim))).to(torch.float8_e8m0fnu)
    ori_kv_quant_param_tensor = ori_kv_quant_param_tensor_npu.to(q_type)
    ori_pad_tensor = torch.zeros((T2, N2, pad_d)).to(torch.float8_e8m0fnu)

    # 3. nope部分*scale，转成fp8，保存为bin文件，再转回bf16
    for d_loop in range(quant_scale_head_dim):
        for tile_loop in range(tile_size):
            offset = d_loop * tile_size + tile_loop
            ori_k_nope_tnd[:, :, offset:offset+1] = torch.mul(ori_k_nope_tnd[:, :, offset:offset+1], ori_kv_quant_param_tensor[:, :, d_loop:d_loop+1])

    # 4. 核心修改：分组填充物理内存
    ori_k_tnd = torch.concat([ori_k_nope_tnd, ori_k_rope_tnd], dim=-1)
    ori_pad_tensor = torch.tensor(np.random.uniform(0, 0, (T2, N2, pad_d))).to(torch.float8_e8m0fnu)
    ori_k_tnd_npu = torch.concat([ori_k_rope_tnd.view(torch.float8_e4m3fn), ori_k_nope_tnd_npu,
        ori_kv_quant_param_tensor_npu.view(torch.float8_e4m3fn), ori_pad_tensor.view(torch.float8_e4m3fn)], dim=-1)
    return ori_k_tnd.clone(), None, ori_k_tnd_npu, None, None

def gen_ori_kv(q_type, layout_q, ori_kv_type, B, S1, T1, N2, K, rope_head_dim, nope_head_dim, tile_size, quant_scale_head_dim, d_combined,
               pad_d, block_num1, block_size1, ori_max_s2, ori_max_block_num_per_batch, cu_seqlens_q,  seqused_kv, ori_mask_mode,
               template_run_mode, topk_value_mode, ori_topk_length, quant_param_range_left, quant_param_range_right):
    # 1. 生成并处理 Nope (448) 和 Rope (64) -> Feature (512)
    ori_k_nope_bnsd_npu = torch.tensor(np.random.uniform(DATA_RANGE_LEFT, DATA_RANGE_RIGHT,
        (B, N2, ori_max_s2, nope_head_dim))).to(torch.float8_e4m3fn)
    ori_k_nope_bnsd = ori_k_nope_bnsd_npu.to(q_type)
    ori_k_rope_bnsd = torch.tensor(np.random.uniform(DATA_RANGE_LEFT, DATA_RANGE_RIGHT,
        (B, N2, ori_max_s2, rope_head_dim))).to(q_type)

    # 2. 生成 Scale (7) 和 Padding (1) -> Metadata (8)
    ori_kv_quant_param_tensor_npu = torch.tensor(np.random.uniform(quant_param_range_left, quant_param_range_right,
        (B, N2, ori_max_s2, quant_scale_head_dim))).to(torch.float8_e8m0fnu)
    ori_kv_quant_param_tensor = ori_kv_quant_param_tensor_npu.to(q_type)
    ori_pad_tensor = torch.zeros((B, N2, ori_max_s2, pad_d)).to(torch.float8_e8m0fnu)
    # 3. nope部分*scale，转成fp8，保存为bin文件，再转回bf16
    for d_loop in range(quant_scale_head_dim):
        for tile_loop in range(tile_size):
            offset = d_loop * tile_size + tile_loop
            ori_k_nope_bnsd[:, :, :, offset:offset+1] = torch.mul(ori_k_nope_bnsd[:, :, :, offset:offset+1], ori_kv_quant_param_tensor[:, :, :, d_loop:d_loop+1])
    ori_k_bnsd = torch.concat([ori_k_nope_bnsd, ori_k_rope_bnsd], dim=3)

    # 4. 生成blockTable: Block映射逻辑 (保持不变)
    ori_block_num_per_batch = []
    ori_block_num_sum = 0
    for cur_ori_act_kv in seqused_kv:
        cur_ori_kv_block_num = math.ceil(cur_ori_act_kv / block_size1)
        ori_block_num_per_batch.append(cur_ori_kv_block_num)
        ori_block_num_sum += cur_ori_kv_block_num

    ori_block_id_list = np.random.permutation(np.arange(block_num1)).astype(np.int32) #生成随机映射
    ori_block_table = np.full((B, ori_max_block_num_per_batch), fill_value=-1, dtype=np.int32) # 初始化blockTable
    cur_block_id = 0
    for b in range(B):
        num = ori_block_num_per_batch[b]
        ori_block_table[b, :num] = ori_block_id_list[cur_block_id : cur_block_id + num]
        cur_block_id += num

    # 5. ================= 核心修改：分组填充物理内存 =================
    # 物理 Block 形状定义：这里我们依然使用 4D tensor，但在 D 维度内部实现分组
    # 此时 d_combined = 584 (448 + 64*2 + 8)
    ori_k_in_pa_shape = torch.zeros((block_num1, block_size1, N2, d_combined + pad_d), dtype=ori_kv_type)

    for i_B in range(B):
        for i_block, cur_phys_block_id in enumerate(ori_block_table[i_B]):
            if cur_phys_block_id == -1: continue

            # 计算该 Block 在逻辑序列中的起始 Token 位置
            start_s = i_block * block_size1
            end_s = start_s + block_size1

            # 计算实际有效的长度（处理边界）
            actual_end_s = min(end_s, ori_max_s2)
            valid_len = actual_end_s - start_s

            if valid_len <= 0: continue

            # --- 填充 Feature 部分 (0:576) ---
            # 排布：block_size * (nope + rope)
            feat_nope = ori_k_nope_bnsd_npu[i_B, :, start_s:actual_end_s, :] # [N, S, 448]
            # 关键点：将 Rope (BF16) view 为 FP8 格式，长度从 64 变为 128
            feat_rope_raw = ori_k_rope_bnsd[i_B, :, start_s:actual_end_s, :].contiguous()
            feat_rope_fp8 = feat_rope_raw.view(torch.float8_e4m3fn) # [N, S, 128]

            feat_all = torch.concat([feat_rope_fp8, feat_nope], dim=-1) # [N, S, 576]

            # 写入物理内存：前 block_size * 576 字节
            # 为了实现 block_size 连排，需要将 [N, S, 576] 转为 [N, S*576]
            feat_flat = feat_all.view(N2, -1)
            # 计算在物理块中的起始偏移
            ori_k_in_pa_shape.permute(0, 2, 1, 3).view(block_num1, N2, -1)[cur_phys_block_id, :, 0 : valid_len * 576] = feat_flat

            # --- B. 准备 Metadata 数据 [N, S, 8] ---
            meta_scale = ori_kv_quant_param_tensor_npu[i_B, :, start_s:actual_end_s, :].view(torch.float8_e4m3fn)
            meta_pad = ori_pad_tensor[i_B, :, start_s:actual_end_s, :].view(torch.float8_e4m3fn)

            meta_all = torch.concat([meta_scale, meta_pad], dim=-1) # [N, S, 8]
            # print("meta_all: ", meta_all)
            meta_flat = meta_all.view(N2, -1)

            # 写入物理内存：从 block_size * 576 字节处开始
            metadata_start_offset = block_size1 * 576
            ori_k_in_pa_shape.permute(0, 2, 1, 3).view(block_num1, N2, -1)[cur_phys_block_id, :, metadata_start_offset : metadata_start_offset + valid_len * 8] = meta_flat

    # generate ori_sparse_data
    ori_sparse_data = None  # SCFA
    ori_sparse_indices = None
    if (template_run_mode == "ORI_SCFA" or template_run_mode == "ALL_SCFA") and ori_max_s2 != 0:
        if topk_value_mode == '1':
            if layout_q == "BSND":
                ori_sparse_data, ori_sparse_indices = gen_sparse_indices_bsnd(1, B, S1, N2, K, seqused_kv, ori_mask_mode, ori_topk_length)
            elif layout_q == "TND":
                ori_sparse_data, ori_sparse_indices = gen_sparse_indices_tnd(1, B, T1, N2, K, cu_seqlens_q, seqused_kv, ori_mask_mode, ori_topk_length)
        else:
            if layout_q == "BSND":
                ori_sparse_data, ori_sparse_indices = gen_sparse_offset_bsnd(1, B, S1, N2, K, d_combined + pad_d, seqused_kv, ori_mask_mode, ori_topk_length, \
                    ori_block_table, ori_max_block_num_per_batch, block_size1)
            elif layout_q == "TND":
                ori_sparse_data, ori_sparse_indices = gen_sparse_offset_tnd(1, B, T1, N2, K, d_combined + pad_d, cu_seqlens_q, seqused_kv, \
                    ori_mask_mode, ori_topk_length, ori_block_table, ori_max_block_num_per_batch, block_size1)

    # if topk_value_mode == 1:
    #     ori_block_table = torch.tensor(ori_block_table).to(torch.int32)
    # else:
    #     ori_block_table = None
    ori_block_table = torch.tensor(ori_block_table).to(torch.int32)

    ori_v_bnsd = ori_k_bnsd.clone()
    ori_v_in_pa_shape = ori_k_in_pa_shape.clone()

    return ori_k_bnsd, ori_sparse_indices, ori_k_in_pa_shape, ori_block_table, ori_sparse_data


def gen_ori_kv_quant_1_pa(q_type, layout_q, ori_kv_type, B, S1, T1, N2, K, rope_head_dim, nope_head_dim, tile_size, quant_scale_head_dim, d_aligned_128,
               pad_d, block_num1, block_size1, ori_max_s2, ori_max_block_num_per_batch, cu_seqlens_q,  seqused_kv, ori_mask_mode,
               template_run_mode, topk_value_mode, ori_topk_length, quant_param_range_left, quant_param_range_right):
    print("generating pa ori kv.......")
    # ori_kv处理流程
    ori_kv_quant_param_tensor_npu = torch.tensor(np.random.uniform(quant_param_range_left, quant_param_range_right,
        (B, N2, ori_max_s2, quant_scale_head_dim))).to(torch.float8_e8m0fnu)
    ori_kv_quant_param_tensor = ori_kv_quant_param_tensor_npu.to(q_type)
    # 分别生成bf16类型的nope合rope矩阵，为了模拟量化的过程，得到准确的量化参数
    ori_k_nope_bnsd_npu = torch.tensor(np.random.uniform(DATA_RANGE_LEFT, DATA_RANGE_RIGHT,
        (B, N2, ori_max_s2, nope_head_dim))).to(torch.float8_e4m3fn)
    ori_k_nope_bnsd = ori_k_nope_bnsd_npu.to(q_type)
    ori_k_rope_bnsd = torch.tensor(np.random.uniform(DATA_RANGE_LEFT, DATA_RANGE_RIGHT,
        (B, N2, ori_max_s2, rope_head_dim))).to(q_type)
    # nope部分*scale，转成fp8，保存为bin文件，再转回bf16
    for d_loop in range(quant_scale_head_dim):
        for tile_loop in range(tile_size):
            offset = d_loop * tile_size + tile_loop
            ori_k_nope_bnsd[:, :, :, offset:offset+1] = torch.mul(ori_k_nope_bnsd[:, :, :, offset:offset+1], ori_kv_quant_param_tensor[:, :, :, d_loop:d_loop+1])
    # bf16的nope+rope拼接在一起，得到ori_k_bnsd和cmp_k_bnsd
    ori_k_bnsd = torch.concat([ori_k_nope_bnsd, ori_k_rope_bnsd], dim=3)

    ori_pad_tensor = torch.tensor(np.random.uniform(0, 0, (B, N2, ori_max_s2, pad_d))).to(torch.float8_e8m0fnu)
    ori_k_bnsd_npu = torch.concat([ori_k_rope_bnsd.view(torch.float8_e4m3fn), ori_k_nope_bnsd_npu, ori_kv_quant_param_tensor_npu.view(torch.float8_e4m3fn), ori_pad_tensor.view(torch.float8_e4m3fn)], dim=3)

    ori_block_num_per_batch = []
    ori_block_num_sum = 0

    for cur_ori_act_kv in seqused_kv:
        cur_ori_kv_block_num = math.ceil(cur_ori_act_kv / block_size1)
        ori_block_num_per_batch.append(cur_ori_kv_block_num)
        ori_block_num_sum += cur_ori_kv_block_num

    if block_num1 < ori_block_num_sum:
        raise ValueError(f"ori_kv actual_block_num < needed_block_num, which is {block_num1 < ori_block_num_sum}")

    ori_block_id_list = np.arange(block_num1)
    ori_block_id_list = np.random.permutation(ori_block_id_list).astype(np.int32)
    cur_block_id = 0
    ori_block_table = np.full((B, ori_max_block_num_per_batch), fill_value=-1, dtype=np.int32)
    batch_idx = 0
    for cur_block_id_threshold in ori_block_num_per_batch:
        for i_block_id in range(cur_block_id_threshold):
            ori_block_table[batch_idx][i_block_id] = ori_block_id_list[cur_block_id]
            cur_block_id += 1
        batch_idx += 1

    # [B, S, N, D] expand to [B, ori_max_block_num_per_batch * block_size1, N, D]
    ori_k_expand = torch.zeros((B, N2, ori_max_block_num_per_batch * block_size1, d_aligned_128), dtype=ori_kv_type)
    ori_k_expand[:, :, :ori_max_s2, :] = ori_k_bnsd_npu
    ori_k_in_pa_shape = torch.zeros((block_num1, block_size1, N2, d_aligned_128), dtype=ori_kv_type)

    for i_B in range(B):
        for i_block, cur_block_id in enumerate(ori_block_table[i_B]):
            block_start_pos = i_block * block_size1
            if cur_block_id == -1:
                continue
            else:
                for i_N2 in range(N2):
                    ori_k_in_pa_shape[cur_block_id, :, i_N2, :] = \
                        ori_k_expand[i_B, i_N2, block_start_pos:block_start_pos + block_size1, :]

    # generate ori_sparse_data
    ori_sparse_data = None  # SCFA
    ori_sparse_indices = None
    if (template_run_mode == "ORI_SCFA" or template_run_mode == "ALL_SCFA") and ori_max_s2 != 0:
        if topk_value_mode == '1':
            if layout_q == "BSND":
                ori_sparse_data, ori_sparse_indices = gen_sparse_indices_bsnd(1, B, S1, N2, K, seqused_kv, ori_mask_mode, ori_topk_length)
            elif layout_q == "TND":
                ori_sparse_data, ori_sparse_indices = gen_sparse_indices_tnd(1, B, T1, N2, K, cu_seqlens_q, seqused_kv, ori_mask_mode, ori_topk_length)
        else:
            if layout_q == "BSND":
                ori_sparse_data, ori_sparse_indices = gen_sparse_offset_bsnd(1, B, S1, N2, K, d_aligned_128, seqused_kv, ori_mask_mode, ori_topk_length, \
                    ori_block_table, ori_max_block_num_per_batch, block_size1)
            elif layout_q == "TND":
                ori_sparse_data, ori_sparse_indices = gen_sparse_offset_tnd(1, B, T1, N2, K, d_aligned_128, cu_seqlens_q, seqused_kv, \
                    ori_mask_mode, ori_topk_length, ori_block_table, ori_max_block_num_per_batch, block_size1)

    # if topk_value_mode == 1:
    #     ori_block_table = torch.tensor(ori_block_table).to(torch.int32)
    # else:
    #     ori_block_table = None
    ori_block_table = torch.tensor(ori_block_table).to(torch.int32)

    ori_v_bnsd = ori_k_bnsd.clone()
    ori_v_in_pa_shape = ori_k_in_pa_shape.clone()

    return ori_k_bnsd, ori_sparse_indices, ori_k_in_pa_shape, ori_block_table, ori_sparse_data


def gen_cmp_kv_quant_1_pa(q_type, layout_q, cmp_kv_type, B, S1, T1, N2, D, K, rope_head_dim, nope_head_dim, tile_size, quant_scale_head_dim, d_aligned_128,
                pad_d, block_num2, block_size2, cmp_max_s2, cmp_max_block_num_per_batch, cu_seqlens_q, seqused_kv, cmp_ratio, cmp_mask_mode, template_run_mode,
                topk_value_mode, cmp_topk_length, quant_param_range_left, quant_param_range_right):
    print("generating pa cmp kv.......")
    if cmp_max_s2 == 0:
            return None, None, None, None
    # cmp kv处理
    cmp_kv_quant_param_tensor_npu = torch.tensor(np.random.uniform(quant_param_range_left, quant_param_range_right,
        (B, N2, cmp_max_s2, quant_scale_head_dim))).to(torch.float8_e8m0fnu)
    cmp_kv_quant_param_tensor = cmp_kv_quant_param_tensor_npu.to(q_type)
    cmp_k_nope_bnsd_npu = torch.tensor(np.random.uniform(DATA_RANGE_LEFT, DATA_RANGE_RIGHT,
        (B, N2, cmp_max_s2, nope_head_dim))).to(torch.float8_e4m3fn)
    cmp_k_nope_bnsd = cmp_k_nope_bnsd_npu.to(q_type)
    cmp_k_rope_bnsd_npu = torch.tensor(np.random.uniform(DATA_RANGE_LEFT, DATA_RANGE_RIGHT,
        (B, N2, cmp_max_s2, rope_head_dim))).to(q_type)
    cmp_k_rope_bnsd = cmp_k_rope_bnsd_npu.to(q_type)
    for d_loop in range(quant_scale_head_dim):
        for tile_loop in range (tile_size):
            offset = d_loop * tile_size + tile_loop
            cmp_k_nope_bnsd[:, :, :, offset:offset+1] = torch.mul(cmp_k_nope_bnsd[:, :, :, offset:offset+1], cmp_kv_quant_param_tensor[:, :, :, d_loop:d_loop+1])
    cmp_k_bnsd = torch.concat([cmp_k_nope_bnsd, cmp_k_rope_bnsd], dim=3)

    cmp_pad_tensor = torch.tensor(np.random.uniform(0, 0, (B, N2, cmp_max_s2, pad_d))).to(torch.float8_e8m0fnu)
    cmp_k_bnsd_npu = torch.concat([cmp_k_rope_bnsd.view(torch.float8_e4m3fn), cmp_k_nope_bnsd_npu, cmp_kv_quant_param_tensor_npu.view(torch.float8_e4m3fn), cmp_pad_tensor.view(torch.float8_e4m3fn)], dim=3)

    cmp_block_num_per_batch = []
    cmp_block_num_sum = 0

    for cur_ori_act_kv in seqused_kv:
        cur_cmp_act_kv = math.floor(cur_ori_act_kv / cmp_ratio)
        cur_cmp_kv_block_num = math.ceil(cur_cmp_act_kv / block_size2)
        cmp_block_num_per_batch.append(cur_cmp_kv_block_num)
        cmp_block_num_sum += cur_cmp_kv_block_num

    if block_num2 < cmp_block_num_sum:
        raise ValueError(f"cmp_kv actual_block_num < needed_block_num, which is {block_num2 < cmp_block_num_sum}")

    cmp_block_id_list = np.arange(block_num2)
    cmp_block_id_list = np.random.permutation(cmp_block_id_list).astype(np.int32)
    cur_block_id = 0
    cmp_block_table = np.full((B, cmp_max_block_num_per_batch), fill_value=-1, dtype=np.int32)
    batch_idx = 0
    for cur_block_id_threshold in cmp_block_num_per_batch:
        for i_block_id in range(cur_block_id_threshold):
            cmp_block_table[batch_idx][i_block_id] = cmp_block_id_list[cur_block_id]
            cur_block_id += 1
        batch_idx += 1

    # [B, S, N, D] expand to [B, cmp_max_block_num_per_batch * block_size1, N, D]
    cmp_k_expand = torch.zeros((B, N2, cmp_max_block_num_per_batch * block_size2, d_aligned_128), dtype=cmp_kv_type)
    cmp_k_expand[:, :, :cmp_max_s2, :] = cmp_k_bnsd_npu
    cmp_k_in_pa_shape = torch.zeros((block_num2, block_size2, N2, d_aligned_128), dtype=cmp_kv_type)

    for i_B in range(B):
        for i_block, cur_block_id in enumerate(cmp_block_table[i_B]):
            block_start_pos = i_block * block_size2
            if cur_block_id == -1:
                continue
            else:
                for i_N2 in range(N2):
                    cmp_k_in_pa_shape[cur_block_id, :, i_N2, :] = \
                        cmp_k_expand[i_B, i_N2, block_start_pos:block_start_pos + block_size2, :]

    # --- 4. 生成 Sparse Indices (保持不变) ---
    cmp_sparse_data = None
    cmp_sparse_indices = None
    if (template_run_mode == "SCFA" or template_run_mode == "ALL_SCFA") and cmp_max_s2 != 0:
        if topk_value_mode == 1:
            if layout_q == "BSND":
                cmp_sparse_data, cmp_sparse_indices = gen_sparse_indices_bsnd(cmp_ratio, B, S1, N2, cmp_max_s2, seqused_kv, cmp_mask_mode, cmp_topk_length)
            elif layout_q == "TND":
                cmp_sparse_data, cmp_sparse_indices = gen_sparse_indices_tnd(cmp_ratio, B, T1, N2, cmp_max_s2, cu_seqlens_q, seqused_kv, cmp_mask_mode, cmp_topk_length)
        else:
            if layout_q == "BSND":
                cmp_sparse_data, cmp_sparse_indices = gen_sparse_offset_bsnd(cmp_ratio, B, S1, N2, cmp_max_s2, d_aligned_128, seqused_kv, cmp_mask_mode, cmp_topk_length, \
                    cmp_block_table, cmp_max_block_num_per_batch, block_size2)
            elif layout_q == "TND":
                cmp_sparse_data, cmp_sparse_indices = gen_sparse_offset_tnd(cmp_ratio, B, T1, N2, cmp_max_s2, d_aligned_128, cu_seqlens_q, seqused_kv, \
                    cmp_mask_mode, cmp_topk_length, cmp_block_table, cmp_max_block_num_per_batch, block_size2)

    # if topk_value_mode == 1:
    #     cmp_block_table = torch.tensor(cmp_block_table).to(torch.int32)
    # else:
    #     cmp_block_table = None
    cmp_block_table = torch.tensor(cmp_block_table).to(torch.int32)
    cmp_v_in_pa_shape = cmp_k_in_pa_shape.clone()

    return cmp_k_bnsd, cmp_sparse_indices, cmp_k_in_pa_shape, cmp_block_table, cmp_sparse_data

def gen_cmp_kv_quant_1_tnd(q_type, B, T1, T3, N2, K, rope_head_dim, nope_head_dim, tile_size, quant_scale_head_dim,
                pad_d, cmp_max_s2, cu_seqlens_q, cu_seqlens_ori_kv, cmp_ratio, cmp_mask_mode, template_run_mode, topk_length,
                quant_param_range_left, quant_param_range_right):
    print("generating tnd ori kv.......")
    if cmp_max_s2 == 0:
        return None, None, None, None
    cmp_kv_quant_param_tensor_npu = torch.tensor(np.random.uniform(quant_param_range_left, quant_param_range_right,
        (T3, N2, quant_scale_head_dim))).to(torch.float8_e8m0fnu)
    cmp_kv_quant_param_tensor = cmp_kv_quant_param_tensor_npu.to(q_type)
    cmp_k_nope_tnd_npu = torch.tensor(np.random.uniform(DATA_RANGE_LEFT, DATA_RANGE_RIGHT,
        (T3, N2, nope_head_dim))).to(torch.float8_e4m3fn)
    cmp_k_nope_tnd = cmp_k_nope_tnd_npu.to(q_type)
    cmp_k_rope_tnd_npu = torch.tensor(np.random.uniform(DATA_RANGE_LEFT, DATA_RANGE_RIGHT,
        (T3, N2, rope_head_dim))).to(q_type)
    cmp_k_rope_tnd = cmp_k_rope_tnd_npu.to(q_type)
    for d_loop in range(quant_scale_head_dim):
        for tile_loop in range (tile_size):
            offset = d_loop * tile_size + tile_loop
            cmp_k_nope_tnd[:, :, offset:offset+1] = torch.mul(cmp_k_nope_tnd[:, :, offset:offset+1], cmp_kv_quant_param_tensor[:, :, d_loop:d_loop+1])
    cmp_k_tnd = torch.concat([cmp_k_nope_tnd, cmp_k_rope_tnd], dim=-1)

    cmp_pad_tensor = torch.tensor(np.random.uniform(0, 0, (T3, N2, pad_d))).to(torch.float8_e8m0fnu)
    cmp_k_tnd_npu = torch.concat([cmp_k_rope_tnd.view(torch.float8_e4m3fn), cmp_k_nope_tnd_npu, cmp_kv_quant_param_tensor_npu.view(torch.float8_e4m3fn), cmp_pad_tensor.view(torch.float8_e4m3fn)], dim=-1)
    act_kv = prefix_sum_to_original(cu_seqlens_ori_kv)
    cmp_sparse_indices, _ = gen_sparse_indices_tnd(cmp_ratio, B, T1, N2, K, cu_seqlens_q, act_kv, cmp_mask_mode, topk_length)
    return cmp_k_tnd.clone(), cmp_sparse_indices, cmp_k_tnd_npu, None, cmp_sparse_indices 

def gen_cmp_kv(q_type, layout_q, cmp_kv_type, B, S1, T1, N2, D, K, rope_head_dim, nope_head_dim, tile_size, quant_scale_head_dim, d_combined,
                pad_d, block_num2, block_size2, cmp_max_s2, cmp_max_block_num_per_batch, cu_seqlens_q, seqused_kv, cmp_ratio, cmp_mask_mode, template_run_mode,
                topk_value_mode, cmp_topk_length, quant_param_range_left, quant_param_range_right):
    if cmp_max_s2 == 0:
        return None, None, None, None
    # --- 1. 生成原始数据 ---
    # 量化参数 (7字节)
    cmp_kv_quant_param_tensor_npu = torch.tensor(np.random.uniform(quant_param_range_left, quant_param_range_right,
        (B, N2, cmp_max_s2, quant_scale_head_dim))).to(torch.float8_e8m0fnu)
    cmp_kv_quant_param_tensor = cmp_kv_quant_param_tensor_npu.to(q_type)

    # Nope 部分 (448字节, FP8)
    cmp_k_nope_bnsd_npu = torch.tensor(np.random.uniform(DATA_RANGE_LEFT, DATA_RANGE_RIGHT,
        (B, N2, cmp_max_s2, nope_head_dim))).to(torch.float8_e4m3fn)

    # Rope 部分 (64个元素, BF16/FP16)
    cmp_k_rope_bnsd_npu = torch.tensor(np.random.uniform(DATA_RANGE_LEFT, DATA_RANGE_RIGHT,
        (B, N2, cmp_max_s2, rope_head_dim))).to(q_type)

    # 模拟量化计算 (用于生成golden计算数据)
    cmp_k_nope_bnsd = cmp_k_nope_bnsd_npu.to(q_type)
    for d_loop in range(quant_scale_head_dim):
        for tile_loop in range(tile_size):
            offset = d_loop * tile_size + tile_loop
            cmp_k_nope_bnsd[:, :, :, offset:offset+1] = torch.mul(
                cmp_k_nope_bnsd[:, :, :, offset:offset+1],
                cmp_kv_quant_param_tensor[:, :, :, d_loop:d_loop+1]
            )
    # 逻辑上的 K (用于对比)
    cmp_k_bnsd = torch.concat([cmp_k_nope_bnsd, cmp_k_rope_bnsd_npu], dim=3)

    # Padding 部分 (1字节)
    cmp_pad_tensor = torch.zeros((B, N2, cmp_max_s2, pad_d)).to(torch.float8_e8m0fnu)

    # --- 2. 计算 Block 映射 (保持不变) ---
    cmp_block_num_per_batch = []
    cmp_block_num_sum = 0
    for cur_ori_act_kv in seqused_kv:
        cur_cmp_act_kv = math.floor(cur_ori_act_kv / cmp_ratio)
        cur_cmp_kv_block_num = math.ceil(cur_cmp_act_kv / block_size2)
        cmp_block_num_per_batch.append(cur_cmp_kv_block_num)
        cmp_block_num_sum += cur_cmp_kv_block_num

    if block_num2 < cmp_block_num_sum:
        raise ValueError(f"cmp_kv actual_block_num < needed_block_num")

    cmp_block_id_list = np.random.permutation(np.arange(block_num2)).astype(np.int32)
    cmp_block_table = np.full((B, cmp_max_block_num_per_batch), fill_value=-1, dtype=np.int32)
    cur_block_id_idx = 0
    for b in range(B):
        for i in range(cmp_block_num_per_batch[b]):
            cmp_block_table[b][i] = cmp_block_id_list[cur_block_id_idx]
            cur_block_id_idx += 1

    # --- 3. 核心修改：实现 [block_size*576 + block_size*8] 排布 ---
    # 定义物理张量：形状设为 (block_num, N2, 每一路的总字节数)
    # d_combined 此时应为 584 (576 + 8)
    total_bytes_per_head_block = block_size2 * (576 + 8)
    cmp_k_in_pa_shape = torch.zeros((block_num2, N2, total_bytes_per_head_block), dtype=cmp_kv_type)

    for i_B in range(B):
        for i_block, cur_phys_block_id in enumerate(cmp_block_table[i_B]):
            if cur_phys_block_id == -1: continue

            start_s = i_block * block_size2
            end_s = start_s + block_size2
            actual_end_s = min(end_s, cmp_max_s2)
            valid_len = actual_end_s - start_s
            if valid_len <= 0: continue

            # --- A. 准备 Feature 数据 (nope + rope) ---
            # nope: [N, S, 448]
            f_nope = cmp_k_nope_bnsd_npu[i_B, :, start_s:actual_end_s, :]
            # rope: [N, S, 64] BF16 -> view 为 [N, S, 128] FP8
            f_rope = cmp_k_rope_bnsd_npu[i_B, :, start_s:actual_end_s, :].contiguous().view(torch.float8_e4m3fn)

            # 拼接成 [N, S, 576]
            feat_all = torch.concat([f_rope, f_nope], dim=-1)

            # --- B. 准备 Metadata 数据 (scale + pad) ---
            m_scale = cmp_kv_quant_param_tensor_npu[i_B, :, start_s:actual_end_s, :].view(torch.float8_e4m3fn)
            m_pad = cmp_pad_tensor[i_B, :, start_s:actual_end_s, :].view(torch.float8_e4m3fn)

            # 拼接成 [N, S, 8]
            meta_all = torch.concat([m_scale, m_pad], dim=-1)

            # --- C. 写入物理内存 ---
            # 按照要求的 Planar 布局：Feature 块在前，Metadata 块在后
            # 对于每个 Head N2：
            for head_idx in range(N2):
                # 写入 Feature: 前 block_size * 576 字节
                # 将该 head 下有效 token 的 576 字节拉平写入
                cmp_k_in_pa_shape[cur_phys_block_id, head_idx, 0 : valid_len * 576] = \
                    feat_all[head_idx].reshape(-1)

                # 写入 Metadata: 起始偏移量为 block_size * 576
                meta_offset = block_size2 * 576
                cmp_k_in_pa_shape[cur_phys_block_id, head_idx, meta_offset : meta_offset + valid_len * 8] = \
                    meta_all[head_idx].reshape(-1)

    cmp_k_in_pa_shape = cmp_k_in_pa_shape.reshape(block_num2, block_size2, N2, d_combined + pad_d)

    # --- 4. 生成 Sparse Indices (保持不变) ---
    cmp_sparse_data = None
    cmp_sparse_indices = None
    if (template_run_mode == "SCFA" or template_run_mode == "ALL_SCFA") and cmp_max_s2 != 0:
        if topk_value_mode == 1:
            if layout_q == "BSND":
                cmp_sparse_data, cmp_sparse_indices = gen_sparse_indices_bsnd(cmp_ratio, B, S1, N2, cmp_max_s2, seqused_kv, cmp_mask_mode, cmp_topk_length)
            elif layout_q == "TND":
                cmp_sparse_data, cmp_sparse_indices = gen_sparse_indices_tnd(cmp_ratio, B, T1, N2, cmp_max_s2, cu_seqlens_q, seqused_kv, cmp_mask_mode, cmp_topk_length)
        else:
            if layout_q == "BSND":
                cmp_sparse_data, cmp_sparse_indices = gen_sparse_offset_bsnd(cmp_ratio, B, S1, N2, cmp_max_s2, d_combined + pad_d, seqused_kv, cmp_mask_mode, cmp_topk_length, \
                    cmp_block_table, cmp_max_block_num_per_batch, block_size2)
            elif layout_q == "TND":
                cmp_sparse_data, cmp_sparse_indices = gen_sparse_offset_tnd(cmp_ratio, B, T1, N2, cmp_max_s2, d_combined + pad_d, cu_seqlens_q, seqused_kv, \
                    cmp_mask_mode, cmp_topk_length, cmp_block_table, cmp_max_block_num_per_batch, block_size2)

    # if topk_value_mode == 1:
    #     cmp_block_table = torch.tensor(cmp_block_table).to(torch.int32)
    # else:
    #     cmp_block_table = None
    cmp_block_table = torch.tensor(cmp_block_table).to(torch.int32)
    cmp_v_in_pa_shape = cmp_k_in_pa_shape.clone()

    return cmp_k_bnsd, cmp_sparse_indices, cmp_k_in_pa_shape, cmp_block_table, cmp_sparse_data


def save_test_case(input_data, output_dir):
    """
    保存单条测试用例到文件
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    case_name = input_data['Testcase_Name']

    # 生成文件名
    input_filename = f"sas_case_{case_name}.pt"
    input_filepath = os.path.join(output_dir, input_filename)

    # 保存数据
    torch.save(input_data, input_filepath)
    print(f"测试用例已保存到: {input_filepath}")

    return input_filepath

def generate_and_save_testdata(params, save_pt=False, save_path=""):
    '''
    生成input param及cpuout
    runNpu: 生成完毕后执行npu计算
    return test_data
    '''

    Testcase_Name, layout_q, layout_kv, q_type, ori_kv_type, cmp_kv_type, B, S1, T1, T2, T3, N1, N2, D, K, block_num1, block_num2, \
    block_size1, block_size2, cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_kv, softmax_scale, cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_win_left, \
    ori_win_right, kv_quant_mode, tile_size, rope_head_dim, ori_topk_length, cmp_topk_length, topk_value_mode, template_run_mode = params

    cu_seqlens_q = torch.tensor(cu_seqlens_q).to(torch.int32)
    cu_seqlens_ori_kv = torch.tensor(cu_seqlens_ori_kv).to(torch.int32) if cu_seqlens_ori_kv is not None else None
    cu_seqlens_cmp_kv = torch.tensor(cu_seqlens_cmp_kv).to(torch.int32) if cu_seqlens_cmp_kv is not None else None
    seqused_kv = torch.tensor(seqused_kv).to(torch.int32) if seqused_kv is not None else None

    # generate q
    if layout_q == "BSND":
        q = torch.tensor(np.random.uniform(DATA_RANGE_LEFT, DATA_RANGE_RIGHT, (B, S1, N1, D))).to(q_type)
        act_q = B * [S1]
    elif layout_q == "TND":
        q = torch.tensor(np.random.uniform(DATA_RANGE_LEFT, DATA_RANGE_RIGHT, (T1, N1, D))).to(q_type)
        if len(cu_seqlens_q) != (B + 1):
            raise ValueError(f"len(cu_seqlens_q) != B + 1, which is {len(cu_seqlens_q)} != {B + 1}")
        else:
            act_q = prefix_sum_to_original(cu_seqlens_q)
            max_s1 = get_max_adjacent_diff(cu_seqlens_q)
    else:
        raise ValueError(f"layout_q is not support {layout_q}")

    # generate ori_kv/cmp_kv (only support PA_ND when Q layout is BSND and support TND and PA_ND when Q layout is TND)
    if layout_kv == "PA_ND" or layout_kv == "TND":
        pass
    else:
        raise ValueError(f"layout_kv is not support {layout_kv}")
    ori_max_s2 = 0
    ori_max_block_num_per_batch = 0
    cmp_max_s2 = 0
    cmp_max_block_num_per_batch = 0
    if layout_kv == "PA_ND":
        if len(seqused_kv) != B:
            raise ValueError(f"len(seqused_kv) != B, which is {len(seqused_kv)} != {B}")
        else:
            ori_max_s2 = max(seqused_kv)
            ori_max_block_num_per_batch = math.ceil(ori_max_s2 / block_size1)

            cmp_max_s2 = math.floor(ori_max_s2 / cmp_ratio)
            cmp_max_block_num_per_batch = math.ceil(cmp_max_s2 / block_size2)
    else: # layout_kv == "TND":
        ori_max_s2 = get_max_adjacent_diff(cu_seqlens_ori_kv).item()
        if template_run_mode == "CFA" or template_run_mode == "SCFA":
            cmp_max_s2 = get_max_adjacent_diff(cu_seqlens_cmp_kv).item()

    if kv_quant_mode != 1 and kv_quant_mode != 2:
        raise ValueError(f"input kv_quant_mode = {kv_quant_mode}, only support (1, 2)")

    # 计算kv每个区域D轴长度
    nope_head_dim = D - rope_head_dim
    quant_scale_head_dim = (nope_head_dim + tile_size - 1) // tile_size
    d_combined = nope_head_dim + rope_head_dim * 2 + quant_scale_head_dim
    d_aligned_128 = (nope_head_dim + rope_head_dim * 2 + quant_scale_head_dim + 127) // 128 * 128
    print(f"d_combined={d_combined}, nope_head_dim={nope_head_dim}, rope_head_dim={rope_head_dim}, quant_scale_head_dim={quant_scale_head_dim}")
    pad_d = 1 if kv_quant_mode == 2 else d_aligned_128 -nope_head_dim - rope_head_dim *2 -quant_scale_head_dim
    # 根据输入的data range，计算scale范围，生成scale tensor，取倒数保存为bin
    quant_param_range_left = DATA_RANGE_LEFT / FP8_DATA_RANGE_LEFT
    quant_param_range_right = DATA_RANGE_RIGHT / FP8_DATA_RANGE_RIGHT

    # generate sinks tensor
    sinks = torch.tensor(np.random.uniform(DATA_RANGE_LEFT/10, DATA_RANGE_RIGHT/10, (N1))).to(torch.float)

    block_num = block_num1 if block_num1 >= block_num2 else block_num2
    # generate ori_kv tensor
    if kv_quant_mode == 1:
        if layout_kv == "PA_ND":
            ori_k_cal, ori_sparse_indices, ori_k_in_pa_shape, ori_block_table, ori_sparse_data = gen_ori_kv_quant_1_pa(q_type, layout_q, ori_kv_type, B, S1, T1, N2, K, rope_head_dim, nope_head_dim, tile_size, quant_scale_head_dim, d_aligned_128,
                                                pad_d, block_num, block_size1, ori_max_s2, ori_max_block_num_per_batch, cu_seqlens_q,
                                                seqused_kv, ori_mask_mode, template_run_mode, topk_value_mode, ori_topk_length, quant_param_range_left, quant_param_range_right)
        else:
            ori_k_cal, ori_sparse_indices, ori_k_in_pa_shape, ori_block_table, ori_sparse_data = gen_ori_kv_quant_1_tnd(q_type, B, T1, T2, N2, K, rope_head_dim, nope_head_dim, tile_size, quant_scale_head_dim,
 	                pad_d, cu_seqlens_q, cu_seqlens_ori_kv, ori_mask_mode, ori_topk_length, quant_param_range_left, quant_param_range_right)
    else:
        ori_k_cal, ori_sparse_indices, ori_k_in_pa_shape, ori_block_table, ori_sparse_data = gen_ori_kv(q_type, layout_q, ori_kv_type, B, S1, T1, N2, K, rope_head_dim, nope_head_dim, tile_size, quant_scale_head_dim, d_combined,
                                                pad_d, block_num, block_size1, ori_max_s2, ori_max_block_num_per_batch, cu_seqlens_q,
                                                seqused_kv, ori_mask_mode, template_run_mode, topk_value_mode, ori_topk_length, quant_param_range_left, quant_param_range_right)
    # generate cmp_kv and sparse_indices
    if template_run_mode == "CFA" or template_run_mode == "SCFA" or template_run_mode == "ALL_SCFA":
        if kv_quant_mode == 1:
            if layout_kv == "PA_ND":
                cmp_k_cal, cmp_sparse_indices, cmp_k_in_pa_shape, cmp_block_table, cmp_sparse_data = gen_cmp_kv_quant_1_pa(q_type, layout_q, cmp_kv_type, B, S1, T1, N2, D, K, rope_head_dim, nope_head_dim, tile_size, quant_scale_head_dim, d_aligned_128,
                                                    pad_d, block_num, block_size2, cmp_max_s2, cmp_max_block_num_per_batch, cu_seqlens_q,
                                                    seqused_kv, cmp_ratio, cmp_mask_mode, template_run_mode, topk_value_mode, cmp_topk_length,
                                                    quant_param_range_left, quant_param_range_right)
            
            else:
                cmp_k_cal, cmp_sparse_indices, cmp_k_in_pa_shape, cmp_block_table, cmp_sparse_data = gen_cmp_kv_quant_1_tnd(q_type, B, T1, T3, N2, K, rope_head_dim, nope_head_dim, tile_size, quant_scale_head_dim,
                                                    pad_d, cmp_max_s2, cu_seqlens_q, cu_seqlens_ori_kv, cmp_ratio, cmp_mask_mode, template_run_mode, cmp_topk_length,
                                                    quant_param_range_left, quant_param_range_right)
        else:
            cmp_k_cal, cmp_sparse_indices, cmp_k_in_pa_shape, cmp_block_table, cmp_sparse_data = gen_cmp_kv(q_type, layout_q, cmp_kv_type, B, S1, T1, N2, D, K, rope_head_dim, nope_head_dim, tile_size, quant_scale_head_dim, d_combined,
                                                    pad_d, block_num, block_size2, cmp_max_s2, cmp_max_block_num_per_batch, cu_seqlens_q,
                                                    seqused_kv, cmp_ratio, cmp_mask_mode, template_run_mode, topk_value_mode, cmp_topk_length,
                                                    quant_param_range_left, quant_param_range_right)
    else:
        cmp_k_in_pa_shape = None
        cmp_sparse_data = None
        cmp_sparse_indices = None
        cmp_block_table = None
        cmp_k_cal = None

    if layout_kv == "PA_ND" and (template_run_mode == "CFA" or template_run_mode == "SCFA" or template_run_mode == "ALL_SCFA"):
        fusionBlock = torch.zeros((block_num, block_size1 + block_size2, N2, d_combined + pad_d), dtype=ori_kv_type)
        fusionBlock.view(block_num, -1)[:, : block_size1 * N2 * (d_combined + pad_d)] = ori_k_in_pa_shape.view(block_num, -1)
        fusionBlock.view(block_num, -1)[:, block_size1 * N2 * (d_combined + pad_d) :] = cmp_k_in_pa_shape.view(block_num, -1)
        fusionBlock.npu()
        ori_k_in_pa_shape = fusionBlock.view(block_num, -1)[:, : block_size1 * N2 * (d_combined + pad_d)].view(block_num, block_size1, N2, d_combined + pad_d)
        cmp_k_in_pa_shape = fusionBlock.view(block_num, -1)[:, block_size1 * N2 * (d_combined + pad_d) :].view(block_num, block_size2, N2, d_combined + pad_d)

    test_sas = GeneralizedSFAQuant(layout_q, layout_kv, q_type, ori_kv_type, cmp_kv_type, B, S1, T1, T2, T3, N1, N2, D, K,
                              block_num1, block_num2, block_size1, block_size2, cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_kv, softmax_scale, cmp_ratio,
                              ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right, kv_quant_mode, topk_value_mode, tile_size, rope_head_dim,
                              ori_topk_length, cmp_topk_length, template_run_mode)
    cpu_result = test_sas.forward(q, ori_k_cal, cmp_k_cal, ori_sparse_indices, cmp_sparse_indices, cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_kv,
        ori_topk_length, cmp_topk_length, sinks)

    print("mode:%s\n",template_run_mode)

    max_seqlen_q = S1
    if layout_q == "TND":
        max_seqlen_q = cu_seqlens_q.max().item()
    else:
        cu_seqlens_q = None
        max_seqlen_q = S1
    max_seqlen_kv = seqused_kv.max().item() if layout_kv == "PA_ND" else get_max_adjacent_diff(cu_seqlens_ori_kv).item()

    # 保存pt
    input_data = {
        # 命名用变量
        'Testcase_Name':Testcase_Name,

        # 固定参数
        'params': params,

        # 输入张量
        'metadata_input': {
            'num_heads_q': N1,
            'num_heads_kv': N2,
            'head_dim': D,
            'cu_seqlens_q': cu_seqlens_q,
            'cu_seqlens_ori_kv':cu_seqlens_ori_kv,
            'cu_seqlens_cmp_kv':cu_seqlens_cmp_kv,
            'seqused_kv': seqused_kv,
            'batch_size': B,
            'max_seqlen_q': max_seqlen_q,
            'max_seqlen_kv': max_seqlen_kv,
            'topk': K if template_run_mode == "SCFA" or template_run_mode == "ALL_SCFA" else 0, # SCFA和ALL_SCFA需要
            'cmp_ratio': cmp_ratio if template_run_mode != "SWA" else 1,  # 仅SWA 不需要
            'ori_mask_mode': ori_mask_mode,
            'cmp_mask_mode': cmp_mask_mode,
            'ori_win_left': ori_win_left,
            'ori_win_right': ori_win_right,
            'layout_q': layout_q,
            'layout_kv': layout_kv,
            'has_ori_kv': True,
            'has_cmp_kv': False if template_run_mode == "SWA" or template_run_mode == "ORI_SCFA" else True  # SWA和ORI_SCFA 为false
        },

        'input': {
            'q': q,
            'ori_kv': ori_k_in_pa_shape,
            'cmp_kv': cmp_k_in_pa_shape,
            'ori_sparse_data': ori_sparse_data,
            'cmp_sparse_data': cmp_sparse_data,
            'ori_block_table': ori_block_table,
            'cmp_block_table': cmp_block_table,
            'cu_seqlens_q': cu_seqlens_q,
            'cu_seqlens_ori_kv': cu_seqlens_ori_kv,
            'cu_seqlens_cmp_kv': cu_seqlens_cmp_kv,
            'seqused_kv': seqused_kv,
            'ori_topk_length': ori_topk_length,
            'cmp_topk_length': cmp_topk_length,
            'topk_value_mode': topk_value_mode,
            'sinks': sinks,
            'kv_quant_mode': kv_quant_mode,
            'tile_size': 64,
            'rope_head_dim': 64,
            'softmax_scale': softmax_scale,
            'cmp_ratio': cmp_ratio if template_run_mode != "SWA" else 1,
            'ori_mask_mode': ori_mask_mode,
            'cmp_mask_mode': cmp_mask_mode,
            'ori_win_left': ori_win_left,
            'ori_win_right': ori_win_right,
            'layout_q': layout_q,
            'layout_kv': layout_kv
        },

        'cpu_output': cpu_result
    }

    if save_pt:
        save_test_case(input_data, save_path)

    return input_data
