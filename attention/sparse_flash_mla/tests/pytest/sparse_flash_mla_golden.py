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
import numpy as np
import os
import pytest
import random
import sys
import torch

DATA_RANGE_LEFT = -10
DATA_RANGE_RIGHT = 10

# 三种运行模式
# 0 不切S2
# 1 切S2
# 2 AMLA

RUN_MODE = 1

# np.random.seed(42)
# torch.manual_seed(42)

AIC_CORE_MAX_NUM = 36
AIV_CORE_MAX_NUM = 72
FA_METADATA_SIZE = 9
FD_METADATA_SIZE = 8
S2_BASE_SIZE = 512

FA_CORE_ENABLE_INDEX = 0
FA_BN2_START_INDEX = 1
FA_M_START_INDEX = 2
FA_S2_START_INDEX = 3
FA_BN2_END_INDEX = 4
FA_M_END_INDEX = 5
FA_S2_END_INDEX = 6
FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 7

FD_CORE_ENABLE_INDEX = 0
FD_BN2_IDX_INDEX = 1
FD_M_IDX_INDEX = 2
FD_WORKSPACE_IDX_INDEX = 3
FD_WORKSPACE_NUM_INDEX = 4
FD_M_START_INDEX = 5
FD_M_NUM_INDEX = 6


def _as_int(value):
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _metadata_to_cpu(metadata):
    if metadata is None:
        return None
    if isinstance(metadata, torch.Tensor):
        return metadata.detach().cpu().to(torch.int32)
    return torch.tensor(metadata, dtype=torch.int32)


def parse_smla_metadata(metadata):
    metadata = _metadata_to_cpu(metadata)
    if metadata is None:
        return {"fa": [], "fd": []}
    flat = metadata.reshape(-1)
    fa_records = []
    for core_idx in range(AIC_CORE_MAX_NUM):
        base = core_idx * FA_METADATA_SIZE
        row = flat[base:base + FA_METADATA_SIZE].tolist()
        # import pdb; pdb.set_trace()
        print(row)
        if row[FA_CORE_ENABLE_INDEX] == 0:
            continue
        fa_records.append({
            "core_idx": core_idx,
            "bn2_start": int(row[FA_BN2_START_INDEX]),
            "m_start": int(row[FA_M_START_INDEX]),
            "s2_start_block": int(row[FA_S2_START_INDEX]),
            "bn2_end": int(row[FA_BN2_END_INDEX]),
            "m_end": int(row[FA_M_END_INDEX]),
            "s2_end_block": int(row[FA_S2_END_INDEX]),
            "first_workspace_idx": int(row[FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX]),
        })
    print(fa_records)
    # import pdb; pdb.set_trace()
    fd_records = []
    fd_base = AIC_CORE_MAX_NUM * FA_METADATA_SIZE
    for core_idx in range(AIV_CORE_MAX_NUM):
        base = fd_base + core_idx * FD_METADATA_SIZE
        row = flat[base:base + FD_METADATA_SIZE].tolist()
        # if row[FD_CORE_ENABLE_INDEX] == 0:
        #     continue
        fd_records.append({
            "core_idx": core_idx,
            "bn2_idx": int(row[FD_BN2_IDX_INDEX]),
            "m_idx": int(row[FD_M_IDX_INDEX]),
            "workspace_idx": int(row[FD_WORKSPACE_IDX_INDEX]),
            "workspace_num": int(row[FD_WORKSPACE_NUM_INDEX]),
            "m_start_idx": int(row[FD_M_START_INDEX]),
            "m_num": int(row[FD_M_NUM_INDEX]),
        })
    return {"fa": fa_records, "fd": fd_records}


def _register_torch_extension():
    torch_ext_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   '../../../../torch_extension'))
    if torch_ext_path not in sys.path:
        sys.path.insert(0, torch_ext_path)
    from cann_ops_transformer.ops import sparse_flash_mla_metadata as _metadata_registration  # noqa: F401


USE_MOCK_METADATA = True


def _compute_row_s2_length(b_idx, n2_idx, s1_idx, B, N2, act_q, ori_act_kv, cmp_act_kv,
                           ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right, cmp_ratio):
    if (ori_mask_mode == 3 or ori_mask_mode == 4) and s1_idx < act_q - ori_act_kv:
        return (0, 0)
    if ori_mask_mode == 4:
        ori_threshold = ori_act_kv - act_q + s1_idx + 1
        ori_win_end = ori_threshold + ori_win_right
        ori_win_start = 0 if ori_win_left == -1 else max(ori_threshold - ori_win_left - 1, 0)
    elif ori_mask_mode == 3:
        ori_threshold = ori_act_kv - act_q + s1_idx + 1
        ori_win_end = ori_threshold + ori_win_right
        ori_win_start = 0
    elif ori_mask_mode == 0:
        ori_win_start = 0
        ori_win_end = ori_act_kv
    else:
        ori_win_start = 0
        ori_win_end = 0
    ori_len = max(0, ori_win_end - ori_win_start)
    cmp_len = 0
    if cmp_mask_mode == 3:
        threshold = math.floor((ori_act_kv - act_q + s1_idx + 1) / cmp_ratio)
    elif cmp_mask_mode == 0:
        threshold = math.floor(ori_act_kv / cmp_ratio)
    else:
        threshold = 0
    cmp_len = max(0, min(threshold, cmp_act_kv))
    return (ori_len, cmp_len)


def _build_mock_metadata(metadata_input, tensor_input):
    B = metadata_input['B']
    N1 = metadata_input['N1']
    N2 = metadata_input['N2']
    G = N1 // N2
    ori_mask_mode = tensor_input['ori_mask_mode']
    cmp_mask_mode = tensor_input['cmp_mask_mode'] if tensor_input['cmp_mask_mode'] is not None else 3
    ori_win_left = tensor_input['ori_win_left']
    ori_win_right = tensor_input['ori_win_right']
    cmp_ratio = metadata_input['cmp_ratio'] if metadata_input['cmp_ratio'] is not None else 1

    seqused_q = tensor_input.get('seqused_q')
    seqused_ori_kv = tensor_input.get('seqused_ori_kv')
    seqused_cmp_kv = tensor_input.get('seqused_cmp_kv')
    cu_seqlens_q = tensor_input.get('cu_seqlens_q')

    if seqused_q is not None:
        act_q_list = [_as_int(seqused_q[b]) for b in range(B)]
    elif cu_seqlens_q is not None:
        act_q_list = [_as_int(cu_seqlens_q[b + 1] - cu_seqlens_q[b]) for b in range(B)]
    else:
        act_q_list = [metadata_input['max_seqlen_q']] * B

    if seqused_ori_kv is not None:
        ori_kv_list = [_as_int(seqused_ori_kv[b]) for b in range(B)]
    else:
        ori_kv_list = [metadata_input['max_seqlen_ori_kv']] * B

    if seqused_cmp_kv is not None:
        cmp_kv_list = [_as_int(seqused_cmp_kv[b]) for b in range(B)]
    else:
        cmp_kv_list = [0] * B

    metadata = torch.zeros(1024, dtype=torch.int32)
    core_idx = 0
    for b_idx in range(B):
        cur_act_q = act_q_list[b_idx]
        cur_ori_kv = ori_kv_list[b_idx]
        cur_cmp_kv = cmp_kv_list[b_idx]
        for n2_idx in range(N2):
            bn2_idx = b_idx * N2 + n2_idx
            for s1_idx in range(cur_act_q):
                ori_len, cmp_len = _compute_row_s2_length(
                    b_idx, n2_idx, s1_idx, B, N2, cur_act_q, cur_ori_kv, cur_cmp_kv,
                    ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right, cmp_ratio)
                total_len = ori_len + cmp_len
                if total_len <= 0:
                    continue
                ori_block_num = math.ceil(ori_len / S2_BASE_SIZE) if ori_len > 0 else 0
                total_len = ori_len + cmp_len
                num_s2_blocks = math.ceil(total_len / S2_BASE_SIZE) if total_len > 0 else 0
                s2_block_boundaries = list(range(1, num_s2_blocks + 1))
                if num_s2_blocks <= 1:
                    if core_idx < AIC_CORE_MAX_NUM:
                        base = core_idx * FA_METADATA_SIZE
                        metadata[base + FA_CORE_ENABLE_INDEX] = 1
                        metadata[base + FA_BN2_START_INDEX] = bn2_idx
                        metadata[base + FA_M_START_INDEX] = s1_idx
                        metadata[base + FA_S2_START_INDEX] = 1
                        metadata[base + FA_BN2_END_INDEX] = bn2_idx
                        metadata[base + FA_M_END_INDEX] = s1_idx
                        metadata[base + FA_S2_END_INDEX] = 0
                        core_idx += 1
                    continue
                num_g_groups = max(1, min(AIC_CORE_MAX_NUM // num_s2_blocks, 2))
                for g_grp in range(num_g_groups):
                    for i in range(num_s2_blocks):
                        if core_idx >= AIC_CORE_MAX_NUM:
                            break
                        base = core_idx * FA_METADATA_SIZE
                        s2_start = s2_block_boundaries[i]
                        s2_end = s2_block_boundaries[i + 1] if i + 1 < num_s2_blocks else 0
                        metadata[base + FA_CORE_ENABLE_INDEX] = 1
                        metadata[base + FA_BN2_START_INDEX] = bn2_idx
                        metadata[base + FA_M_START_INDEX] = s1_idx
                        metadata[base + FA_S2_START_INDEX] = s2_start
                        metadata[base + FA_BN2_END_INDEX] = bn2_idx
                        metadata[base + FA_M_END_INDEX] = s1_idx
                        metadata[base + FA_S2_END_INDEX] = s2_end
                        core_idx += 1
    return metadata


def generate_smla_metadata(metadata_input, tensor_input):
    if USE_MOCK_METADATA:
        return _build_mock_metadata(metadata_input, tensor_input)
    _register_torch_extension()
    layout_q = tensor_input['layout_q'] if isinstance(tensor_input['layout_q'], str) else tensor_input['layout_q'][0]
    ori_sparse_indices = tensor_input['ori_sparse_indices']
    cmp_sparse_indices = tensor_input['cmp_sparse_indices']

    def maybe_npu(tensor):
        return tensor.npu() if tensor is not None else None

    return torch.ops.cann_ops_transformer.sparse_flash_mla_metadata(
        num_heads_q=metadata_input['N1'],
        num_heads_kv=metadata_input['N2'],
        head_dim=metadata_input['D'],
        cu_seqlens_q=maybe_npu(tensor_input.get('cu_seqlens_q')),
        cu_seqlens_ori_kv=maybe_npu(tensor_input.get('cu_seqlens_ori_kv')),
        cu_seqlens_cmp_kv=maybe_npu(tensor_input.get('cu_seqlens_cmp_kv')),
        seqused_q=maybe_npu(tensor_input.get('seqused_q')),
        seqused_ori_kv=maybe_npu(tensor_input.get('seqused_ori_kv')),
        seqused_cmp_kv=maybe_npu(tensor_input.get('seqused_cmp_kv')),
        cmp_residual_kv=maybe_npu(tensor_input.get('cmp_residual_kv')),
        ori_topk_length=None,
        cmp_topk_length=None,
        batch_size=metadata_input['B'],
        max_seqlen_q=metadata_input['max_seqlen_q'],
        max_seqlen_ori_kv=metadata_input['max_seqlen_ori_kv'],
        max_seqlen_cmp_kv=metadata_input['max_seqlen_cmp_kv'],
        ori_topk=metadata_input['K'] if ori_sparse_indices is not None else 0,
        cmp_topk=metadata_input['K'] if cmp_sparse_indices is not None else 0,
        cmp_ratio=metadata_input['cmp_ratio'] if metadata_input['cmp_ratio'] is not None else 1,
        ori_mask_mode=tensor_input['ori_mask_mode'],
        cmp_mask_mode=tensor_input['cmp_mask_mode'] if tensor_input['cmp_mask_mode'] is not None else 3,
        ori_win_left=tensor_input['ori_win_left'],
        ori_win_right=tensor_input['ori_win_right'],
        layout_q=layout_q,
        layout_kv=tensor_input['layout_kv'],
        has_ori_kv=tensor_input['ori_kv'] is not None,
        has_cmp_kv=tensor_input['cmp_kv'] is not None)


class GeneralizedSFA:
    def __init__(self, layout_q, layout_kv, q_type, ori_kv_type, cmp_kv_type, B, S1, S2, T1, N1, N2, D, K,
                 block_num1, block_num2, block_size1, block_size2, cu_seqlens_q, seqused_ori_kv, seqused_cmp_kv,
                 cmp_residual_kv, softmax_scale, cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_win_left,
                 ori_win_right, ori_topk_length, cmp_topk_length, metadata=None):
        self.layout_q = layout_q
        self.layout_kv = layout_kv
        self.q_type = q_type
        self.ori_kv_type = ori_kv_type
        self.cmp_kv_type = cmp_kv_type
        self.B = B
        self.S1 = S1
        self.S2 = S2
        self.T1 = T1
        self.N1 = N1
        self.N2 = N2
        self.D = D
        self.K = K
        self.block_num1 = block_num1
        self.block_num2 = block_num2
        self.block_size1 = block_size1
        self.block_size2 = block_size2
        self.cu_seqlens_q = cu_seqlens_q
        self.seqused_ori_kv=seqused_ori_kv
        self.seqused_cmp_kv=seqused_cmp_kv
        self.cmp_residual_kv=cmp_residual_kv
        # self.seqused_kv = seqused_kv
        self.softmax_scale = softmax_scale
        self.cmp_ratio = cmp_ratio
        self.ori_mask_mode = ori_mask_mode
        self.cmp_mask_mode = cmp_mask_mode
        self.ori_win_left = ori_win_left
        self.ori_win_right = ori_win_right
        self.ori_topk_length = ori_topk_length
        self.cmp_topk_length = cmp_topk_length
        self.metadata_info = parse_smla_metadata(metadata)
        self.fd_split_ranges = []

    def _bn2_idx(self, batch_idx, n2_idx):
        return batch_idx * self.N2 + n2_idx

    def _convert_s2_block_to_token(self, s2_block_idx, ori_len, cmp_len):
        total_len = ori_len + cmp_len
        block_idx = int(s2_block_idx - 1)
        return min(block_idx * S2_BASE_SIZE, total_len)

    def _build_fd_split_ranges(self, row_s2_lengths, m_block_size=1):
        if self.fd_split_ranges:
            return self.fd_split_ranges
        row_split_ranges = {}
        for fa in self.metadata_info.get("fa", []):
            bn2_start = fa["bn2_start"]
            m_start = fa["m_start"]
            bn2_end = fa["bn2_end"]
            m_end = fa["m_end"]
            s2_start_block = fa["s2_start_block"]
            s2_end_block = fa["s2_end_block"]

            def add_split_range(bn2_idx, m_idx, start_block, end_block):
                row_key = (bn2_idx, m_idx)
                if row_key not in row_s2_lengths:
                    return
                ori_len, cmp_len = row_s2_lengths[row_key]
                total_len = ori_len + cmp_len
                if total_len <= 0:
                    return
                start_token = self._convert_s2_block_to_token(start_block, ori_len, cmp_len) if start_block != 0 else 0
                end_token = self._convert_s2_block_to_token(end_block, ori_len, cmp_len) if end_block != 0 else total_len
                if end_token <= start_token:
                    return
                ranges = row_split_ranges.setdefault(row_key, [])
                if not any(r["start"] == start_token and r["end"] == end_token for r in ranges):
                    ranges.append({
                        "bn2_idx": bn2_idx,
                        "m_idx": m_idx,
                        "start": start_token,
                        "end": end_token,
                    })

            first_m_idx = m_start * m_block_size
            last_m_idx = m_end * m_block_size
            if s2_start_block != 0 and s2_end_block != 0 and bn2_end == bn2_start and last_m_idx == first_m_idx:
                add_split_range(bn2_start, first_m_idx, s2_start_block, s2_end_block)
            else:
                if s2_start_block != 0:
                    add_split_range(bn2_start, first_m_idx, s2_start_block, 0)
                if s2_end_block != 0:
                    add_split_range(bn2_end, last_m_idx, 0, s2_end_block)
        split_rows = []
        for row_key, ranges in row_split_ranges.items():
            if len(ranges) <= 1:
                continue
            ranges.sort(key=lambda item: item["start"])
            bn2_idx, m_idx = row_key
            split_rows.append({
                "bn2_idx": bn2_idx,
                "m_idx": m_idx,
                "ranges": ranges,
            })
        self.fd_split_ranges = split_rows
        return self.fd_split_ranges

    def _get_fd_ranges_for_row(self, bn2_idx, m_idx, head_start=None, head_num=None):
        for split_row in self.fd_split_ranges:
            if split_row["bn2_idx"] == bn2_idx and split_row["m_idx"] == m_idx:
                return split_row["ranges"]
        return None

    def _run_online_softmax_range(self, q_curr_fp32, cur_sinks, cur_ori_k_fp32, cur_cmp_k_fp32, start_token, end_token,
                                  include_sinks=True):
        ori_len = cur_ori_k_fp32.size(0)
        cmp_len = 0 if cur_cmp_k_fp32 is None else cur_cmp_k_fp32.size(0)
        if include_sinks:
            row_sum = torch.ones((q_curr_fp32.size(0)), dtype=torch.float32)
            row_max = cur_sinks.clone().to(dtype=torch.float32)
        else:
            row_sum = torch.zeros((q_curr_fp32.size(0)), dtype=torch.float32)
            row_max = torch.full((q_curr_fp32.size(0),), float('-inf'), dtype=torch.float32)
        cur_attn_out = torch.zeros((q_curr_fp32.size(0), q_curr_fp32.size(1)), dtype=torch.float32)
        segments = []
        ori_start = min(start_token, ori_len)
        ori_end = min(end_token, ori_len)
        if ori_end > ori_start:
            segments.append(cur_ori_k_fp32[ori_start:ori_end, :])
        cmp_start = max(start_token - ori_len, 0)
        cmp_end = min(end_token - ori_len, cmp_len)
        if cur_cmp_k_fp32 is not None and cmp_end > cmp_start:
            segments.append(cur_cmp_k_fp32[cmp_start:cmp_end, :])
        for segment in segments:
            for offset in range(0, segment.size(0), S2_BASE_SIZE):
                k_tile = segment[offset:offset + S2_BASE_SIZE, :]
                v_tile = k_tile.clone()
                scale_res = torch.matmul(q_curr_fp32, k_tile.T) * self.softmax_scale
                row_max_old = row_max.clone()
                row_max_tmp = torch.max(scale_res, dim=1)[0]
                row_max = torch.max(row_max, row_max_tmp)
                update_mul = torch.exp(row_max_old - row_max)
                cur_softmax_res = torch.exp(scale_res - row_max.unsqueeze(1))
                row_sum = update_mul * row_sum + torch.sum(cur_softmax_res, dim=1)
                cur_o = torch.matmul(cur_softmax_res.to(dtype=q_curr_fp32.dtype), v_tile)
                cur_attn_out = cur_attn_out * update_mul.unsqueeze(1) + cur_o
        return row_max, row_sum, cur_attn_out / row_sum.unsqueeze(1).clamp(min=1e-30)

    def _run_fd_golden(self, q_curr_fp32, cur_sinks, cur_ori_k_fp32, cur_cmp_k_fp32, fd_ranges):
        partial_max = []
        partial_sum = []
        partial_out = []
        for split_idx, fd_range in enumerate(fd_ranges):
            row_max, row_sum, out = self._run_online_softmax_range(
                q_curr_fp32, cur_sinks, cur_ori_k_fp32, cur_cmp_k_fp32, fd_range["start"], fd_range["end"],
                include_sinks=(split_idx == 0))
            partial_max.append(row_max)
            partial_sum.append(row_sum)
            partial_out.append(out)
        stacked_max = torch.stack(partial_max, dim=0)
        stacked_sum = torch.stack(partial_sum, dim=0)
        global_max = torch.max(stacked_max, dim=0)[0]
        lse_exp = torch.exp(stacked_max - global_max.unsqueeze(0)) * stacked_sum
        global_sum = torch.sum(lse_exp, dim=0)
        result = torch.zeros_like(partial_out[0])
        for split_idx, out in enumerate(partial_out):
            weight = (lse_exp[split_idx] / global_sum).unsqueeze(1)
            result = result + weight * out
        return result, global_max, global_sum

    def calculate_by_bnsd(self, q_bnsd, ori_k_bnsd, ori_sparse_indices_bnsd, cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_ori_kv, seqused_cmp_kv,
                          cmp_residual_kv, sinks, template_idx, cmp_k_bnsd=None, cmp_sparse_indices_bnsd=None, seqused_q=None,
                          ori_topk_length_bnsd=None, cmp_topk_length_bnsd=None, return_softmax_lse=False):
        attn_out = torch.zeros(q_bnsd.shape, dtype=q_bnsd.dtype)
        softmax_lse = None
        if return_softmax_lse:
            softmax_max = torch.zeros((q_bnsd.shape[0], ori_k_bnsd.shape[1], q_bnsd.shape[2], q_bnsd.shape[1] // ori_k_bnsd.shape[1]), dtype=torch.float32)
            softmax_sum = torch.zeros((q_bnsd.shape[0], ori_k_bnsd.shape[1], q_bnsd.shape[2], q_bnsd.shape[1] // ori_k_bnsd.shape[1]), dtype=torch.float32)
            softmax_lse = torch.zeros((q_bnsd.shape[0], ori_k_bnsd.shape[1], q_bnsd.shape[2], q_bnsd.shape[1] // ori_k_bnsd.shape[1]), dtype=torch.float32)
        B = q_bnsd.shape[0]
        if seqused_q is not None:
            act_q = seqused_q
        elif self.layout_q in ["TND"]:
            act_q = prefix_sum_to_original(cu_seqlens_q)
        else:
            act_q = [q_bnsd.shape[2]] * B
        G = int(self.N1 / self.N2)
        s2_base_size = 512
        row_s2_lengths = {}
        if self.metadata_info.get("fd"):
            for b_idx in range(B):
                cur_act_q_pre = _as_int(act_q[b_idx])
                if seqused_ori_kv != None:
                    cur_ori_act_kv_pre = _as_int(seqused_ori_kv[b_idx])
                elif cu_seqlens_ori_kv != None:
                    cur_ori_act_kv_pre = _as_int(cu_seqlens_ori_kv[b_idx + 1] - cu_seqlens_ori_kv[b_idx])
                else:
                    cur_ori_act_kv_pre = ori_k_bnsd.size()[2]
                if seqused_cmp_kv != None:
                    cur_cmp_act_kv_pre = _as_int(seqused_cmp_kv[b_idx])
                elif cu_seqlens_cmp_kv != None:
                    cur_cmp_act_kv_pre = _as_int(cu_seqlens_cmp_kv[b_idx + 1] - cu_seqlens_cmp_kv[b_idx])
                else:
                    cur_cmp_act_kv_pre = 0
                for n2_idx in range(self.N2):
                    bn2_idx = self._bn2_idx(b_idx, n2_idx)
                    for s1_idx in range(cur_act_q_pre):
                        if (self.ori_mask_mode == 3 or self.ori_mask_mode == 4) and s1_idx < cur_act_q_pre - cur_ori_act_kv_pre:
                            row_s2_lengths[(bn2_idx, s1_idx)] = (0, 0)
                            continue
                        if self.ori_mask_mode == 4:
                            ori_threshold = cur_ori_act_kv_pre - cur_act_q_pre + s1_idx + 1
                            ori_win_end = ori_threshold + self.ori_win_right
                            if self.ori_win_left == -1:
                                ori_win_start = 0
                            else:
                                ori_win_start = max(ori_threshold - self.ori_win_left - 1, 0)
                        elif self.ori_mask_mode == 3:
                            ori_threshold = cur_ori_act_kv_pre - cur_act_q_pre + s1_idx + 1
                            ori_win_end = ori_threshold + self.ori_win_right
                            ori_win_start = 0
                        elif self.ori_mask_mode == 0:
                            ori_win_start = 0
                            ori_win_end = cur_ori_act_kv_pre
                        else:
                            ori_win_start = 0
                            ori_win_end = 0
                        ori_len = max(0, ori_win_end - ori_win_start)
                        cmp_len = 0
                        if template_idx == 1:
                            threshold = 0
                            if self.cmp_mask_mode == 3:
                                threshold = math.floor((cur_ori_act_kv_pre - cur_act_q_pre + s1_idx + 1) / self.cmp_ratio)
                            elif self.cmp_mask_mode == 0:
                                threshold = math.floor(cur_ori_act_kv_pre / self.cmp_ratio)
                            cmp_len = max(0, min(threshold, cur_cmp_act_kv_pre))
                        elif template_idx == 2:
                            cmp_topk_id = cmp_sparse_indices_bnsd[b_idx, n2_idx, s1_idx, :]
                            empty_flag_pre, cur_cmp_k_pre = self.gather_kv(
                                cmp_k_bnsd, cmp_topk_id, b_idx, n2_idx, s1_idx, cur_ori_act_kv_pre,
                                cur_act_q_pre, self.cmp_mask_mode, self.K, cmp_topk_length_bnsd,
                                self.cmp_ratio, cur_cmp_act_kv_pre)
                            cmp_len = 0 if empty_flag_pre else cur_cmp_k_pre.size(0)
                        row_s2_lengths[(bn2_idx, s1_idx)] = (ori_len, cmp_len)
            self._build_fd_split_ranges(row_s2_lengths)

        for i_B in range(B):
            print(f"i_B = {i_B}/{B}")
            cur_act_q = act_q[i_B]
            if seqused_ori_kv != None:
                cur_ori_act_kv = seqused_ori_kv[i_B]
            elif cu_seqlens_ori_kv != None:
                cur_ori_act_kv = cu_seqlens_ori_kv[i_B + 1] - cu_seqlens_ori_kv[i_B]
            else:
                cur_ori_act_kv = ori_k_bnsd.size()[2]
            if seqused_cmp_kv != None:
                cur_cmp_act_kv = seqused_cmp_kv[i_B]
            elif cu_seqlens_cmp_kv != None:
                cur_cmp_act_kv = cu_seqlens_cmp_kv[i_B + 1] - cu_seqlens_cmp_kv[i_B]
            else:
                cur_cmp_act_kv = 0
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
                    if (self.ori_mask_mode == 3 or self.ori_mask_mode == 4) and i_S1 < cur_act_q - cur_ori_act_kv:  # 根据 ori_kv 判断行无效
                        attn_out[i_B, i_N2 * G: (i_N2 + 1) * G, i_S1, :] = torch.zeros([G, self.D], dtype=torch.float)
                        continue

                    q_curr = q_bnsd[i_B, i_N2 * G: (i_N2 + 1) * G, i_S1, :]
                    q_curr_fp32 = q_curr.to(dtype=torch.float32)

                    if template_idx == 1 or template_idx == 2:
                        if template_idx == 1:
                            threshold = 0
                            if self.cmp_mask_mode == 3:
                                threshold = math.floor((cur_ori_act_kv - cur_act_q + i_S1 + 1) / (self.cmp_ratio))
                            elif self.cmp_mask_mode == 0:
                                threshold = math.floor(cur_ori_act_kv / (self.cmp_ratio))
                            threshold = min(threshold, cur_cmp_act_kv)
                            if threshold <= 0:
                                empty_flag = True
                            else:
                                empty_flag = False
                            cur_cmp_k = cmp_k_bnsd[i_B, i_N2, :threshold, :] if not empty_flag else []

                        else:
                            cmp_topk_id = cmp_sparse_indices_bnsd[i_B, i_N2, i_S1, :]
                            empty_flag, cur_cmp_k = self.gather_kv(cmp_k_bnsd, cmp_topk_id, i_B, i_N2, i_S1, cur_ori_act_kv,
                                                                    cur_act_q, self.cmp_mask_mode, self.K, cmp_topk_length_bnsd,
                                                                    self.cmp_ratio, cur_cmp_act_kv)
                        if cur_cmp_k == []:
                            cmp_s2_loop_time = 0
                            cur_cmp_k_fp32 = []
                        else:
                            cmp_s2_loop_time = math.ceil(cur_cmp_k.size(0) / s2_base_size)
                            cur_cmp_k_fp32 = cur_cmp_k.to(dtype=torch.float32)
                    else:
                        empty_flag = True
                        cur_cmp_k = []
                        cur_cmp_k_fp32 = []
                        cmp_s2_loop_time = 0

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
                    empty_flag_ori = False

                    cur_attn_out = attn_out[i_B, i_N2 * G: (i_N2 + 1) * G, i_S1, :]
                    if RUN_MODE == 0:
                        if empty_flag:
                            k_concat = cur_ori_k_bnsd
                        else:
                            k_concat = torch.concat([cur_ori_k_bnsd, cur_cmp_k], dim=0)

                        k_concat_fp32 = k_concat.to(dtype=torch.float32)
                        v_concat_fp32 = k_concat_fp32.clone()

                        mm1_res = torch.matmul(q_curr_fp32, k_concat_fp32.T)
                        scale_res = mm1_res * self.softmax_scale
                        softmax_res, x_max, x_sum = self.sinks_softmax(scale_res, cur_sinks_expand)
                        mm2_res = torch.matmul(softmax_res.to(dtype=q_bnsd.dtype).to(dtype=torch.float), v_concat_fp32)
                        attn_out[i_B, i_N2 * G: (i_N2 + 1) * G, i_S1, :] = mm2_res.to(dtype=q_bnsd.dtype)
                        if return_softmax_lse:
                            softmax_max[batch, n2Idx, s1Idx, :] = x_max[:, 0]
                            softmax_sum[batch, n2Idx, s1Idx, :] = x_sum[:, 0]
                    elif RUN_MODE == 1:
                        bn2_idx = self._bn2_idx(i_B, i_N2)
                        fd_ranges = self._get_fd_ranges_for_row(bn2_idx, i_S1, 0, G)
                        if fd_ranges:
                            cur_ori_k_bnsd_fp32 = cur_ori_k_bnsd.to(dtype=torch.float32)
                            cur_cmp_k_for_fd = None if cur_cmp_k == [] else cur_cmp_k.to(dtype=torch.float32)
                            fd_out, fd_max, fd_sum = self._run_fd_golden(
                                q_curr_fp32, cur_sinks, cur_ori_k_bnsd_fp32, cur_cmp_k_for_fd, fd_ranges)
                            attn_out[i_B, i_N2 * G: (i_N2 + 1) * G, i_S1, :] = fd_out.to(dtype=q_bnsd.dtype)
                            if return_softmax_lse:
                                softmax_lse[i_B, i_N2, i_S1, :] = fd_max + torch.log(fd_sum + 1e-10)
                            continue
                        if empty_flag_ori:
                            ori_s2_loop_time = 0
                            total_s2_loop_time = 0
                            row_sum = torch.empty((G), dtype=torch.float32).uniform_(1.0, 1.0)
                        else:
                            ori_s2_loop_time = math.ceil(cur_ori_k_bnsd.size(0) / s2_base_size)
                            total_s2_loop_time = ori_s2_loop_time + cmp_s2_loop_time
                            cur_ori_k_bnsd_fp32 = cur_ori_k_bnsd.to(dtype=torch.float32)
                            row_sum = torch.empty((G), dtype=torch.float32).uniform_(1.0, 1.0)
                            row_max = torch.empty((G, 1), dtype=torch.float32)
                            row_max = cur_sinks

                        for i_S2 in range(total_s2_loop_time):
                            if i_S2 < ori_s2_loop_time: # ori_kv
                                if i_S2 < ori_s2_loop_time - 1:
                                    k_tile = cur_ori_k_bnsd_fp32[i_S2 * s2_base_size:(i_S2 + 1) * s2_base_size, :]
                                else:
                                    k_tile = cur_ori_k_bnsd_fp32[i_S2 * s2_base_size:, :]
                            else: # cmp_kv
                                if i_S2 < total_s2_loop_time - 1:
                                    k_tile = cur_cmp_k_fp32[(i_S2 - ori_s2_loop_time) * s2_base_size:(i_S2 - ori_s2_loop_time + 1) * s2_base_size, :]
                                else:
                                    k_tile = cur_cmp_k_fp32[(i_S2 - ori_s2_loop_time) * s2_base_size:, :]
                            v_tile = k_tile.clone()
                            mm1_res = torch.matmul(q_curr_fp32, k_tile.T)
                            scale_res = mm1_res * self.softmax_scale  # 外层for S1 循环，据实拷入数据，因此不需要mask
                            row_max_old = row_max.clone()
                            row_max_tmp = torch.max(scale_res, dim=1)[0]
                            # row_max_tmp = row_max_tmp.unsqueeze(1)
                            row_max = torch.max(row_max, row_max_tmp)
                            update_mul = torch.exp(row_max_old - row_max)
                            row_max_expand = row_max.unsqueeze(1)
                            update_mul_expand = update_mul.unsqueeze(1)
                            cur_softmax_res = torch.exp(scale_res - row_max_expand)
                            row_sum = update_mul * row_sum + torch.sum(cur_softmax_res, dim=1)
                            # print(row_sum)
                            cur_softmax_res = cur_softmax_res.to(dtype=q_bnsd.dtype).to(dtype=torch.float)
                            cur_o = torch.matmul(cur_softmax_res, v_tile)
                            cur_attn_out = cur_attn_out * update_mul_expand + cur_o
                        row_sum_expand = row_sum.unsqueeze(1)
                        attn_out[i_B, i_N2 * G: (i_N2 + 1) * G, i_S1, :] = (cur_attn_out / row_sum_expand).to(dtype=q_bnsd.dtype)
                        if return_softmax_lse:
                            softmax_lse[i_B, i_N2, i_S1, :] = row_max + torch.log(row_sum + 1e-10)
                    elif RUN_MODE == 2:
                        ori_s2_loop_time = math.ceil(cur_ori_k_bnsd.size(0) / s2_base_size)
                        total_s2_loop_time = ori_s2_loop_time + cmp_s2_loop_time

                        cur_ori_k_bnsd_fp32 = cur_ori_k_bnsd.to(dtype=torch.float32)
                        row_sum = torch.empty((G), dtype=torch.float32).uniform_(1.0, 1.0)  # due to sinks
                        row_max = torch.empty((G, 1), dtype=torch.float32)
                        row_max = cur_sinks
                        O_flash = torch.empty((G, q_bnsd.shape[3]), dtype=torch.float32).uniform_(2 ** (-80), 2 ** (-80)) # 注意D轴

                        rcof_old = torch.empty((G), dtype=torch.float32).uniform_(1.0, 1.0)

                        for i_S2 in range(total_s2_loop_time):
                            if i_S2 < ori_s2_loop_time: # ori_kv
                                if i_S2 < ori_s2_loop_time - 1:
                                    k_tile = cur_ori_k_bnsd_fp32[i_S2 * s2_base_size:(i_S2 + 1) * s2_base_size, :]
                                else:
                                    k_tile = cur_ori_k_bnsd_fp32[i_S2 * s2_base_size:, :]
                            else: # cmp_kv
                                if i_S2 < total_s2_loop_time - 1:
                                    k_tile = cur_cmp_k_fp32[(i_S2 - ori_s2_loop_time) * s2_base_size:(i_S2 + 1) * s2_base_size, :]
                                else:
                                    k_tile = cur_cmp_k_fp32[(i_S2 - ori_s2_loop_time) * s2_base_size:, :]
                            v_tile = k_tile.clone()

                            mm1_res = torch.matmul(q_curr_fp32, k_tile.T)
                            scale_res = mm1_res * self.softmax_scale  # 外层for S1 循环，据实拷入数据，因此不需要mask

                            row_max_old = row_max.clone()
                            row_max_tmp = torch.max(scale_res, dim=1)[0]
                            row_max = torch.max(row_max, row_max_tmp)
                            update_mul = torch.exp(row_max_old - row_max)
                            row_max_expand = row_max.unsqueeze(1)
                            log_2 = torch.log(torch.tensor(2.0))

                            if i_S2 == 0:
                                ni = torch.round((-row_max) / log_2)
                                ni_old = ni.clone()
                            else:
                                ni_old = ni.clone()
                                ni = torch.round((-row_max) / log_2)

                            cur_softmax_res = torch.exp(scale_res - row_max_expand)
                            row_sum = update_mul * row_sum + torch.sum(cur_softmax_res, dim=1)
                            tmp_scale = torch.exp(row_max / log_2 + ni) * log_2
                            tmp_scale_16 = tmp_scale.to(dtype=q_bnsd.dtype).to(dtype=torch.float)
                            tmp_scale_16_expand = tmp_scale_16.unsqueeze(1)
                            cur_softmax_res = cur_softmax_res * tmp_scale_16_expand
                            cur_softmax_res = cur_softmax_res.to(dtype=q_bnsd.dtype).to(dtype=torch.float)
                            cur_o = torch.matmul(cur_softmax_res, v_tile)
                            rcof = tmp_scale / tmp_scale_16
                            eps = (rcof_old / rcof - 1) * 1.5

                            if i_S2 != 0:
                                N = torch.clamp((ni - ni_old), min=-30).to(torch.float32) + eps + 1e-6
                                N_up = (N * torch.tensor(2 ** (23), dtype=torch.float32)).to(torch.int32)
                                N_up_expand = N_up.unsqueeze(1)
                                O_int = O_flash.view(torch.int32) + N_up_expand
                                O_fp = O_int.view(torch.float32)
                                O_flash = O_fp + cur_o
                            else:
                                O_flash = O_flash + cur_o
                            rcof_old = rcof
                        row_sum_expand = row_sum.unsqueeze(1)
                        attn_out[i_B, i_N2 * G: (i_N2 + 1) * G, i_S1, :] = (O_flash / row_sum_expand / tmp_scale_16_expand).to(dtype=q_bnsd.dtype)
                    else:
                        raise ValueError(f"unsupported RUN_MODE:{RUN_MODE}")
        return attn_out, softmax_lse

    def gather_kv(self, k_tensor, topk_id, i_B, i_N2, i_S1, cur_ori_act_kv, cur_act_q, mask_mode, K, topk_length_bnsd, cmp_ratio,
                    cur_cmp_act_kv=None, sparse_block_size=1):
        s2_sparse = list()
        if cur_cmp_act_kv is None:
            cur_cmp_act_kv = math.floor(cur_ori_act_kv / cmp_ratio)
        threshold = 0
        if mask_mode == 3:
            threshold = math.floor((cur_ori_act_kv - cur_act_q + i_S1 + 1) / cmp_ratio)
        elif mask_mode == 0:
            threshold = math.floor(cur_ori_act_kv / cmp_ratio)
        threshold = min(threshold, cur_cmp_act_kv)

        valid_count = min(K, math.ceil(threshold / sparse_block_size))
        if topk_length_bnsd != None:
            valid_count = min(topk_length_bnsd[i_B, 0, i_S1, 0], valid_count)

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
            cur_cmp_k = torch.tensor([])
            empty_flag = True
        else:
            cur_cmp_k = k_tensor[i_B, i_N2, s2_sparse, :]
        return empty_flag, cur_cmp_k

    def sinks_softmax(self, x, sinks):  # [G, S2] [G, 1]
        x = x.to(dtype=torch.float)
        x_concat = torch.cat([x, sinks], dim=1)
        x_max = x_concat.max(dim=-1, keepdims=True)[0]
        x_sub = x - x_max
        y = torch.exp(x_sub)
        x_sum = y.sum(dim=-1, keepdims=True) + torch.exp(sinks - x_max)
        ans = y / x_sum
        return ans, x_max, x_sum

    def trans_shape_to_bnsd(self, tensor, shape, layout, act_seq=None, seqused=None):
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
            seqused_per_batch = seqused if seqused is not None else act_seq_per_batch # padding
            new_tensor = torch.zeros((B, N, max_s1, D), dtype=tensor.dtype)
            # t_start = 0
            for b_index in range(B):
                t_start = int(act_seq[b_index])
                cur_act_seq = int(act_seq_per_batch[b_index])
                cur_seqused = int(seqused_per_batch[b_index])
                # t_end = t_start + cur_act_seq
                if cur_act_seq == 0:
                    continue
                for n_index in range(N):
                    new_tensor[b_index, n_index, 0:cur_seqused, :] = tensor[t_start:t_start + cur_seqused, n_index, :]
                # t_start += cur_act_seq
            return new_tensor, [B, N, max_s1, D]
        else:
            return tensor, shape

    def trans_topk_length_shape_to_bnsd(self, tensor, shape, layout, act_seq=None):
        if layout in ["BSND"]:
            B = shape[0]
            S = shape[1]
            tensor = tensor.reshape(B, 1, S, 1)
            return tensor, [B, 1, S, 1]
        elif layout in ["TND"]:
            T = shape[0]
            B = len(act_seq) - 1  # TND act_q is cumulative
            max_s1 = get_max_adjacent_diff(act_seq)
            act_seq_per_batch = prefix_sum_to_original(act_seq)
            new_tensor = torch.zeros((B, 1, max_s1, 1), dtype=tensor.dtype)
            t_start = 0
            for b_index in range(B):
                cur_act_seq = act_seq_per_batch[b_index]
                t_end = t_start + cur_act_seq
                if cur_act_seq == 0:
                    continue
                for n_index in range(1):
                    new_tensor[b_index, 0, 0:cur_act_seq, 0] = tensor[t_start:t_end]
                t_start += cur_act_seq
            return new_tensor, [B, 1, max_s1, 1]
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

    def lse_trans_bnsd_to_target_layout(self, tensor, layout, act_seq=None):
        if layout in ["BSND"]: #B N S G
            return tensor
        elif layout in ["TND"]: #B N2 S1 G  --> T1 N2 G --> N2 T1 G
            T = act_seq[-1]
            B = tensor.shape[0]
            N2 = tensor.shape[1]
            G = tensor.shape[3]
            output = torch.zeros((N2, T, G), dtype=torch.float)
            t_start = 0
            act_seq_per_batch = prefix_sum_to_original(act_seq)  # prefix_sum_to_original 还原成每个batch的真实长度
            for b_index in range(B):
                cur_act_seq = act_seq_per_batch[b_index]
                t_end = t_start + cur_act_seq
                if cur_act_seq == 0:
                    continue
                for n_index in range(N2):
                    output[n_index, t_start:t_end, :] = tensor[b_index, n_index, :cur_act_seq, :]
                    # output[t_start:t_end, n_index, :] = tensor[b_index, n_index, :cur_act_seq, :]
                t_start += cur_act_seq
            output = output.transpose(0, 1).contiguous()
            return output
        else:
            return tensor

    def forward(self, q, ori_k, ori_sparse_indices, cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_ori_kv,
                seqused_cmp_kv, cmp_residual_kv, sinks, template_idx, cmp_k=None, cmp_sparse_indices=None, seqused_q=None,
                ori_topk_length=None, cmp_topk_length=None, return_softmax_lse=False):
        q_bnsd, q_bnsd_shape = self.trans_shape_to_bnsd(q, q.shape, self.layout_q, cu_seqlens_q, seqused_q)
        ori_k_bnsd, ori_k_bnsd_shape = self.trans_shape_to_bnsd(ori_k, ori_k.shape, self.layout_kv, cu_seqlens_ori_kv, seqused_ori_kv)
        if template_idx == 1 or template_idx == 2:
            cmp_k_bnsd, cmp_k_bnsd_shape = self.trans_shape_to_bnsd(cmp_k, cmp_k.shape, self.layout_kv, cu_seqlens_cmp_kv, seqused_cmp_kv)
        else:
            cmp_k_bnsd = None
            cmp_k_bnsd_shape = None
        if template_idx == 2:
            cmp_sparse_indices_bnsd, cmp_sparse_indices_bnsd_shape = self.trans_shape_to_bnsd(cmp_sparse_indices,
                                                                     cmp_sparse_indices.shape, self.layout_q,
                                                                     cu_seqlens_q, seqused_q)
        else:
            cmp_sparse_indices_bnsd = None
        ori_sparse_indices_bnsd = None
        ori_topk_length_bnsd = None
        cmp_topk_length_bnsd = None
        cmp_topk_length_shape = None
        attn_out, softmax_lse = self.calculate_by_bnsd(q_bnsd, ori_k_bnsd, ori_sparse_indices_bnsd, cu_seqlens_q, cu_seqlens_ori_kv,
                                                       cu_seqlens_cmp_kv, seqused_ori_kv, seqused_cmp_kv, cmp_residual_kv, sinks, template_idx,
                                                       cmp_k_bnsd, cmp_sparse_indices_bnsd, seqused_q, ori_topk_length_bnsd,
                                                       cmp_topk_length_bnsd,return_softmax_lse)
        attn_out = self.trans_bnsd_to_target_layout(attn_out, self.layout_q, cu_seqlens_q)
        if return_softmax_lse:
            softmax_lse = self.lse_trans_bnsd_to_target_layout(softmax_lse, self.layout_q, cu_seqlens_q)
        return attn_out, softmax_lse

def prefix_sum_to_original(cu_seqlens_q):
    """
    从前缀和张量反向计算出原始的非前缀和张量（替代原列表逻辑）

    Args:
        cu_seqlens_q (torch.Tensor): 形状为 [B+1] 的一维前缀和张量（元素为数字类型，如int/float）

    Returns:
        torch.Tensor: 原始的非前缀和张量，形状为 [B]（与原列表长度一致）

    Raises:
        TypeError: 输入非tensor/非一维tensor
        ValueError: tensor长度<2（无法计算差值）
    """
    # 1. 基础类型校验：必须是torch.Tensor
    if not isinstance(cu_seqlens_q, torch.Tensor):
        raise TypeError(f"输入必须是torch.Tensor，当前类型：{type(cu_seqlens_q)}")

    # 2. 维度校验：必须是一维tensor（原列表对应一维）
    if cu_seqlens_q.ndim != 1:
        raise TypeError(f"输入必须是一维tensor，当前维度：{cu_seqlens_q.ndim}，形状：{cu_seqlens_q.shape}")

    # 3. 长度校验（前缀和tensor至少需2个元素才能反向计算）
    if len(cu_seqlens_q) < 2:
        raise ValueError(f"前缀和tensor长度需≥2，当前长度：{len(cu_seqlens_q)}")

    # 4. 核心逻辑：计算相邻元素差值（用tensor向量化运算替代循环，效率更高）
    # 原理：original_val[i] = cu_seqlens_q[i+1] - cu_seqlens_q[i]
    # 切片实现：cu_seqlens_q[1:] 取第2个到最后一个元素，cu_seqlens_q[:-1] 取第1个到倒数第2个元素
    original_tensor = cu_seqlens_q[1:] - cu_seqlens_q[:-1]

    return original_tensor

def get_max_adjacent_diff(cu_seqlens_q):
    """
    计算前缀和列表中相邻元素（后-前）的最大差值

    Args:
        cu_seqlens_q (list): 长度为 B+1 的前缀和列表

    Returns:
        float/int: 相邻元素的最大差值；若列表长度<2，返回 None
    """
    # 边界检查：列表长度不足2时无相邻元素
    if len(cu_seqlens_q) < 2:
        return None

    # 初始化最大差值为第一个相邻对的差值
    max_diff = cu_seqlens_q[1] - cu_seqlens_q[0]

    # 遍历所有相邻元素对（从第2对开始）
    for i in range(1, len(cu_seqlens_q)-1):
        current_diff = cu_seqlens_q[i+1] - cu_seqlens_q[i]
        # 更新最大差值
        if current_diff > max_diff:
            max_diff = current_diff

    return max_diff

def gen_sparse_indices_bsnd(cmp_ratio, B, S1, S2, N2, K, seqused_ori_kv, seqused_cmp_kv, cmp_residual_kv, mask_mode, sparse_indices_mode, kv_topk_mode,
                            seqused_q, cu_seqlens_q):
    if sparse_indices_mode == None: # 不传默认取full, 兼容老用例
        sparse_indices_mode = "full"
    if sparse_indices_mode not in ["full", "random"]:
        raise ValueError(f"sparse_indices_mode only support full/random, which is {sparse_indices_mode}")
    if kv_topk_mode not in ["full", "fullK", "random", "no"]:
        raise ValueError(f"kv_topk_mode only support no/full/random, which is {kv_topk_mode}")
    # 先生成sparse_indices
    # 有效索引在叠加了causal后有效tokens中选取，不足sparse_block_count，尾部填充-1
    sparse_indices = torch.full((B, S1, N2, K), fill_value=-1, dtype=torch.int32)
    if kv_topk_mode == "no":
        topk_length = None
    else:
        topk_length = torch.zeros((B, S1), dtype = torch.int32)
    for i_B in range(B):
        cur_act_kv = seqused_ori_kv[i_B] if seqused_ori_kv is not None else S2
        if seqused_cmp_kv is not None:
            cur_cmp_act_kv = seqused_cmp_kv[i_B]
        else:
            cur_cmp_act_kv = math.floor(cur_act_kv / cmp_ratio)
        if seqused_q is not None:
            cur_act_q = seqused_q[i_B]
        else:
            cur_act_q = cu_seqlens_q[i_B + 1] - cu_seqlens_q[i_B]
        for i_N2 in range(N2):
            for i_S1 in range(cur_act_q):
                if mask_mode == 3:
                    cur_valid_s2_max = math.floor((cur_act_kv - cur_act_q + i_S1 + 1) / cmp_ratio)
                elif mask_mode == 0:
                    cur_valid_s2_max = math.floor(cur_act_kv / cmp_ratio)
                else:
                    raise ValueError(f"mask_mode only support 0/3, which is {mask_mode}")
                cur_valid_s2_max = min(cur_valid_s2_max, cur_cmp_act_kv)
                if sparse_indices_mode == "random": # sparse_indices random模式, 长度随机值生成
                    if cur_valid_s2_max > 1:
                        cur_valid_s2_max_update = torch.randint(1, cur_valid_s2_max, (1, 1))[0]
                    else:
                        cur_valid_s2_max_update = 1
                else:
                    cur_valid_s2_max_update = cur_valid_s2_max

                valid_blocks_max = max(0, cur_valid_s2_max_update)
                block_indices = torch.randperm(valid_blocks_max).to(torch.int32)
                valid_blocks_topk = min(valid_blocks_max, K)
                sparse_indices[i_B, i_S1, i_N2, :valid_blocks_topk] = block_indices[0:valid_blocks_topk]
                if kv_topk_mode == "full":
                    topk_length[i_B, i_S1] = min(cur_valid_s2_max_update, K)
                elif kv_topk_mode == "fullK":
                    topk_length[i_B, i_S1] = min(cur_valid_s2_max, K)
                elif kv_topk_mode == "random":
                    if K > 1:
                        topk_length[i_B, i_S1] = torch.randint(1, K, (1, 1))[0]
                    else:
                        topk_length[i_B, i_S1] = 1
    return sparse_indices, topk_length

def gen_sparse_indices_tnd(cmp_ratio, B, T1, N2, K, cu_seqlens_q, cu_seqlens_kv, seqused_ori_kv, seqused_cmp_kv, cmp_residual_kv, mask_mode, sparse_indices_mode,
                            kv_topk_mode, seqused_q):
    if sparse_indices_mode == None: # 不传默认取full, 兼容老用例
        sparse_indices_mode = "full"
    if sparse_indices_mode not in ["full", "random"]:
        raise ValueError(f"sparse_indices_mode only support full/random, which is {sparse_indices_mode}")
    if kv_topk_mode not in ["full", "fullK", "random", "no"]:
        raise ValueError(f"kv_topk_mode only support no/full/random, which is {kv_topk_mode}")
    # 有效索引在叠加了causal后有效tokens中选取，不足sparse_block_count，尾部填充-1
    sparse_indices = torch.full((T1, N2, K), fill_value=-1, dtype=torch.int32)
    if kv_topk_mode == "no":
        topk_length = None
    else:
        topk_length = torch.zeros((T1), dtype = torch.int32)
    for i_B in range(B):
        if seqused_q is not None:
            cur_act_q = seqused_q[i_B]
        else:
            cur_act_q = cu_seqlens_q[i_B + 1] - cu_seqlens_q[i_B]
        s1_prefix = cu_seqlens_q[i_B]
        if seqused_ori_kv != None:
            cur_act_kv = seqused_ori_kv[i_B]
        else:
            cur_act_kv = cu_seqlens_kv[i_B + 1] - cu_seqlens_kv[i_B]
        if seqused_cmp_kv != None:
            cur_cmp_act_kv = seqused_cmp_kv[i_B]
        else:
            cur_cmp_act_kv = math.floor(cur_act_kv / cmp_ratio)
        for i_N2 in range(N2):
            for i_S1 in range(cur_act_q):
                if mask_mode == 3:
                    cur_valid_s2_max = math.floor((cur_act_kv - cur_act_q + i_S1 + 1) / cmp_ratio)
                elif mask_mode == 0:
                    cur_valid_s2_max = math.floor(cur_act_kv / cmp_ratio)
                else:
                    raise ValueError(f"mask_mode only support 0/3, which is {mask_mode}")
                cur_valid_s2_max = min(cur_valid_s2_max, cur_cmp_act_kv)
                if sparse_indices_mode == "random": # sparse_indices random模式, 长度随机值生成
                    if cur_valid_s2_max > 1:
                        cur_valid_s2_max_update = torch.randint(1, cur_valid_s2_max, (1, 1))[0]
                    else:
                        cur_valid_s2_max_update = 1
                else:
                    cur_valid_s2_max_update = cur_valid_s2_max
                valid_blocks_max = max(0, cur_valid_s2_max_update)
                block_indices = torch.randperm(valid_blocks_max).to(torch.int32)
                valid_blocks_topk = min(valid_blocks_max, K)
                sparse_indices[s1_prefix + i_S1, i_N2, :valid_blocks_topk] = block_indices[0:valid_blocks_topk]
                if kv_topk_mode == "full":
                    topk_length[s1_prefix + i_S1] = min(cur_valid_s2_max_update, K)
                elif kv_topk_mode == "fullK":
                    topk_length[s1_prefix + i_S1] = min(cur_valid_s2_max, K)
                elif kv_topk_mode == "random":
                    if K > 1:
                        topk_length[s1_prefix + i_S1] = torch.randint(1, K, (1, 1))[0]
                    else:
                        topk_length[s1_prefix + i_S1] = 1
    return sparse_indices, topk_length

def gen_ori_kv(layout_q, layout_kv, ori_kv_type, B, S1, S2, T1, T2, N2, D, block_num1, block_size1,
    cu_seqlens_q, cu_seqlens_ori_kv, seqused_ori_kv, seqused_cmp_kv, cmp_residual_kv, ori_mask_mode,
    ori_sparse_indices_mode, ori_kv_topk_mode, template_idx, data_range_left=DATA_RANGE_LEFT,
    data_range_right=DATA_RANGE_RIGHT):
    if layout_kv == "PA_BBND":
        ori_max_s2 = max(seqused_ori_kv)
        ori_max_block_num_per_batch = math.ceil(ori_max_s2 / block_size1)

        ori_k_bnsd = (torch.rand((B, N2, ori_max_s2, D)) * (data_range_left - data_range_right) + data_range_left).to(ori_kv_type)
        ori_block_num_per_batch = []
        ori_block_num_sum = 0

        for cur_ori_act_kv in seqused_ori_kv:
            cur_ori_kv_block_num = math.ceil(cur_ori_act_kv / block_size1)
            ori_block_num_per_batch.append(cur_ori_kv_block_num)
            ori_block_num_sum += cur_ori_kv_block_num

        if block_num1 < ori_block_num_sum:
            raise ValueError(f"ori_kv actual_block_num < needed_block_num, which is {block_num1} < {ori_block_num_sum}")

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
        ori_k_expand = torch.zeros((B, N2, ori_max_block_num_per_batch * block_size1, D), dtype=ori_kv_type)
        ori_k_expand[:, :, :ori_max_s2, :] = ori_k_bnsd
        ori_k_in_pa_shape = torch.zeros((block_num1, block_size1, N2, D), dtype=ori_kv_type)

        for i_B in range(B):
            for i_block, cur_block_id in enumerate(ori_block_table[i_B]):
                block_start_pos = i_block * block_size1
                if cur_block_id == -1:
                    continue
                else:
                    for i_N2 in range(N2):
                        ori_k_in_pa_shape[cur_block_id, :, i_N2, :] = \
                            ori_k_expand[i_B, i_N2, block_start_pos:block_start_pos + block_size1, :]

        ori_block_table = torch.tensor(ori_block_table).to(torch.int32)
        ori_k = ori_k_bnsd
    elif layout_kv == "TND":
        ori_block_table = None
        ori_max_s2 = get_max_adjacent_diff(cu_seqlens_ori_kv)
        ori_k_tnd = (torch.rand((T2, N2, D)) * (data_range_left - data_range_right) + data_range_left).to(ori_kv_type)
        ori_k_in_pa_shape = ori_k_tnd
        ori_k = ori_k_tnd
    elif layout_kv == "BSND":
        ori_block_table = None
        ori_max_s2 = S2
        ori_k_bnsd = (torch.rand((B, ori_max_s2, N2, D)) * (data_range_left - data_range_right) + data_range_left).to(ori_kv_type)
        ori_k_in_pa_shape = ori_k_bnsd
        ori_k = ori_k_bnsd

    ori_sparse_indices = None
    ori_topk_length = None

    return ori_k_in_pa_shape, ori_block_table, ori_k, ori_sparse_indices, ori_topk_length

def gen_cmp_kv(layout_q, layout_kv, cmp_kv_type, B, S1, S2, T1, T2, T3, N2, D, K, block_num2, block_size2,
    seqused_q, cu_seqlens_q, cu_seqlens_ori_kv, seqused_ori_kv, seqused_cmp_kv, cmp_residual_kv, cmp_ratio,
    cmp_mask_mode, cmp_sparse_indices_mode, cmp_kv_topk_mode, template_idx, data_range_left=DATA_RANGE_LEFT,
    data_range_right=DATA_RANGE_RIGHT):
    if cmp_ratio is None:
        raise ValueError(f"cmp_ratio can't be None")
    if cmp_ratio < 1 or cmp_ratio > 128:
        raise ValueError(f"cmp_ratio should be in range [1, 128], but got {cmp_ratio}")

    if layout_kv == "PA_BBND":
        ori_max_s2 = max(seqused_ori_kv)
        cmp_max_s2 = math.floor(ori_max_s2 / cmp_ratio)
        cmp_max_block_num_per_batch = math.ceil(cmp_max_s2 / block_size2)

        cmp_k = (torch.rand((B, N2, cmp_max_s2, D)) * (data_range_left - data_range_right) + data_range_left).to(cmp_kv_type)
        cmp_block_num_per_batch = []
        cmp_block_num_sum = 0
        for cur_ori_act_kv in seqused_ori_kv:
            cur_cmp_act_kv = math.floor(cur_ori_act_kv / cmp_ratio)
            cur_cmp_kv_block_num = math.ceil(cur_cmp_act_kv / block_size2)
            cmp_block_num_per_batch.append(cur_cmp_kv_block_num)
            cmp_block_num_sum += cur_cmp_kv_block_num
        if block_num2 < cmp_block_num_sum:
            raise ValueError(f"cmp_kv actual_block_num < needed_block_num, which is {block_num2} < {cmp_block_num_sum}")

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

        cmp_k_expand = torch.zeros((B, N2, cmp_max_block_num_per_batch * block_size2, D), dtype=cmp_kv_type)
        cmp_k_expand[:, :, :cmp_max_s2, :] = cmp_k
        cmp_k_in_pa_shape = torch.zeros((block_num2, block_size2, N2, D), dtype=cmp_kv_type)
        for i_B in range(B):
            for i_block, cur_block_id in enumerate(cmp_block_table[i_B]):
                block_start_pos = i_block * block_size2
                if cur_block_id == -1:
                    continue
                else:
                    for i_N2 in range(N2):
                        cmp_k_in_pa_shape[cur_block_id, :, i_N2, :] = \
                            cmp_k_expand[i_B, i_N2, block_start_pos:block_start_pos + block_size2, :]

        cmp_block_table = torch.tensor(cmp_block_table).to(torch.int32)
        if cmp_block_table.shape[1] == 0:
            cmp_block_table = None
    elif layout_kv == "TND":
        cmp_block_table = None
        cmp_max_s2 = get_max_adjacent_diff([x // cmp_ratio for x in cu_seqlens_ori_kv])
        cmp_k = (torch.rand((T3, N2, D)) * (data_range_left - data_range_right) + data_range_left).to(cmp_kv_type)
        cmp_k_in_pa_shape = cmp_k
    elif layout_kv == "BSND":
        cmp_block_table = None
        ori_max_s2 = S2
        cmp_max_s2 = math.floor(ori_max_s2 / cmp_ratio)
        cmp_k = (torch.rand((B, cmp_max_s2, N2, D)) * (data_range_left - data_range_right) + data_range_left).to(cmp_kv_type)
        cmp_k_in_pa_shape = cmp_k

    # generate cmp_sparse_indices
    if template_idx == 2:
        if layout_q == "BSND":
            cmp_sparse_indices, cmp_topk_length = gen_sparse_indices_bsnd(cmp_ratio, B, S1, S2, N2, K, seqused_ori_kv, seqused_cmp_kv, cmp_residual_kv,
                                    cmp_mask_mode, cmp_sparse_indices_mode, cmp_kv_topk_mode, seqused_q, cu_seqlens_q)
        elif layout_q == "TND":
            cmp_sparse_indices, cmp_topk_length = gen_sparse_indices_tnd(cmp_ratio, B, T1, N2, K, cu_seqlens_q, cu_seqlens_ori_kv, seqused_ori_kv, seqused_cmp_kv,
                                    cmp_residual_kv, cmp_mask_mode, cmp_sparse_indices_mode, cmp_kv_topk_mode, seqused_q)
    else:
        cmp_sparse_indices = None
        cmp_topk_length = None
    return cmp_k_in_pa_shape, cmp_sparse_indices, cmp_block_table, cmp_k, cmp_topk_length

def gen_data(params, template_mode=None):
    # 从字典中提取参数
    layout_q = params.get('layout_q')
    layout_kv = params.get('layout_kv')
    q_type = params.get('q_type')
    ori_kv_type = params.get('ori_kv_type')
    cmp_kv_type = params.get('cmp_kv_type')
    B = params.get('B')
    S1 = params.get('S1')
    T1 = params.get('T1')
    T2 = params.get('T2')
    T3 = params.get('T3')
    N1 = params.get('N1')
    N2 = params.get('N2')
    D = params.get('D')
    K = params.get('K')
    block_num1 = params.get('block_num1')
    block_num2 = params.get('block_num2')
    block_size1 = params.get('block_size1')
    block_size2 = params.get('block_size2')
    cu_seqlens_q = params.get('cu_seqlens_q')
    cu_seqlens_ori_kv = params.get('cu_seqlens_ori_kv')
    cu_seqlens_cmp_kv = params.get('cu_seqlens_cmp_kv')
    # seqused_kv = params.get('seqused_kv')
    seqused_ori_kv = params.get('seqused_ori_kv')
    seqused_cmp_kv = params.get('seqused_cmp_kv')
    cmp_residual_kv = params.get('cmp_residual_kv')
    softmax_scale = params.get('softmax_scale')
    cmp_ratio = params.get('cmp_ratio')
    ori_mask_mode = params.get('ori_mask_mode')
    cmp_mask_mode = params.get('cmp_mask_mode')
    ori_win_left = params.get('ori_win_left')
    ori_win_right = params.get('ori_win_right')
    case_name = params.get('testcase_name')
    S2 = params.get('S2')
    q_datarange = params.get('q_datarange')
    ori_kv_datarange = params.get('ori_kv_datarange')
    cmp_kv_datarange = params.get('cmp_kv_datarange')
    seqused_q = params.get('seqused_q')
    ori_kv_topk_mode = params.get('ori_kv_topk_mode')
    cmp_kv_topk_mode = params.get('cmp_kv_topk_mode')
    ori_sparse_indices_mode = params.get('ori_sparse_indices_mode')
    cmp_sparse_indices_mode = params.get('cmp_sparse_indices_mode')
    return_softmax_lse = params.get('return_softmax_lse')

    if layout_q == "TND" and T1 == None:
        raise ValueError(f"T1 must be provided when layout_kv is TND")
    if layout_kv == "TND" and T2 == None:
        raise ValueError(f"T2 must be provided when layout_kv is TND")
    if layout_kv == "BSND" and S2 == None:
        raise ValueError(f"S2 must be provided when layout_kv is BSND")

    if cu_seqlens_q is not None:
        cu_seqlens_q = torch.tensor(cu_seqlens_q).to(torch.int32)
    if cu_seqlens_ori_kv is not None:
        cu_seqlens_ori_kv = torch.tensor(cu_seqlens_ori_kv).to(torch.int32)
    if cu_seqlens_cmp_kv is not None:
        cu_seqlens_cmp_kv = torch.tensor(cu_seqlens_cmp_kv).to(torch.int32)
    if seqused_ori_kv is not None:
        seqused_ori_kv = torch.tensor(seqused_ori_kv).to(torch.int32)
    if seqused_cmp_kv is not None:
        seqused_cmp_kv = torch.tensor(seqused_cmp_kv).to(torch.int32)
    if cmp_residual_kv is not None:
        cmp_residual_kv = torch.tensor(cmp_residual_kv).to(torch.int32)
    if seqused_q is not None:
        seqused_q = torch.tensor(seqused_q).to(torch.int32)

    if layout_kv == "PA_BBND" and len(seqused_ori_kv) != B:
        raise ValueError(f"len(seqused_ori_kv) != B, which is {len(seqused_ori_kv)} != {B}")
    elif layout_kv == "TND" :
        if len(cu_seqlens_ori_kv) != B + 1:
            raise ValueError(f"len(cu_seqlens_ori_kv) != B, which is {len(cu_seqlens_ori_kv)} != {B + 1}")
        if cu_seqlens_ori_kv[-1] != T2:
            raise ValueError(f"cu_seqlens_ori_kv[-1] != T2, which is {cu_seqlens_ori_kv[-1]} != {T2}")
    else:
        pass

    # 获取最长 q
    if layout_q == 'TND':
        seq_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        max_seqlen_q = torch.max(seq_lens).item()
    else:
        max_seqlen_q = S1

    # generate q
    if layout_q == "BSND":
        S1, B = int(S1), int(B)
        q = (torch.rand((B, S1, N1, D)) * (q_datarange[1] - q_datarange[0]) + q_datarange[0]).to(q_type)
        if seqused_q is None:
            act_q = B * [S1]
            #seqused_q = torch.tensor(B * [S1]).to(torch.int32)
        else:
            act_q = seqused_q
            seqused_q = torch.tensor(seqused_q).to(torch.int32)
    elif layout_q == "TND":
        T1, B = int(T1), int(B)
        q = (torch.rand((T1, N1, D)) * (q_datarange[1] - q_datarange[0]) + q_datarange[0]).to(q_type)
        if len(cu_seqlens_q) != (B + 1):
            raise ValueError(f"len(cu_seqlens_q) != B + 1, which is {len(cu_seqlens_q)} != {B + 1}")
        else:
            act_q = prefix_sum_to_original(cu_seqlens_q)
            max_s1 = get_max_adjacent_diff(cu_seqlens_q)
    else:
        raise ValueError(f"layout_q is not support {layout_q}")

    # generate ori_kv/cmp_kv (only support PA_BBND)
    if layout_kv in ["PA_BBND", "TND", "BSND"]:
        pass
    else:
        raise ValueError(f"layout_kv is not support {layout_kv}")

    if seqused_ori_kv is not None and cmp_ratio is not None:
        if cmp_ratio < 1:
            raise ValueError(f"cmp_ratio should be in range [1, 128], but got {cmp_ratio}")
        if seqused_cmp_kv is None:
            seqused_cmp_kv = seqused_ori_kv // cmp_ratio
        if cmp_residual_kv is None:
            cmp_residual_kv = seqused_ori_kv % cmp_ratio
    # 路由到三个算子的逻辑：
    template_idx = 0

    mapping = {"SWA": 0, "HCA": 1, "CSA": 2}
    if template_mode is not None and template_mode in mapping:
        template_run_mode = template_mode
        template_idx = mapping[template_mode]
        if template_mode == "SWA":
            cmp_ratio = None
            K = None
        elif template_mode == "HCA":
            cmp_ratio = int(cmp_ratio)
            K = None
        elif template_mode == "CSA":
            cmp_ratio = int(cmp_ratio)
            K = int(K)
    else:
        if K is None or K == ['None']:
            if cmp_ratio is None:
                template_idx = 0  # SWA
                template_run_mode = 'SWA'
            else:
                template_idx = 1  # HCA
                template_run_mode = 'HCA'
                cmp_ratio = int(cmp_ratio)
        else:
            template_idx = 2  # CSA
            template_run_mode = 'CSA'
            if cmp_ratio == None:
                cmp_ratio = 1
            else:
                cmp_ratio = int(cmp_ratio)
            K = int(K)
    print("template_run_mode: ", template_run_mode)
    if layout_kv == "PA_BBND":
        block_size1, block_num1 = int(block_size1), int(block_num1)
        if template_idx != 0:
            block_size2, block_num2 = int(block_size2), int(block_num2)

    ori_k_in_pa_shape, ori_block_table, ori_k, ori_sparse_indices, ori_topk_length = gen_ori_kv(layout_q, layout_kv, ori_kv_type, B, S1, S2, T1, T2, N2, D, block_num1,
        block_size1, cu_seqlens_q, cu_seqlens_ori_kv, seqused_ori_kv, seqused_cmp_kv, cmp_residual_kv, ori_mask_mode, ori_sparse_indices_mode, ori_kv_topk_mode,
        template_idx, ori_kv_datarange[0], ori_kv_datarange[1])
    if template_idx == 1 or template_idx == 2:
        cmp_k_in_pa_shape, cmp_sparse_indices, cmp_block_table, cmp_k, cmp_topk_length = gen_cmp_kv(layout_q, layout_kv, cmp_kv_type, B, S1, S2,
                                                                                        T1, T2, T3, N2, D, K, block_num2,
                                                                                        block_size2, seqused_q, cu_seqlens_q, cu_seqlens_ori_kv,
                                                                                        seqused_ori_kv, seqused_cmp_kv, cmp_residual_kv, cmp_ratio,
                                                                                        cmp_mask_mode, cmp_sparse_indices_mode, cmp_kv_topk_mode,
                                                                                        template_idx, cmp_kv_datarange[0], cmp_kv_datarange[1])
    else:
        cmp_k_in_pa_shape = None
        cmp_sparse_indices = None
        cmp_block_table = None
        cmp_k = None
        cmp_topk_length = None

    sinks = (torch.rand((N1,)) * (q_datarange[1] - q_datarange[0])/10 + q_datarange[0]/10).to(torch.float)

    # kv_cache 0轴非连续 (NPU not available, skipping)
    # properties = torch.npu.get_device_properties()
    # if "Ascend950" in properties.name and layout_kv == "PA_BBND":
    #     key_stride = 10
    #     bytes_per_token = D + key_stride
    #     blockFusion1 = torch.zeros((block_num1, block_size1 * N2 * bytes_per_token), dtype=ori_kv_type)
    #     ori_key_flat = ori_k_in_pa_shape.view(block_num1, block_size1 * N2 * D)
    #     blockFusion1[:, :block_size1 * N2 * D] = ori_key_flat
    #     blockFusion1 = blockFusion1.npu()
    #     ori_k_in_pa_shape = blockFusion1[:, :block_size1 * N2 * D].view(block_num1, block_size1, N2, D)
    #
    #     if template_idx != 0:
    #         blockFusion2 = torch.zeros((block_num2, block_size2 * N2 * bytes_per_token), dtype=cmp_kv_type)
    #         cmp_key_flat = cmp_k_in_pa_shape.view(block_num2, block_size2 * N2 * D)
    #         blockFusion2[:, :block_size2 * N2 * D] = cmp_key_flat
    #         blockFusion2 = blockFusion2.npu()
    #         cmp_k_in_pa_shape = blockFusion2[:, :block_size2 * N2 * D].view(block_num2, block_size2, N2, D)

    metadata_input = {
        'N1': N1,
        'N2': N2,
        'D': D,
        'cu_seqlens_q': cu_seqlens_q,
        'cu_seqlens_ori_kv': cu_seqlens_ori_kv,
        'cu_seqlens_cmp_kv': cu_seqlens_cmp_kv,
        'seqused_q': seqused_q,
        'seqused_ori_kv': seqused_ori_kv,
        'seqused_cmp_kv': seqused_cmp_kv,
        'cmp_residual_kv': cmp_residual_kv,
        'ori_topk_length': None,
        'cmp_topk_length': None,
        'B': B,
        'max_seqlen_q': max_seqlen_q,
        'max_seqlen_ori_kv': max(seqused_ori_kv) if seqused_ori_kv is not None else (T2 if layout_kv == "TND" else S2),
        'max_seqlen_cmp_kv': max(seqused_cmp_kv) if seqused_cmp_kv is not None else 0,
        'K': K,
        'cmp_ratio': cmp_ratio,
        'ori_mask_mode': ori_mask_mode,
        'cmp_mask_mode': cmp_mask_mode,
        'ori_win_left': ori_win_left,
        'ori_win_right': ori_win_right,
        'layout_q': layout_q,
        'layout_kv': layout_kv,
    }
    tensor_input = {
        'q': q,
        'ori_kv': ori_k if layout_kv == "TND" else ori_k_in_pa_shape,
        'cmp_kv': cmp_k_in_pa_shape,
        'ori_sparse_indices': ori_sparse_indices,
        'cmp_sparse_indices': cmp_sparse_indices,
        'ori_block_table': ori_block_table,
        'cmp_block_table': cmp_block_table,
        'cu_seqlens_q': cu_seqlens_q,
        'cu_seqlens_ori_kv':cu_seqlens_ori_kv,
        'cu_seqlens_cmp_kv':cu_seqlens_cmp_kv,
        'seqused_q': seqused_q,
        # 'seqused_kv': seqused_kv,
        'seqused_ori_kv': seqused_ori_kv,
        'seqused_cmp_kv': seqused_cmp_kv,
        'cmp_residual_kv': cmp_residual_kv,
        'sinks': sinks,
        'softmax_scale': softmax_scale,
        'ori_mask_mode': ori_mask_mode,
        'cmp_mask_mode': cmp_mask_mode,
        'ori_win_left': ori_win_left,
        'ori_win_right': ori_win_right,
        'layout_q': layout_q,
        'layout_kv': layout_kv
    }
    metadata = generate_smla_metadata(metadata_input, tensor_input)

    test_sas = GeneralizedSFA(layout_q, layout_kv, q_type, ori_kv_type, cmp_kv_type, B, S1, S2, T1, N1, N2, D, K,
                              block_num1, block_num2, block_size1, block_size2, cu_seqlens_q, seqused_ori_kv, seqused_cmp_kv, cmp_residual_kv, softmax_scale,
                              cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right, ori_topk_length, cmp_topk_length, metadata)
    cpu_result, softmax_lse = test_sas.forward(q, ori_k, ori_sparse_indices, cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_ori_kv, seqused_cmp_kv,
                                               cmp_residual_kv, sinks, template_idx, cmp_k, cmp_sparse_indices, seqused_q, ori_topk_length, cmp_topk_length, return_softmax_lse)
    input_data = {
        # 命名用变量
        'B': B,
        'S1': S1,
        'S2': S2,
        'T1': T1,
        'T2': T2,
        'T3': T3,
        'N1': N1,
        'N2': N2,
        'K': K,
        'layout_q': layout_q,
        'layout_kv': layout_kv,
        'template_run_mode': template_run_mode,
        'case_name': case_name,
        # 固定参数
        'params': params,
        # 输入张量
        'metadata_input': metadata_input,
        'metadata': metadata,
        'input': tensor_input,
        'cpu_output': cpu_result,
        'softmax_lse': softmax_lse
    }
    return input_data

def save_test_case(input_data, output_dir):
    """
    保存单条测试用例到文件
    """
    print("正在保存pt文件...")
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    case_name = f"{input_data['template_run_mode']}_layoutQ_{input_data['layout_q']}_layoutKV_{input_data['layout_kv']}_B_{input_data['B']}_S1_{input_data['S1']}_S2_{input_data['S2']}_T1_{input_data['T1']}_N1_{input_data['N1']}_N2_{input_data['N2']}_{input_data['case_name']}"

    # 生成文件名
    input_filename = f"sas_case_{case_name}.pt"
    input_filepath = os.path.join(output_dir, input_filename)

    # 保存数据
    torch.save(input_data, input_filepath)
    print(f"测试用例已保存到: {input_filepath}")
    return input_filepath
