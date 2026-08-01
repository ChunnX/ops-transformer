#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""UndGenQkvRmsNormRopeCache 的 ST golden / aclnn 执行器。

golden 逻辑与算子设计文档配套的 golden.py 完全一致：
  split -> index(cat_indices) -> RMSNorm(Q/K) -> MRoPE(Q/K) -> Q 输出 + K/V 按 slot_mapping 写入 KV Cache。

NOTE: kernel 计算实现尚未补齐，本 ST 当前会因输出全 0 而失败；实现落地后即可直接使用。
"""
import copy

import torch

from atk.common.log import Logger
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi

MROPE_AXIS_NUM = 3


def mrope_axis_map(head_dim, mrope_section):
    """每个 half 索引取哪一轴的 cos/sin（与 kernel 侧 axisLut 规则一致）。"""
    half = head_dim // 2
    if not mrope_section:
        mrope_section = [half, 0, 0]
    idx = torch.arange(half, dtype=torch.int64)
    axis = torch.zeros(half, dtype=torch.int64)
    axis[(idx % 3 == 1) & (idx < 3 * int(mrope_section[1]))] = 1
    axis[(idx % 3 == 2) & (idx < 3 * int(mrope_section[2]))] = 2
    return axis


@register("function_und_gen_qkv_rms_norm_rope_cache")
class FunctionUndGenQkvRmsNormRopeCache(BaseApi):
    @staticmethod
    def rms_norm(x, weight, eps):
        """x: [T, H, D] float32; weight: [T, D] float32。"""
        inv = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        return x * inv * weight.unsqueeze(1)

    @staticmethod
    def rope(x, cos, sin):
        """x: [T, H, D] float32; cos/sin: [T, D/2] float32（三轴已合并）。"""
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        c, s = cos.unsqueeze(1), sin.unsqueeze(1)
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)

    def golden(self, und_qkv, und_weights_q, und_weights_k, cos_sin_cache, k_cache_ori, v_cache_ori,
               slot_mapping, positions, gen_qkv, gen_weights_q, gen_weights_k, cat_indices,
               num_heads_q, num_heads_k, num_heads_v, norm_eps, mrope_section):
        k_cache = copy.deepcopy(k_cache_ori)
        v_cache = copy.deepcopy(v_cache_ori)

        und_len, num_head, head_dim = und_qkv.shape
        gen_len = 0 if gen_qkv is None else gen_qkv.shape[0]
        assert num_head == num_heads_q + num_heads_k + num_heads_v, "N != Hq + Hk + Hv"
        total = und_len + gen_len

        src = torch.arange(total, dtype=torch.int64) if cat_indices is None else cat_indices.to(torch.int64)
        is_und = src < und_len

        rows = torch.empty(total, num_head, head_dim, dtype=torch.float32)
        if bool(is_und.any()):
            rows[is_und] = und_qkv[src[is_und]].float()
        if bool((~is_und).any()):
            rows[~is_und] = gen_qkv[src[~is_und] - und_len].float()

        q = rows[:, :num_heads_q, :]
        k = rows[:, num_heads_q:num_heads_q + num_heads_k, :]
        v = rows[:, num_heads_q + num_heads_k:, :]

        und_wq, und_wk = und_weights_q.float(), und_weights_k.float()
        gen_wq = und_wq if gen_weights_q is None else gen_weights_q.float()
        gen_wk = und_wk if gen_weights_k is None else gen_weights_k.float()
        sel = is_und.unsqueeze(-1)
        w_q = torch.where(sel, und_wq.unsqueeze(0), gen_wq.unsqueeze(0))
        w_k = torch.where(sel, und_wk.unsqueeze(0), gen_wk.unsqueeze(0))
        q = self.rms_norm(q, w_q, norm_eps)
        k = self.rms_norm(k, w_k, norm_eps)

        half = head_dim // 2
        pos = positions.to(torch.int64)
        if pos.dim() == 1:
            pos = pos.unsqueeze(0).expand(MROPE_AXIS_NUM, -1)
        axis = mrope_axis_map(head_dim, mrope_section)
        pos_sel = pos[axis].transpose(0, 1).contiguous()          # [T, half]
        col = torch.arange(half, dtype=torch.int64).unsqueeze(0)
        cos_sin = cos_sin_cache.float()
        cos = cos_sin[pos_sel, col]
        sin = cos_sin[pos_sel, col + half]
        q = self.rope(q, cos, sin)
        k = self.rope(k, cos, sin)

        # 分页写入：cache 为连续 BBND [Bn, Bs, N, D]
        block_size = k_cache.shape[1]
        slot = slot_mapping.to(torch.int64)
        k_view = k_cache.reshape(-1, k_cache.shape[2], head_dim)
        v_view = v_cache.reshape(-1, v_cache.shape[2], head_dim)
        k_view[slot] = k.to(k_cache.dtype)
        v_view[slot] = v.to(v_cache.dtype)
        Logger.info("golden done: total=%d, block_size=%d" % (total, block_size))

        return q.to(und_qkv.dtype), k_cache, v_cache


@register("function_pyaclnn_und_gen_qkv_rms_norm_rope_cache")
class FunctionPyAclnnUndGenQkvRmsNormRopeCache(AclnnBaseApi):
    def init_by_input_data(self, input_data):
        input_args, output_packages = super().init_by_input_data(input_data)
        # q / kCacheOut / vCacheOut：kCache/vCache 为原地更新，输出取回同一块内存
        return input_args, output_packages

    def after_call(self, output_packages):
        return super().after_call(output_packages)
