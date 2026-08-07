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
"""aclnnUndGenQkvRmsNormRopeCache 的 ATK executor。

golden 不在此处重写，直接复用 tests/assets/golden.py —— 那是本算子唯一的 golden
实现，torch 样例与 TTK 的 kernel/aclnn 两层插件都用它，多写一份必然漂移。

索引类输入（slotMapping / positions / catIndices）不能用 ATK 按 range 随机生成：
slotMapping 必须互不重复（重复槽位多核写同一 cache 行，结果不确定），positions 必须
落在 [0, max_pos)，catIndices 必须落在 [0, T)。因此两个类都在拿到输入后按 case id
重新生成这三者，取值规则与生成顺序保持一致，golden 与实调用才喂到同一份数据。
"""

import os
import sys

import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
from assets.golden import golden_und_gen_qkv_rms_norm_rope_cache  # noqa: E402

# aclnn 形参序：12 个张量 + 4 个标量属性 + mropeSection + qOut
IDX_UND_QKV = 0
IDX_K_CACHE = 4
IDX_SLOT_MAPPING = 6
IDX_POSITIONS = 7
IDX_CAT_INDICES = 11
IDX_Q_OUT = 17


def regen_indexes(input_data, case_id):
    """按 case id 确定性重建三个索引类输入，并写回 input_data.kwargs。"""
    kwargs = input_data.kwargs
    slot_mapping = kwargs['slotMapping']
    k_cache = kwargs['kCacheRef']
    total = int(slot_mapping.shape[0])
    capacity = int(k_cache.shape[0]) * int(k_cache.shape[1])
    max_pos = int(kwargs['cosSinCache'].shape[0])
    if capacity < total:
        raise RuntimeError(
            "KV Cache 容量不足：Bn*Bs=%d < T=%d，请调整用例的 kCacheRef/vCacheRef shape"
            % (capacity, total))

    gen = torch.Generator()
    gen.manual_seed(int(case_id))
    kwargs['slotMapping'] = torch.randperm(capacity, generator=gen)[:total].to(torch.int64)
    kwargs['catIndicesOptional'] = torch.randperm(total, generator=gen).to(torch.int64)
    kwargs['positions'] = torch.randint(0, max(max_pos, 1), (3, total),
                                        generator=gen, dtype=torch.int64)
    return kwargs


@register("function_pyaclnn_und_gen_qkv_rms_norm_rope_cache")
class FunctionPyaclnnUndGenQkvRmsNormRopeCache(AclnnBaseApi):
    def init_by_input_data(self, input_data):
        regen_indexes(input_data, self.task_result.case_config.id)
        input_args, output_packages = super().init_by_input_data(input_data)
        # kCacheRef / vCacheRef 原地更新，与 qOut 一并作为待比对输出
        output_packages[:] = [input_args[IDX_Q_OUT], input_args[IDX_K_CACHE], input_args[IDX_K_CACHE + 1]]
        return input_args, output_packages

    def after_call(self, output_packages):
        q_out, k_cache, v_cache = super().after_call(output_packages)
        return q_out, k_cache, v_cache


@register("function_und_gen_qkv_rms_norm_rope_cache")
class FunctionUndGenQkvRmsNormRopeCache(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        kwargs = regen_indexes(input_data, self.task_result.case_config.id)
        mrope_section = kwargs.get('mropeSection')
        mrope_section = list(mrope_section) if mrope_section is not None else None

        q_out, k_cache, v_cache = golden_und_gen_qkv_rms_norm_rope_cache(
            kwargs['undQkv'],
            kwargs['undWeightsQ'],
            kwargs['undWeightsK'],
            kwargs['cosSinCache'],
            kwargs['kCacheRef'],
            kwargs['vCacheRef'],
            kwargs['slotMapping'],
            kwargs['positions'],
            gen_qkv=kwargs['genQkvOptional'],
            gen_weights_q=kwargs['genWeightsQOptional'],
            gen_weights_k=kwargs['genWeightsKOptional'],
            cat_indices=kwargs['catIndicesOptional'],
            num_heads_q=int(kwargs['numHeadsQ']),
            num_heads_k=int(kwargs['numHeadsK']),
            num_heads_v=int(kwargs['numHeadsV']),
            norm_eps=float(kwargs['normEps']),
            mrope_section=mrope_section,
            inplace=False)
        return q_out, k_cache, v_cache
