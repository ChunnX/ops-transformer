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

golden 直接委托给 tests/golden.py（本算子唯一 golden 实现），
避免同一套数学逻辑在 ST、torch 用例、设计文档之间各写一份而漂移。

NOTE: kernel 计算实现尚未补齐，本 ST 当前预期不通过；实现落地后即可直接使用。
"""
import os
import sys

from atk.common.log import Logger
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi

# tests/golden.py 位于本文件上两级目录
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from golden import golden_und_gen_qkv_rms_norm_rope_cache  # noqa: E402


@register("function_und_gen_qkv_rms_norm_rope_cache")
class FunctionUndGenQkvRmsNormRopeCache(BaseApi):
    def golden(self, und_qkv, und_weights_q, und_weights_k, cos_sin_cache, k_cache, v_cache,
               slot_mapping, positions, gen_qkv, gen_weights_q, gen_weights_k, cat_indices,
               num_heads_q, num_heads_k, num_heads_v, norm_eps, mrope_section):
        q, k_cache_out, v_cache_out = golden_und_gen_qkv_rms_norm_rope_cache(
            und_qkv, und_weights_q, und_weights_k, cos_sin_cache, k_cache, v_cache,
            slot_mapping, positions,
            gen_qkv=gen_qkv, gen_weights_q=gen_weights_q, gen_weights_k=gen_weights_k,
            cat_indices=cat_indices,
            num_heads_q=num_heads_q, num_heads_k=num_heads_k, num_heads_v=num_heads_v,
            norm_eps=norm_eps, mrope_section=list(mrope_section) if mrope_section else [])
        Logger.info("golden done: T=%d" % q.shape[0])
        return q, k_cache_out, v_cache_out


@register("function_pyaclnn_und_gen_qkv_rms_norm_rope_cache")
class FunctionPyAclnnUndGenQkvRmsNormRopeCache(AclnnBaseApi):
    def init_by_input_data(self, input_data):
        input_args, output_packages = super().init_by_input_data(input_data)
        # q / kCacheOut / vCacheOut：kCache/vCache 为原地更新，输出取回同一块内存
        return input_args, output_packages

    def after_call(self, output_packages):
        return super().after_call(output_packages)
