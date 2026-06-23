# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import torch
import torch_npu
from torch.library import impl
from npu_ops_transformer.op_builder.builder import OpBuilder
from npu_ops_transformer.op_builder.builder import AS_LIBRARY


class MlaPreprocessOpBuilder(OpBuilder):
    def __init__(self):
        super(MlaPreprocessOpBuilder, self).__init__("npu_mla_preprocess")

    def sources(self):
        return ['ops/csrc/mla_preprocess.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        return "npu_mla_preprocess(" \
            "Tensor input, Tensor gamma0, Tensor beta0, Tensor quant_scale0, Tensor quant_offset0, " \
            "Tensor wdqkv, Tensor de_scale0, Tensor bias0, Tensor gamma1, Tensor beta1, " \
            "Tensor quant_scale1, Tensor quant_offset1, Tensor wuq, Tensor de_scale1, Tensor bias1, " \
            "Tensor gamma2, Tensor cos, Tensor sin, Tensor wuk, Tensor kv_cache, Tensor slot_mapping, " \
            "Tensor? kv_cache_rope=None, Tensor? ctkv_scale=None, Tensor? q_nope_scale=None, " \
            "int wdq_dim=1536, int q_rope_dim=64, int k_rope_dim=64, float epsilon=1e-5, " \
            "int q_rotary_coeff=2, int k_rotary_coeff=2, bool transpose_wdq=True, bool transpose_wuq=True, bool transpose_wuk=True, " \
            "int cache_mode=1, int quant_mode=0, bool do_rms_norm=True, int wdkv_split_count=1" \
            ") -> (Tensor, Tensor, Tensor, Tensor)"

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_mla_preprocess_meta(input, gamma0, beta0, quant_scale0, quant_offset0,
                                      wdqkv, de_scale0, bias0, gamma1, beta1,
                                      quant_scale1, quant_offset1, wuq, de_scale1, bias1,
                                      gamma2, cos, sin, wuk, kv_cache, slot_mapping,
                                      kv_cache_rope=None, ctkv_scale=None, q_nope_scale=None,
                                      wdq_dim=1536, q_rope_dim=64, k_rope_dim=64, epsilon=1e-5,
                                      q_rotary_coeff=2, k_rotary_coeff=2,
                                      transpose_wdq=True, transpose_wuq=True, transpose_wuk=True,
                                      cache_mode=1, quant_mode=0, do_rms_norm=True, wdkv_split_count=1
                                      ):
            token_num = input.size(0)
            head_num = wuk.size(0)

            if cache_mode == 0:
                q_out_size = (token_num, head_num, 576)
            else:
                q_out_size = (token_num, head_num, 512)

            q_out_dtype = torch.int8 if cache_mode == 2 else input.dtype
            q_out = torch.empty(q_out_size, dtype=q_out_dtype, device='meta')

            # kv_cache is in-place
            kv_cache_out = torch.empty(kv_cache.size(), dtype=kv_cache.dtype, device='meta')

            if cache_mode == 0:
                q_rope_out = torch.empty((0,), dtype=input.dtype, device='meta')
                kr_cache_out = torch.empty((0,), dtype=input.dtype, device='meta')
            else:
                q_rope_out = torch.empty((token_num, head_num, 64), dtype=input.dtype, device='meta')
                if cache_mode == 1:
                    block_num = kv_cache.size(0)
                    block_size = kv_cache.size(1)
                    kr_cache_out = torch.empty((block_num, block_size, 1, 64), dtype=input.dtype, device='meta')
                else:
                    block_num = kv_cache.size(0)
                    block_size = kv_cache.size(2)
                    kr_cache_out = torch.empty((block_num, 4, block_size, 16), dtype=input.dtype, device='meta')

            return (q_out, kv_cache_out, q_rope_out, kr_cache_out)


mla_preprocess_op_builder = MlaPreprocessOpBuilder()
op_module = mla_preprocess_op_builder.load()


@impl(AS_LIBRARY, mla_preprocess_op_builder.name, "PrivateUse1")
def npu_mla_preprocess(input, gamma0, beta0, quant_scale0, quant_offset0,
                         wdqkv, de_scale0, bias0, gamma1, beta1,
                         quant_scale1, quant_offset1, wuq, de_scale1, bias1,
                         gamma2, cos, sin, wuk, kv_cache, slot_mapping,
                         kv_cache_rope=None, ctkv_scale=None, q_nope_scale=None,
                         wdq_dim=1536, q_rope_dim=64, k_rope_dim=64, epsilon=1e-5,
                         q_rotary_coeff=2, k_rotary_coeff=2,
                         transpose_wdq=True, transpose_wuq=True, transpose_wuk=True,
                         cache_mode=1, quant_mode=0, do_rms_norm=True, wdkv_split_count=1
                         ):
    return op_module.npu_mla_preprocess(
        input, gamma0, beta0, quant_scale0, quant_offset0,
        wdqkv, de_scale0, bias0, gamma1, beta1,
        quant_scale1, quant_offset1, wuq, de_scale1, bias1,
        gamma2, cos, sin, wuk, kv_cache, slot_mapping,
        kv_cache_rope, ctkv_scale, q_nope_scale,
        wdq_dim, q_rope_dim, k_rope_dim, epsilon,
        q_rotary_coeff, k_rotary_coeff,
        transpose_wdq, transpose_wuq, transpose_wuk,
        cache_mode, quant_mode, do_rms_norm, wdkv_split_count
    )
