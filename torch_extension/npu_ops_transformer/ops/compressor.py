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


class CompressorOpBuilder(OpBuilder):
    def __init__(self):
        super(CompressorOpBuilder, self).__init__("npu_compressor")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/compressor.cpp']

    def schema(self):
        """PyTorch operator signature."""
        return "npu_compressor(Tensor x, Tensor wkv, Tensor wgate, Tensor(a!) state_cache, " \
               "Tensor ape, Tensor norm_weight, Tensor rope_sin, Tensor rope_cos, " \
               "int rope_head_dim=64, int cmp_ratio=4, " \
               "Tensor? state_block_table=None, Tensor? cu_seqlens=None, " \
               "Tensor? seqused=None, Tensor? start_pos=None, " \
               "int coff=1, float norm_eps=1e-6, int rotary_mode=1, int cache_mode=1) -> Tensor"

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_compressor_meta(x, wkv, wgate, state_cache, 
                                ape, norm_weight, rope_sin, rope_cos,
                                rope_head_dim=64, cmp_ratio=4,
                                state_block_table=None, cu_seqlens=None,
                                seqused=None, start_pos=None,
                                coff=1, norm_eps=1e-6, rotary_mode=1, cache_mode=1):
            if x.dim() == 3:
                B = x.size(0)
                S = x.size(1)
                H = x.size(2)
                Sr = (S + cmp_ratio - 1) // cmp_ratio
                cmp_kv_size = (B, Sr, H)
            else:
                T = x.size(0)
                H = x.size(1)
                Sr = (T + cmp_ratio - 1) // cmp_ratio
                cmp_kv_size = (Sr, H)
            
            return torch.empty(cmp_kv_size, dtype=x.dtype, device='meta')


compressor_op_builder = CompressorOpBuilder()
op_module = compressor_op_builder.load()


@impl(AS_LIBRARY, compressor_op_builder.name, "PrivateUse1")
def npu_compressor(x, wkv, wgate, state_cache, 
                   ape, norm_weight, rope_sin, rope_cos,
                   rope_head_dim=64, cmp_ratio=4,
                   state_block_table=None, cu_seqlens=None,
                   seqused=None, start_pos=None,
                   coff=1, norm_eps=1e-6, rotary_mode=1, cache_mode=1):
    """
    dispatcher implementation for NPU.
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    return op_module.npu_compressor(x, wkv, wgate, state_cache,
                                    ape, norm_weight, rope_sin, rope_cos,
                                    rope_head_dim, cmp_ratio,
                                    state_block_table, cu_seqlens,
                                    seqused, start_pos,
                                    coff, norm_eps, rotary_mode, cache_mode)