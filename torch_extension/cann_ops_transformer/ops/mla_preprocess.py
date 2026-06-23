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
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY


class MlaPreprocessOpBuilder(OpBuilder):
    def __init__(self):
        super(MlaPreprocessOpBuilder, self).__init__("mla_preprocess")

    def sources(self):
        return ['ops/csrc/mla_preprocess.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        return "mla_preprocess(" \
            "Tensor input, Tensor gamma0, Tensor beta0, Tensor quant_scale0, Tensor quant_offset0, " \
            "Tensor wdqkv, Tensor de_scale0, Tensor bias0, Tensor gamma1, Tensor beta1, " \
            "Tensor quant_scale1, Tensor quant_offset1, Tensor wuq, Tensor de_scale1, Tensor bias1, " \
            "Tensor gamma2, Tensor cos, Tensor sin, Tensor wuk, Tensor kv_cache, Tensor slot_mapping, " \
            "Tensor? kv_cache_rope=None, Tensor? ctkv_scale=None, Tensor? q_nope_scale=None, " \
            "int wdq_dim=1536, int q_rope_dim=64, int k_rope_dim=64, float epsilon=1e-5, " \
            "int q_rotary_coeff=2, int k_rotary_coeff=2, bool transpose_wdq=True, " \
            "bool transpose_wuq=True, bool transpose_wuk=True, int cache_mode=1, " \
            "int quant_mode=0, bool do_rms_norm=True, int wdkv_split_count=1" \
            ") -> (Tensor, Tensor, Tensor, Tensor)"

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def mla_preprocess_meta(hidden_states, gamma0, beta0, quant_scale0, quant_offset0,
                                      wdqkv, de_scale0, bias0, gamma1, beta1,
                                      quant_scale1, quant_offset1, wuq, de_scale1, bias1,
                                      gamma2, cos, sin, wuk, kv_cache, slot_mapping,
                                      kv_cache_rope=None, ctkv_scale=None, q_nope_scale=None,
                                      wdq_dim=1536, q_rope_dim=64, k_rope_dim=64, epsilon=1e-5,
                                      q_rotary_coeff=2, k_rotary_coeff=2,
                                      transpose_wdq=True, transpose_wuq=True, transpose_wuk=True,
                                      cache_mode=1, quant_mode=0, do_rms_norm=True, wdkv_split_count=1
                                      ):
            token_num = hidden_states.size(0)
            head_num = wuk.size(0)

            if cache_mode == 0:
                q_out_size = (token_num, head_num, 576)
            else:
                q_out_size = (token_num, head_num, 512)

            q_out_dtype = torch.int8 if cache_mode == 2 else hidden_states.dtype
            q_out = torch.empty(q_out_size, dtype=q_out_dtype, device='meta')

            # kv_cache is in-place
            kv_cache_out = torch.empty(kv_cache.size(), dtype=kv_cache.dtype, device='meta')

            if cache_mode == 0:
                q_rope_out = torch.empty((0,), dtype=hidden_states.dtype, device='meta')
                kr_cache_out = torch.empty((0,), dtype=hidden_states.dtype, device='meta')
            else:
                q_rope_out = torch.empty((token_num, head_num, 64), dtype=hidden_states.dtype, device='meta')
                if cache_mode == 1:
                    block_num = kv_cache.size(0)
                    block_size = kv_cache.size(1)
                    kr_cache_out = torch.empty((block_num, block_size, 1, 64), dtype=hidden_states.dtype, device='meta')
                else:
                    block_num = kv_cache.size(0)
                    block_size = kv_cache.size(2)
                    kr_cache_out = torch.empty((block_num, 4, block_size, 16), dtype=hidden_states.dtype, device='meta')

            return (q_out, kv_cache_out, q_rope_out, kr_cache_out)


mla_preprocess_op_builder = MlaPreprocessOpBuilder()
op_module = mla_preprocess_op_builder.load()


@impl(AS_LIBRARY, mla_preprocess_op_builder.name, "PrivateUse1")
def mla_preprocess(hidden_states, gamma0, beta0, quant_scale0, quant_offset0,
                         wdqkv, de_scale0, bias0, gamma1, beta1,
                         quant_scale1, quant_offset1, wuq, de_scale1, bias1,
                         gamma2, cos, sin, wuk, kv_cache, slot_mapping,
                         kv_cache_rope=None, ctkv_scale=None, q_nope_scale=None,
                         wdq_dim=1536, q_rope_dim=64, k_rope_dim=64, epsilon=1e-5,
                         q_rotary_coeff=2, k_rotary_coeff=2,
                         transpose_wdq=True, transpose_wuq=True, transpose_wuk=True,
                         cache_mode=1, quant_mode=0, do_rms_norm=True, wdkv_split_count=1
                         ):
    """MLA Preprocess 前处理算子，封装 aclnnMlaPreprocess。

    Args:
        hidden_states (Tensor): 输入 hidden states，shape [token_num, hidden_size]，
            hidden_size 通常为 7168，dtype 支持 float16/bfloat16。
        gamma0 (Tensor): 第一层 RMS Norm 的 gamma 参数，shape [hidden_size]。
        beta0 (Tensor): 第一层 RMS Norm 的 beta 参数，shape [hidden_size]。
        quant_scale0 (Tensor): 第一层 RMS Norm 的量化 scale，shape [1]。
        quant_offset0 (Tensor): 第一层 RMS Norm 的量化 offset，dtype int8，shape [1]。
        wdqkv (Tensor): 第一层 MatMul 权重，FRACTAL_NZ 格式，shape [q_lora_dim + key_total_dim, hidden_size]。
        de_scale0 (Tensor): 第一层 MatMul 反量化 scale，bf16 输入时为 float32，fp16 输入时为 int64。
        bias0 (Tensor): 第一层 MatMul 的 bias，dtype int32，quant_mode 为 1/3 时可传空 Tensor。
        gamma1 (Tensor): 第二层 RMS Norm 的 gamma 参数，shape [q_lora_dim]。
        beta1 (Tensor): 第二层 RMS Norm 的 beta 参数，shape [q_lora_dim]。
        quant_scale1 (Tensor): 第二层 RMS Norm 的量化 scale，shape [1]。
        quant_offset1 (Tensor): 第二层 RMS Norm 的量化 offset，dtype int8，shape [1]。
        wuq (Tensor): 第二层 MatMul 权重，FRACTAL_NZ 格式，shape [head_num * (nope_dim + rope_dim), q_lora_dim]。
        de_scale1 (Tensor): 第二层 MatMul 反量化 scale，bf16 输入时为 float32，fp16 输入时为 int64。
        bias1 (Tensor): 第二层 MatMul 的 bias，dtype int32，quant_mode 为 1/3 时可传空 Tensor。
        gamma2 (Tensor): 第三层 RMS Norm 的 gamma 参数，shape [512]。
        cos (Tensor): RoPE 的余弦参数，shape [token_num, k_rope_dim]，k_rope_dim 通常为 64。
        sin (Tensor): RoPE 的正弦参数，shape [token_num, k_rope_dim]。
        wuk (Tensor): Einsum 权重，shape [head_num, nope_dim, 512]。
        kv_cache (Tensor): KV Cache 缓存张量，in-place 更新。shape 因 cache_mode 而异。
        slot_mapping (Tensor): token 在 KV Cache 中的写入位置索引，dtype int32，shape [token_num]。
        kv_cache_rope (Tensor, optional): RoPE 缓存张量，in-place 更新。cache_mode=0 时不需传入。
        ctkv_scale (Tensor, optional): KV Cache 量化 scale，仅 cache_mode=2 时使用，shape [1]。
        q_nope_scale (Tensor, optional): Q 输出量化 scale，仅 cache_mode=2 时使用，shape [head_num]。
        wdq_dim (int): Q LoRA 维度，目前仅支持 1536。默认 1536。
        q_rope_dim (int): Q RoPE 维度，目前仅支持 64。默认 64。
        k_rope_dim (int): K RoPE 维度，目前仅支持 64。默认 64。
        epsilon (float): RMS Norm 防除零 epsilon。默认 1e-5。
        q_rotary_coeff (int): Q 旋转系数，目前仅支持 2。默认 2。
        k_rotary_coeff (int): K 旋转系数，目前仅支持 2。默认 2。
        transpose_wdq (bool): wdqkv 是否转置，目前仅支持 True。默认 True。
        transpose_wuq (bool): wuq 是否转置，目前仅支持 True。默认 True。
        transpose_wuk (bool): wuk 是否转置，目前仅支持 True。默认 True。
        cache_mode (int): 缓存模式，0~3。默认 1。
        quant_mode (int): 量化模式，当前仅支持 0。默认 0。
        do_rms_norm (bool): 是否执行 RMS Norm，目前仅支持 True。默认 True。
        wdkv_split_count (int): wdqkv 分拆次数，目前仅支持 1。默认 1。

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]:
            q_out、kv_cache_out、q_rope_out、kr_cache_out。
            kv_cache_out 与 kv_cache 共享存储（in-place）；
            kr_cache_out 与 kv_cache_rope 共享存储（in-place）；
            cache_mode=0 时 q_rope_out 和 kr_cache_out 为空张量 shape [0]。
    """
    return op_module.mla_preprocess(
        hidden_states, gamma0, beta0, quant_scale0, quant_offset0,
        wdqkv, de_scale0, bias0, gamma1, beta1,
        quant_scale1, quant_offset1, wuq, de_scale1, bias1,
        gamma2, cos, sin, wuk, kv_cache, slot_mapping,
        kv_cache_rope, ctkv_scale, q_nope_scale,
        wdq_dim, q_rope_dim, k_rope_dim, epsilon,
        q_rotary_coeff, k_rotary_coeff,
        transpose_wdq, transpose_wuq, transpose_wuk,
        cache_mode, quant_mode, do_rms_norm, wdkv_split_count
    )
