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
            "Tensor input, Tensor gamma0, Tensor beta0, Tensor wdqkv, Tensor de_scale0, " \
            "Tensor gamma1, Tensor beta1, Tensor wuq, Tensor de_scale1, " \
            "Tensor gamma2, Tensor cos, Tensor sin, Tensor wuk, " \
            "Tensor kv_cache, Tensor kv_cache_rope, Tensor slot_mapping, *, " \
            "Tensor? quant_scale0=None, Tensor? quant_offset0=None, Tensor? bias0=None, " \
            "Tensor? quant_scale1=None, Tensor? quant_offset1=None, Tensor? bias1=None, " \
            "Tensor? ctkv_scale=None, Tensor? q_nope_scale=None, " \
            "str cache_mode=krope_ctkv, str quant_mode=per_token_quant_symm" \
            ") -> (Tensor, Tensor, Tensor, Tensor)"

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def mla_preprocess_meta(
            input, gamma0, beta0, wdqkv, de_scale0,
            gamma1, beta1, wuq, de_scale1,
            gamma2, cos, sin, wuk,
            kv_cache, kv_cache_rope, slot_mapping, *,
            quant_scale0=None, quant_offset0=None, bias0=None,
            quant_scale1=None, quant_offset1=None, bias1=None,
            ctkv_scale=None, q_nope_scale=None,
            cache_mode="krope_ctkv", quant_mode="per_token_quant_symm"
        ):
            token_num = input.size(0)
            head_num = wuk.size(0)

            _CACHE_MODE = {"kcache": 0, "krope_ctkv": 1, "int8_nzcache": 2, "nzcache": 3}
            cm = _CACHE_MODE[cache_mode]

            if cm == 0:
                q_out_size = (token_num, head_num, 576)
            else:
                q_out_size = (token_num, head_num, 512)

            q_out_dtype = torch.int8 if cm == 2 else input.dtype
            q_out = torch.empty(q_out_size, dtype=q_out_dtype, device='meta')

            kv_cache_out = torch.empty(kv_cache.size(), dtype=kv_cache.dtype, device='meta')

            if cm == 0:
                q_rope_out = torch.empty((0,), dtype=input.dtype, device='meta')
                kr_cache_out = torch.empty((0,), dtype=input.dtype, device='meta')
            else:
                q_rope_out = torch.empty((token_num, head_num, 64), dtype=input.dtype, device='meta')
                if cm == 1:
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
def mla_preprocess(
    input, gamma0, beta0, wdqkv, de_scale0,
    gamma1, beta1, wuq, de_scale1,
    gamma2, cos, sin, wuk,
    kv_cache, kv_cache_rope, slot_mapping, *,
    quant_scale0=None, quant_offset0=None, bias0=None,
    quant_scale1=None, quant_offset1=None, bias1=None,
    ctkv_scale=None, q_nope_scale=None,
    cache_mode="krope_ctkv", quant_mode="per_token_quant_symm"
):
    """MLA Preprocess 前处理算子，封装 aclnnMlaPreprocess。

    Args:
        input (Tensor): 输入 hidden states，shape [token_num, hidden_size]，
            hidden_size 通常为 7168，dtype 支持 float16/bfloat16。
        gamma0 (Tensor): 第一层 RMS Norm 的 gamma 参数，shape [hidden_size]。
        beta0 (Tensor): 第一层 RMS Norm 的 beta 参数，shape [hidden_size]。
        wdqkv (Tensor): 第一层 MatMul 权重，FRACTAL_NZ 格式，shape [q_lora_dim + key_total_dim, hidden_size]。
        de_scale0 (Tensor): 第一层 MatMul 反量化 scale，bf16 时为 float32，fp16 时为 int64。
        gamma1 (Tensor): 第二层 RMS Norm 的 gamma 参数，shape [q_lora_dim]。
        beta1 (Tensor): 第二层 RMS Norm 的 beta 参数，shape [q_lora_dim]。
        wuq (Tensor): 第二层 MatMul 权重，FRACTAL_NZ 格式，shape [head_num * (nope_dim + rope_dim), q_lora_dim]。
        de_scale1 (Tensor): 第二层 MatMul 反量化 scale，bf16 时为 float32，fp16 时为 int64。
        gamma2 (Tensor): 第三层 RMS Norm 的 gamma 参数，shape [512]。
        cos (Tensor): RoPE 的余弦参数，shape [token_num, k_rope_dim]。
        sin (Tensor): RoPE 的正弦参数，shape [token_num, k_rope_dim]。
        wuk (Tensor): Einsum 权重，shape [head_num, nope_dim, 512]。
        kv_cache (Tensor): KV Cache 缓存张量，in-place 更新，shape 因 cache_mode 而异。
        kv_cache_rope (Tensor): RoPE 缓存张量，in-place 更新。cache_mode='kcache' 时传空 Tensor。
        slot_mapping (Tensor): token 写入位置索引，dtype int32，shape [token_num]。

    Keyword Args:
        quant_scale0 (Tensor, optional): 第一层 RMS Norm 量化 scale，shape [1]。
        quant_offset0 (Tensor, optional): 第一层 RMS Norm 量化 offset，dtype int8，shape [1]。
        bias0 (Tensor, optional): 第一层 MatMul bias，dtype int32，shape [q_lora_dim + key_total_dim]。
        quant_scale1 (Tensor, optional): 第二层 RMS Norm 量化 scale，shape [1]。
        quant_offset1 (Tensor, optional): 第二层 RMS Norm 量化 offset，dtype int8，shape [1]。
        bias1 (Tensor, optional): 第二层 MatMul bias，dtype int32。
        ctkv_scale (Tensor, optional): KV Cache 量化 scale，仅 cache_mode='int8_nzcache' 时需要，shape [1]。
        q_nope_scale (Tensor, optional): Q 输出量化 scale，仅 cache_mode='int8_nzcache' 时需要，shape [head_num]。
        cache_mode (str): 缓存模式，默认 'krope_ctkv'。
            - 'kcache'：key+rope 合并存储
            - 'krope_ctkv'：key/rope 分离，ND 格式
            - 'int8_nzcache'：量化 int8，NZ 格式
            - 'nzcache'：NZ 格式，不量化
        quant_mode (str): 量化模式，默认 'per_token_quant_symm'。
            - 'per_tensor_quant_asymm'：per_tensor 静态非对称量化
            - 'per_token_quant_symm'：per_token 动态对称量化
            - 'per_token_quant_asymm'：per_token 动态非对称量化
            - 'no_quant'：不量化

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]:
            q_out、kv_cache_out、q_rope_out、kr_cache_out。
            kv_cache_out、kr_cache_out 均为 in-place 输出。
    """
    _CACHE_MODE_MAP = {
        "kcache": 0,
        "krope_ctkv": 1,
        "int8_nzcache": 2,
        "nzcache": 3,
    }
    _QUANT_MODE_MAP = {
        "per_tensor_quant_asymm": 0,
        "per_token_quant_symm": 1,
        "per_token_quant_asymm": 2,
        "no_quant": 3,
    }
    cache_mode_int = _CACHE_MODE_MAP[cache_mode]
    quant_mode_int = _QUANT_MODE_MAP[quant_mode]

    return op_module.mla_preprocess(
        input, gamma0, beta0,
        quant_scale0 if quant_scale0 is not None else torch.empty(0, dtype=input.dtype, device=input.device),
        quant_offset0 if quant_offset0 is not None else torch.empty(0, dtype=torch.int8, device=input.device),
        wdqkv,
        de_scale0,
        bias0 if bias0 is not None else torch.empty(0, dtype=torch.int32, device=input.device),
        gamma1, beta1,
        quant_scale1 if quant_scale1 is not None else torch.empty(0, dtype=input.dtype, device=input.device),
        quant_offset1 if quant_offset1 is not None else torch.empty(0, dtype=torch.int8, device=input.device),
        wuq,
        de_scale1,
        bias1 if bias1 is not None else torch.empty(0, dtype=torch.int32, device=input.device),
        gamma2, cos, sin, wuk,
        kv_cache, kv_cache_rope, slot_mapping,
        ctkv_scale if ctkv_scale is not None else torch.empty(0, dtype=input.dtype, device=input.device),
        q_nope_scale if q_nope_scale is not None else torch.empty(0, dtype=input.dtype, device=input.device),
        1536, 64, 64,           # wdq_dim, q_rope_dim, k_rope_dim (预留)
        1e-5,                    # epsilon (预留)
        2, 2,                    # q_rotary_coeff, k_rotary_coeff (预留)
        True, True, True,        # transpose_wdq/wuq/wuk (预留)
        cache_mode_int,
        quant_mode_int,
        True,  # do_rms_norm (预留)
        1      # wdkv_split_count (预留)
    )
