# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Python frontend + JIT builder for `aclnnBlockSparseAttention`
(from `attention/block_sparse_attention/`).

Usage:
    import torch, torch_npu, npu_ops_transformer
    out, lse = npu_ops_transformer.ops.block_sparse_attention(
        query, key, value, block_sparse_mask, [128, 128],
        actual_seq_lengths=[S], actual_seq_lengths_kv=[S],
        num_heads=H, num_key_value_heads=H,
        scale_value=1.0/math.sqrt(D), softmax_lse_flag=True,
    )

Prerequisite:
    The kernel + aclnn library must already be built and installed into the
    Ascend custom-transformer vendor path:

        bash build.sh --make_clean -j96 --pkg --soc=ascend910b \\
                      --ops=block_sparse_attention
        ./build/cann-ops-transformer-custom_linux-aarch64.run

    `ACLNN_CMD` (in the C++ wrapper) dlopens `libcust_opapi.so` at runtime to
    resolve `aclnnBlockSparseAttention` — no link-time dependency on the
    vendor library.
"""

import torch
import torch_npu  # noqa: F401  (registers the NPU backend)
from torch.library import impl
from npu_ops_transformer.op_builder.builder import OpBuilder
from npu_ops_transformer.op_builder.builder import AS_LIBRARY


class BlockSparseAttentionOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("block_sparse_attention")

    def sources(self):
        return ['ops/csrc/block_sparse_attention.cpp']

    def schema(self) -> str:
        return (
            "block_sparse_attention("
            "Tensor query, Tensor key, Tensor value, "
            "Tensor block_sparse_mask, "
            "int[] block_shape, "
            "int[] actual_seq_lengths, "
            "int[] actual_seq_lengths_kv, "
            "int num_heads, "
            "int num_key_value_heads, "
            "str q_input_layout, "
            "str kv_input_layout, "
            "int mask_type, "
            "float scale_value, "
            "int inner_precise, "
            "int block_size, "
            "int pre_tokens, "
            "int next_tokens, "
            "bool softmax_lse_flag"
            ") -> (Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def _meta(query, key, value, block_sparse_mask, block_shape,
                  actual_seq_lengths, actual_seq_lengths_kv,
                  num_heads, num_key_value_heads,
                  q_input_layout, kv_input_layout,
                  mask_type, scale_value, inner_precise, block_size,
                  pre_tokens, next_tokens, softmax_lse_flag):
            attn = torch.empty_like(query)
            if softmax_lse_flag:
                # LSE_DIM_D=1 in the kernel's infershape → [b, n, s, 1] float32.
                # Assumes BNSD layout (validated by the C++ wrapper).
                b = query.size(0)
                n = num_heads
                s = query.size(2)
                lse = query.new_empty((b, n, s, 1), dtype=torch.float32)
            else:
                lse = query.new_empty((0,), dtype=torch.float32)
            return attn, lse


_op_builder = BlockSparseAttentionOpBuilder()
_op_module = _op_builder.load()      # JIT-compiles + loads block_sparse_attention.so


@impl(AS_LIBRARY, _op_builder.name, "PrivateUse1")
def block_sparse_attention(query,
                           key,
                           value,
                           block_sparse_mask,
                           block_shape,
                           actual_seq_lengths,
                           actual_seq_lengths_kv,
                           num_heads,
                           num_key_value_heads,
                           q_input_layout="BNSD",
                           kv_input_layout="BNSD",
                           mask_type=0,
                           scale_value=1.0,
                           inner_precise=0,
                           block_size=128,
                           pre_tokens=2147483647,
                           next_tokens=2147483647,
                           softmax_lse_flag=False):
    """Block-sparse attention with optional softmax_lse output.

    Args:
        query, key, value: BF16 / FP16 tensors. BNSD: [B, N, S, D]; TND: [T, N, D].
        block_sparse_mask: int8 [B, N, ceil(S_q/block_x), ceil(S_kv/block_y)],
            1 = block kept, 0 = block skipped.
        block_shape: [block_x, block_y], typically [128, 128].
        actual_seq_lengths, actual_seq_lengths_kv: list[int] per batch.
        num_heads, num_key_value_heads: Q heads, KV heads (GQA = same).
        q_input_layout, kv_input_layout: "BNSD" or "TND".
        mask_type: 0 = no mask (default).
        scale_value: softmax scale (typically 1/sqrt(D)).
        inner_precise: 0 = fp32 softmax (high precision), 1 = fp16.
        block_size: paged-attention block size (unused; 128).
        pre_tokens, next_tokens: 2147483647 = no causal window (default).
        softmax_lse_flag: if True, return [B, N, S, 1] float32 LSE; else
            an empty {0} placeholder.

    Returns:
        (attention_out, softmax_lse)
    """
    return _op_module.npu_block_sparse_attention(
        query, key, value, block_sparse_mask,
        block_shape, actual_seq_lengths, actual_seq_lengths_kv,
        num_heads, num_key_value_heads,
        q_input_layout, kv_input_layout,
        mask_type, scale_value, inner_precise, block_size,
        pre_tokens, next_tokens, softmax_lse_flag,
    )
