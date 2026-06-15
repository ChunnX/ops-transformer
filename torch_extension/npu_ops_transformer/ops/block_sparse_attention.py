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
Python frontend + JIT builder for `aclnnBlockSparseAttentionV2`
(from `attention/block_sparse_attention/`).

Usage:
    import torch, torch_npu, npu_ops_transformer
    out, lse = npu_ops_transformer.ops.block_sparse_attention(
        query, key, value, block_sparse_mask, [128, 128],
        actual_seq_lengths=[S], actual_seq_lengths_kv=[S],
        num_heads=H, num_key_value_heads=H,
        scale_value=1.0/math.sqrt(D), softmax_lse_flag=True,
    )

`aclnnBlockSparseAttentionV2` adds `q/k/v_dequant_scale` over V1 — FP8-only
(FLOAT8_E4M3FN) FP32 reverse-quant factors. They default to None and are passed
through as nullptr on the FP16/BF16 path, so existing callers are unaffected.

Prerequisite:
    The kernel + aclnn library must already be built and installed into the
    Ascend custom-transformer vendor path:

        bash build.sh --make_clean -j96 --pkg --soc=ascend910b \\
                      --ops=block_sparse_attention
        ./build/cann-ops-transformer-custom_linux-aarch64.run

    `ACLNN_CMD` (in the C++ wrapper) dlopens `libcust_opapi.so` at runtime to
    resolve `aclnnBlockSparseAttentionV2` — no link-time dependency on the
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
            "str q_input_layout=\"BNSD\", "
            "str kv_input_layout=\"BNSD\", "
            "int mask_type=0, "
            "float scale_value=1.0, "
            "int inner_precise=0, "
            "int block_size=0, "
            "int pre_tokens=2147483647, "
            "int next_tokens=2147483647, "
            "bool softmax_lse_flag=False, "
            "Tensor? q_dequant_scale=None, "
            "Tensor? k_dequant_scale=None, "
            "Tensor? v_dequant_scale=None"
            ") -> (Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def _meta(query, key, value, block_sparse_mask, block_shape,
                  actual_seq_lengths, actual_seq_lengths_kv,
                  num_heads, num_key_value_heads,
                  q_input_layout, kv_input_layout,
                  mask_type, scale_value, inner_precise, block_size,
                  pre_tokens, next_tokens, softmax_lse_flag,
                  q_dequant_scale=None, k_dequant_scale=None,
                  v_dequant_scale=None):
            # Mirror the C++ wrapper's input constraints (block_sparse_attention.cpp)
            # so graph / FakeTensor tracing fails at the same boundary as real
            # execution instead of only erroring deep inside the kernel at run time.
            torch._check(
                block_size == 0,
                lambda: "block_size must be 0 — paged KV cache is not yet supported by the kernel",
            )
            torch._check(
                mask_type == 0,
                lambda: "mask_type must be 0 — fine-grained attention masks are not yet supported by the kernel",
            )
            # FP8 dequant is all-or-none: the kernel needs q, k and v scales together.
            n_dequant = sum(
                t is not None
                for t in (q_dequant_scale, k_dequant_scale, v_dequant_scale)
            )
            torch._check(
                n_dequant == 0 or n_dequant == 3,
                lambda: "q_dequant_scale, k_dequant_scale and v_dequant_scale must be "
                        "provided all together (FP8 dequant path) or all omitted "
                        "(FP16/BF16 path)",
            )
            attn = torch.empty_like(query)
            if softmax_lse_flag:
                # Mirror the C++ wrapper's constraint (block_sparse_attention.cpp:108)
                # so the meta / FakeTensor path fails the same way as real execution
                # instead of silently inferring a wrong LSE shape under TND.
                torch._check(
                    q_input_layout == "BNSD",
                    lambda: "softmax_lse_flag=True only supported for BNSD Q layout",
                )
                # LSE_DIM_D=1 in the kernel's infershape → [b, n, s, 1] float32.
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
                           block_size=0,
                           pre_tokens=2147483647,
                           next_tokens=2147483647,
                           softmax_lse_flag=False,
                           q_dequant_scale=None,
                           k_dequant_scale=None,
                           v_dequant_scale=None):
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
        inner_precise: 0 = fp32 softmax (high precision), 1 = fp16, 4 = mixed.
            NOTE: Ascend 950PR/950DT only supports 4; A2/A3 only support 0/1.
        block_size: paged-attention block size. Must be 0 — kernel does not
            yet support paged cache (see block_sparse_attention_tiling.cpp:528).
        pre_tokens, next_tokens: 2147483647 = no causal window (default).
        softmax_lse_flag: if True, return [B, N, S, 1] float32 LSE; else
            an empty {0} placeholder.
        q_dequant_scale, k_dequant_scale, v_dequant_scale: FP8-only (FLOAT8_E4M3FN
            inputs) FP32 4-D reverse-quant factors for aclnnBlockSparseAttentionV2.
            Leave as None (default) for FP16/BF16 inputs — they are then passed
            through as nullptr. Shapes (when provided): q is
            [B, N, ceil(maxQ/128), 1]; k/v are [B, kvN, ceil(maxKV/{256|512}), 1].

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
        q_dequant_scale, k_dequant_scale, v_dequant_scale,
    )
