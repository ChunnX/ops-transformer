# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import List, Optional, Tuple

import torch
import torch_npu
from torch.library import impl

from cann_ops_transformer.op_builder.builder import AS_LIBRARY
from cann_ops_transformer.op_builder.builder import OpBuilder


OP_NAME = "qkv_rms_norm_rope_cache_with_k_scale"
FUNCTIONAL_OP_NAME = "qkv_rms_norm_rope_cache_with_k_scale_functional"
QKV_LAYOUT_TND = "TND"
QKV_LAYOUT_NTD = "NTD"


class QkvRmsNormRopeCacheWithKScaleOpBuilder(OpBuilder):
    def __init__(self):
        super(QkvRmsNormRopeCacheWithKScaleOpBuilder, self).__init__(OP_NAME)

    def sources(self):
        """Path to C++ source code."""
        return ["ops/csrc/qkv_rms_norm_rope_cache_with_k_scale.cpp"]

    def schema(self) -> List[str]:
        """PyTorch operator signatures."""
        return [
            "qkv_rms_norm_rope_cache_with_k_scale("
            "Tensor qkv, Tensor q_gamma, Tensor k_gamma, Tensor cos_sin, Tensor slot_mapping, "
            "Tensor(a!) k_cache, Tensor(b!) v_cache, Tensor(c!) k_scale_cache, "
            "Tensor cu_seq_len, Tensor actual_seq_lens, int[] head_nums, str qkv_layout, *, "
            "Tensor? rotation=None, Tensor? v_scale=None, "
            "float epsilon=0.000001) -> (Tensor, Tensor)",
            "qkv_rms_norm_rope_cache_with_k_scale_functional("
            "Tensor qkv, Tensor q_gamma, Tensor k_gamma, Tensor cos_sin, Tensor slot_mapping, "
            "Tensor k_cache, Tensor v_cache, Tensor k_scale_cache, "
            "Tensor cu_seq_len, Tensor actual_seq_lens, int[] head_nums, str qkv_layout, *, "
            "Tensor? rotation=None, Tensor? v_scale=None, "
            "float epsilon=0.000001) -> (Tensor, Tensor, Tensor, Tensor, Tensor)",
        ]

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @impl(AS_LIBRARY, OP_NAME, "Meta")
        def qkv_rms_norm_rope_cache_with_k_scale_meta(
            qkv,
            q_gamma,
            k_gamma,
            cos_sin,
            slot_mapping,
            k_cache,
            v_cache,
            k_scale_cache,
            cu_seq_len,
            actual_seq_lens,
            head_nums,
            qkv_layout,
            *,
            rotation=None,
            v_scale=None,
            epsilon=1e-6,
        ):
            return _qkv_rms_norm_rope_cache_with_k_scale_meta_outputs(
                qkv, head_nums, qkv_layout)

        @impl(AS_LIBRARY, FUNCTIONAL_OP_NAME, "Meta")
        def qkv_rms_norm_rope_cache_with_k_scale_functional_meta(
            qkv,
            q_gamma,
            k_gamma,
            cos_sin,
            slot_mapping,
            k_cache,
            v_cache,
            k_scale_cache,
            cu_seq_len,
            actual_seq_lens,
            head_nums,
            qkv_layout,
            *,
            rotation=None,
            v_scale=None,
            epsilon=1e-6,
        ):
            q_out, q_scale = _qkv_rms_norm_rope_cache_with_k_scale_meta_outputs(
                qkv, head_nums, qkv_layout)
            return q_out, q_scale, torch.empty_like(k_cache), torch.empty_like(v_cache), torch.empty_like(k_scale_cache)


def _qkv_rms_norm_rope_cache_with_k_scale_meta_outputs(
    qkv,
    head_nums,
    qkv_layout,
):
    if head_nums is None:
        raise RuntimeError("qkv_rms_norm_rope_cache_with_k_scale: head_nums must not be None")
    if len(head_nums) < 1:
        raise RuntimeError("qkv_rms_norm_rope_cache_with_k_scale: head_nums must contain q head num")

    n_q = head_nums[0]
    if n_q <= 0:
        raise RuntimeError("qkv_rms_norm_rope_cache_with_k_scale: head_nums[0] must be greater than 0")
    if qkv.dim() < 3:
        raise RuntimeError("qkv_rms_norm_rope_cache_with_k_scale: qkv must be at least 3D")
    if qkv_layout not in (QKV_LAYOUT_TND, QKV_LAYOUT_NTD):
        raise RuntimeError("qkv_rms_norm_rope_cache_with_k_scale: qkv_layout must be TND or NTD")

    is_ntd = qkv_layout == QKV_LAYOUT_NTD
    head_dim = 0 if is_ntd else 1
    token_dim = 1 if is_ntd else 0
    token_num = qkv.size(token_dim)
    head_size = qkv.size(2)
    if token_num <= 0 or head_size <= 0:
        raise RuntimeError("qkv_rms_norm_rope_cache_with_k_scale: qkv token and head dimensions must be positive")
    if qkv.size(head_dim) < n_q:
        raise RuntimeError(
            "qkv_rms_norm_rope_cache_with_k_scale: qkv head dimension must be greater than or equal to head_nums[0]")

    if is_ntd:
        q_out_shape = (n_q, token_num, head_size)
        q_scale_shape = (n_q, token_num)
    else:
        q_out_shape = (token_num, n_q, head_size)
        q_scale_shape = (token_num, n_q)

    # q_out dtype is fixed by the ACLNN contract: Q is dynamically quantized to FP8 E4M3FN.
    q_out = torch.empty(q_out_shape, dtype=torch.float8_e4m3fn, device="meta")
    q_scale = torch.empty(q_scale_shape, dtype=torch.float32, device="meta")
    return q_out, q_scale


qkv_rms_norm_rope_cache_with_k_scale_op_builder = QkvRmsNormRopeCacheWithKScaleOpBuilder()
op_module = qkv_rms_norm_rope_cache_with_k_scale_op_builder.load()


@impl(AS_LIBRARY, OP_NAME, "PrivateUse1")
def qkv_rms_norm_rope_cache_with_k_scale(
    qkv: torch.Tensor,
    q_gamma: torch.Tensor,
    k_gamma: torch.Tensor,
    cos_sin: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    cu_seq_len: torch.Tensor,
    actual_seq_lens: torch.Tensor,
    head_nums: List[int],
    qkv_layout: str,
    *,
    rotation: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run Q/K/V RMSNorm, RoPE, rotation matmul, FP8 quantization, and in-place KV cache update.
    """
    return op_module.qkv_rms_norm_rope_cache_with_k_scale(
        qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache, v_cache, k_scale_cache,
        cu_seq_len, actual_seq_lens, head_nums, qkv_layout, rotation, v_scale, epsilon)


@impl(AS_LIBRARY, FUNCTIONAL_OP_NAME, "PrivateUse1")
def qkv_rms_norm_rope_cache_with_k_scale_functional(
    qkv: torch.Tensor,
    q_gamma: torch.Tensor,
    k_gamma: torch.Tensor,
    cos_sin: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    cu_seq_len: torch.Tensor,
    actual_seq_lens: torch.Tensor,
    head_nums: List[int],
    qkv_layout: str,
    *,
    rotation: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Functional variant returning cloned cache outputs instead of mutating caller-visible caches.
    """
    return op_module.qkv_rms_norm_rope_cache_with_k_scale_functional(
        qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache, v_cache, k_scale_cache,
        cu_seq_len, actual_seq_lens, head_nums, qkv_layout, rotation, v_scale, epsilon)
