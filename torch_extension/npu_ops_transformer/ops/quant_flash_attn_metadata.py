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
from typing import Optional

def _calculate_batch_size(batch_size, cu_seqlens_q, seqused_q):
    batchSize = 0
    if batch_size is not None:
        batchSize = batch_size
    elif cu_seqlens_q is not None and cu_seqlens_q.size(0) > 0:
        batchSize = cu_seqlens_q.size(0) - 1
    elif seqused_q is not None:
        batchSize = seqused_q.size(0)
    return batchSize

def _calculate_metadata_size(batch_size, num_heads_kv):
    metadataSize = ((36 + 72) * batch_size * num_heads_kv + 1) * 16
    alignedSize = ((metadataSize + 4095) // 4096) * 4096
    alignedSize = 4096
    return alignedSize


class QuantFlashAttnMetadataOpBuilder(OpBuilder):
    def __init__(self):
        super(QuantFlashAttnMetadataOpBuilder, self).__init__("npu_quant_flash_attn_metadata")

    def sources(self):
        return ['ops/csrc/quant_flash_attn_metadata.cpp']

    def schema(self) -> str:
        return "npu_quant_flash_attn_metadata(int num_heads_q, int num_heads_kv, int head_dim, int q_quant_mode, int k_quant_mode, int v_quant_mode, *, " \
            "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_kv=None, Tensor? seqused_q=None, Tensor? seqused_kv=None, " \
            "int? batch_size=None, int? max_seqlen_q=None, int? max_seqlen_kv=None, " \
            "int? q_dtype=None, int? k_dtype=None, int? v_dtype=None, " \
            "int? mask_mode=None, int? win_left=None, int? win_right=None, " \
            "str? layout_q=None, str? layout_kv=None, str? layout_out=None) -> Tensor"

    def register_meta(self):
        @torch.library.register_fake("npu_ops_transformer::" + self.name)
        def npu_quant_flash_attn_metadata_meta(
            num_heads_q, num_heads_kv, head_dim, q_quant_mode, k_quant_mode, v_quant_mode,
            cu_seqlens_q=None, cu_seqlens_kv=None, seqused_q=None, seqused_kv=None,
            batch_size: Optional[int] = None,
            max_seqlen_q: Optional[int] = None,
            max_seqlen_kv: Optional[int] = None,
            q_dtype: Optional[int] = None,
            k_dtype: Optional[int] = None,
            v_dtype: Optional[int] = None,
            mask_mode: Optional[int] = None,
            win_left: Optional[int] = None,
            win_right: Optional[int] = None,
            layout_q: Optional[str] = None,
            layout_kv: Optional[str] = None,
            layout_out: Optional[str] = None):
            metadataSize = _calculate_metadata_size(batch_size, num_heads_kv)
            return torch.empty((metadataSize,), dtype=torch.int32, device="meta")

quant_flash_attn_metadata_op_builder = QuantFlashAttnMetadataOpBuilder()
op_module = quant_flash_attn_metadata_op_builder.load()


@impl(AS_LIBRARY, quant_flash_attn_metadata_op_builder.name, "PrivateUse1")
def npu_quant_flash_attn_metadata(
    num_heads_q, num_heads_kv, head_dim, q_quant_mode, k_quant_mode, v_quant_mode,
    cu_seqlens_q=None, cu_seqlens_kv=None, seqused_q=None, seqused_kv=None,
    batch_size: Optional[int] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None,
    q_dtype: Optional[int] = None,
    k_dtype: Optional[int] = None,
    v_dtype: Optional[int] = None,
    mask_mode: Optional[int] = None,
    win_left: Optional[int] = None,
    win_right: Optional[int] = None,
    layout_q: Optional[str] = None,
    layout_kv: Optional[str] = None,
    layout_out: Optional[str] = None):

    batch_size = _calculate_batch_size(batch_size, cu_seqlens_q, seqused_q) if batch_size is None else batch_size
    max_seqlen_q = -1 if max_seqlen_q is None else max_seqlen_q
    max_seqlen_kv = -1 if max_seqlen_kv is None else max_seqlen_kv
    k_dtype = 0 if k_dtype is None else k_dtype
    v_dtype = 0 if v_dtype is None else v_dtype
    q_dtype = 0 if q_dtype is None else q_dtype
    mask_mode = 1 if mask_mode is None else mask_mode
    win_left = -1 if win_left is None else win_left
    win_right = -1 if win_right is None else win_right
    layout_q = "BSND" if layout_q is None else layout_q
    layout_kv = "BSND" if layout_kv is None else layout_kv
    layout_out = "BSND" if layout_out is None else layout_out

    metadataSize = _calculate_metadata_size(batch_size, num_heads_kv)
    output = torch.empty((metadataSize,), dtype=torch.int32, device="npu")

    return op_module.npu_quant_flash_attn_metadata(
        cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv,
        num_heads_q, num_heads_kv, head_dim,
        batch_size, max_seqlen_q, max_seqlen_kv,
        q_quant_mode, k_quant_mode, v_quant_mode,
        q_dtype, k_dtype, v_dtype,
        mask_mode, win_left, win_right,
        layout_q, layout_kv, layout_out, output)


@torch.library.register_kernel("npu_ops_transformer::npu_quant_flash_attn_metadata", None)
def npu_quant_flash_attn_metadata_fallback(
    num_heads_q, num_heads_kv, head_dim, q_quant_mode, k_quant_mode, v_quant_mode,
    cu_seqlens_q=None, cu_seqlens_kv=None, seqused_q=None, seqused_kv=None,
    batch_size: Optional[int] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None,
    q_dtype: Optional[int] = None,
    k_dtype: Optional[int] = None,
    v_dtype: Optional[int] = None,
    mask_mode: Optional[int] = None,
    win_left: Optional[int] = None,
    win_right: Optional[int] = None,
    layout_q: Optional[str] = None,
    layout_kv: Optional[str] = None,
    layout_out: Optional[str] = None):
    return npu_quant_flash_attn_metadata(
        num_heads_q, num_heads_kv, head_dim,
        q_quant_mode, k_quant_mode, v_quant_mode,
        cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv,
        batch_size, max_seqlen_q, max_seqlen_kv,
        q_dtype, k_dtype, v_dtype,
        mask_mode, win_left, win_right,
        layout_q, layout_kv, layout_out)