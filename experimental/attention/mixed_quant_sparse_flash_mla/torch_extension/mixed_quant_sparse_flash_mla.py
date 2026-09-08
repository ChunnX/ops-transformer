# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


def mixed_quant_sparse_flash_mla(
    q,
    *,
    win_kv=None,
    cmp_kv=None,
    win_sparse_indices=None,
    cmp_sparse_indices=None,
    win_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    seqused_q=None,
    seqused_win_kv=None,
    seqused_cmp_kv=None,
    win_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    quant_mode,
    softmax_scale=None,
    layout_q="TND",
    layout_kv="PA_BBND",
    return_softmax_lse=False,
):
    """Allocation-only placeholder. DSL kernel integration is pending.

    Both topk lengths are optional: None means all sparse index columns.
    The returned buffers contain no attention result until a kernel is connected.
    """
    if win_kv is None:
        raise ValueError("win_kv is required by the current operator contract")
    if layout_q != "TND" or layout_kv != "PA_BBND":
        raise ValueError("only TND / PA_BBND is supported")
    if q.dim() != 3 or q.shape[1:] != (64, 512) or q.dtype != torch.bfloat16:
        raise ValueError("q must be BF16 [T1, 64, 512]")
    if quant_mode != 1:
        raise ValueError("only quant_mode=1 is supported")
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    lse_shape = (1, q.shape[0], 64) if return_softmax_lse else (0,)
    lse = torch.empty(lse_shape, dtype=torch.float32, device=q.device)
    # Future DSL call receives out=out and lse=lse along with these inputs.
    return out, lse


class MixedQuantSparseFlashMlaOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("mixed_quant_sparse_flash_mla", category="attention")

    def sources(self):
        # Python allocation stub; no C++ module or kernel is loaded.
        return []

    def schema(self):
        return (
            "mixed_quant_sparse_flash_mla(Tensor q, *, "
            "Tensor? win_kv=None, "
            "Tensor? cmp_kv=None, "
            "Tensor? win_sparse_indices=None, "
            "Tensor? cmp_sparse_indices=None, "
            "Tensor? win_block_table=None, "
            "Tensor? cmp_block_table=None, "
            "Tensor? cu_seqlens_q=None, "
            "Tensor? seqused_q=None, "
            "Tensor? seqused_win_kv=None, "
            "Tensor? seqused_cmp_kv=None, "
            "Tensor? win_topk_length=None, "
            "Tensor? cmp_topk_length=None, "
            "Tensor? sinks=None, "
            "Tensor? metadata=None, "
            "int quant_mode, float? softmax_scale=None, "
            'str layout_q="TND", str layout_kv="PA_BBND", '
            "bool return_softmax_lse=False) -> (Tensor, Tensor)"
        )

    def register_meta(self):
        impl(get_as_library(), self.name, "Meta")(mixed_quant_sparse_flash_mla)


mixed_quant_sparse_flash_mla_op_builder = MixedQuantSparseFlashMlaOpBuilder()
mixed_quant_sparse_flash_mla_op_builder._ensure_initialized()
impl(get_as_library(), mixed_quant_sparse_flash_mla_op_builder.name,
     "PrivateUse1")(mixed_quant_sparse_flash_mla)


def mixed_quant_sparse_flash_mla_metadata(
    *,
    cu_seqlens_q=None,
    seqused_q=None,
    seqused_win_kv=None,
    seqused_cmp_kv=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_win_kv=None,
    max_seqlen_cmp_kv=None,
    win_topk_length=None,
    cmp_topk_length=None,
    num_heads_q,
    num_heads_kv,
    head_dim,
    win_topk=None,
    cmp_topk=None,
    quant_mode,
    layout_q="TND",
    layout_kv="PA_BBND",
    has_win_kv=True,
    has_cmp_kv=True,
):
    """Allocation-only metadata placeholder; DSL integration is pending."""
    tensors = (cu_seqlens_q, seqused_q, seqused_win_kv, seqused_cmp_kv,
               win_topk_length, cmp_topk_length)
    source = next((tensor for tensor in tensors if tensor is not None), None)
    device = source.device if source is not None else torch.device("npu")
    # Meta needs only output capacity, which depends on the target NPU core count.
    target = None if device.type == "meta" else device
    blocks = int(torch.npu.get_device_properties(target).cube_core_num)
    return torch.empty((2 * blocks,), dtype=torch.int32, device=device)


class MixedQuantSparseFlashMlaMetadataOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("mixed_quant_sparse_flash_mla_metadata", category="attention")

    def sources(self):
        return []

    def schema(self):
        return (
            "mixed_quant_sparse_flash_mla_metadata(*, "
            "Tensor? cu_seqlens_q=None, Tensor? seqused_q=None, "
            "Tensor? seqused_win_kv=None, Tensor? seqused_cmp_kv=None, "
            "int? batch_size=None, int? max_seqlen_q=None, "
            "int? max_seqlen_win_kv=None, int? max_seqlen_cmp_kv=None, "
            "Tensor? win_topk_length=None, Tensor? cmp_topk_length=None, "
            "int num_heads_q, int num_heads_kv, int head_dim, "
            "int? win_topk=None, int? cmp_topk=None, int quant_mode, "
            'str layout_q="TND", str layout_kv="PA_BBND", '
            "bool has_win_kv=True, bool has_cmp_kv=True) -> Tensor"
        )

    def register_meta(self):
        impl(get_as_library(), self.name, "Meta")(mixed_quant_sparse_flash_mla_metadata)


mixed_quant_sparse_flash_mla_metadata_op_builder = MixedQuantSparseFlashMlaMetadataOpBuilder()
mixed_quant_sparse_flash_mla_metadata_op_builder._ensure_initialized()
# Composite fallback also handles calls with no Tensor inputs (only attributes).
impl(get_as_library(), mixed_quant_sparse_flash_mla_metadata_op_builder.name,
     "CompositeExplicitAutograd")(mixed_quant_sparse_flash_mla_metadata)
