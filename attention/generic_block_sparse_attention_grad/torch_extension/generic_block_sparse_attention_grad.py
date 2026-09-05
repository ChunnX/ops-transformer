# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PyTorch registration for GenericBlockSparseAttentionGrad."""

from enum import IntEnum
from typing import List, Optional, Tuple, Union

import torch
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library

GSAG_METADATA_SIZE = (
    16464  # legacy fallback; actual size is dynamic via calc_gsag_metadata_size
)
GSAG_TASK_LIST_OFFSET = 80  # 8 + 2 * 36
GSAG_TASK_ENTRY_SIZE = 4
GSAG_METADATA_OP_NAME = "generic_block_sparse_attention_grad_metadata"


class MaskMode(IntEnum):
    """mask_mode 枚举：对外 str/int/IntEnum 经由本枚举映射为传给算子侧的 int。"""

    RIGHT_DOWN_CAUSAL = 1


def _resolve_mask_mode(mask_mode: Union[str, int, MaskMode, None]) -> int:
    """对外 str/int/IntEnum/None mask_mode 统一为传给算子侧的 int。

    None 视为默认 RIGHT_DOWN_CAUSAL（当前唯一支持取值）。
    """
    if mask_mode is None:
        return int(MaskMode.RIGHT_DOWN_CAUSAL)
    if isinstance(mask_mode, str):
        try:
            return int(MaskMode[mask_mode.strip().upper()])
        except KeyError as exc:
            valid = ", ".join(m.name.lower() for m in MaskMode)
            raise ValueError(
                f"mask_mode should be one of [{valid}], but got {mask_mode!r}"
            ) from exc
    try:
        return int(MaskMode(mask_mode))
    except ValueError as exc:
        valid = ", ".join(f"{m.name}={m.value}" for m in MaskMode)
        raise ValueError(
            f"mask_mode should be one of [{valid}], but got {mask_mode!r}"
        ) from exc


def calc_gsag_metadata_size(batch_size: int, num_heads_q: int, num_j: int) -> int:
    """Required metadata int64 length: TASK_LIST_OFFSET + B * N1 * J * TASK_ENTRY_SIZE."""
    return (
        GSAG_TASK_LIST_OFFSET + batch_size * num_heads_q * num_j * GSAG_TASK_ENTRY_SIZE
    )


class GenericBlockSparseAttentionGradOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("generic_block_sparse_attention_grad", category="attention")

    def sources(self):
        """Path to C++ source code (packaged as csrc/attention/*.cpp)."""
        return ["csrc/attention/generic_block_sparse_attention_grad.cpp"]

    def schema(self):
        return [
            "generic_block_sparse_attention_grad_metadata("
            "Tensor rsvd_block_idx, Tensor rsvd_block_count, "
            "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_kv=None, "
            "Tensor? seqused_q=None, Tensor? seqused_kv=None, "
            "int max_seqlen_q=0, int max_seqlen_kv=0, int num_heads_q=0, int num_heads_kv=0, int head_dim=128, "
            "int[] block_shape=[1, 128], int is_packed_gqa=1, "
            'str layout_q="TND", str layout_kv="TND", '
            "int mask_mode=1, int softmax_precision=0, "
            "int win_left=-1, int win_right=-1) -> Tensor",
            "generic_block_sparse_attention_grad("
            "Tensor q, Tensor k, Tensor v, Tensor dout, Tensor attn_out, Tensor softmax_lse, "
            "Tensor rsvd_block_idx, Tensor rsvd_block_count, Tensor metadata, "
            "Tensor? attn_mask=None, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_kv=None, "
            "Tensor? seqused_q=None, Tensor? seqused_kv=None, "
            "int[] block_shape=[0], int is_packed_gqa=1, "
            'str layout_q="TND", str layout_kv="TND", '
            "float softmax_scale=1.0, int mask_mode=1, int softmax_precision=0, "
            "int win_left=-1, int win_right=-1) -> (Tensor, Tensor, Tensor)",
        ]

    def register_meta(self):
        @torch.library.register_fake("cann_ops_transformer::" + GSAG_METADATA_OP_NAME)
        def generic_block_sparse_attention_grad_metadata_meta(
            rsvd_block_idx: torch.Tensor,
            rsvd_block_count: torch.Tensor,
            cu_seqlens_q: Optional[torch.Tensor] = None,
            cu_seqlens_kv: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None,
            seqused_kv: Optional[torch.Tensor] = None,
            max_seqlen_q: int = 0,
            max_seqlen_kv: int = 0,
            num_heads_q: int = 0,
            num_heads_kv: int = 0,
            head_dim: int = 128,
            block_shape: Optional[List[int]] = None,
            is_packed_gqa: int = 1,
            layout_q: str = "TND",
            layout_kv: str = "TND",
            mask_mode: int = 1,
            softmax_precision: int = 0,
            win_left: int = -1,
            win_right: int = -1,
        ) -> torch.Tensor:
            del (
                rsvd_block_count,
                cu_seqlens_q,
                cu_seqlens_kv,
                seqused_q,
                seqused_kv,
                max_seqlen_q,
                num_heads_kv,
                head_dim,
                is_packed_gqa,
                layout_q,
                layout_kv,
                mask_mode,
                softmax_precision,
                win_left,
                win_right,
            )
            block_y = 128
            if block_shape is not None and len(block_shape) >= 2:
                block_y = int(block_shape[1])
            batch = int(rsvd_block_idx.shape[0])
            num_j = int(rsvd_block_idx.shape[2]) if rsvd_block_idx.dim() == 4 else 0
            if num_j <= 0 and max_seqlen_kv > 0:
                num_j = (int(max_seqlen_kv) + block_y - 1) // block_y
            meta_size = calc_gsag_metadata_size(batch, int(num_heads_q), max(num_j, 0))
            return torch.empty(
                (meta_size,), dtype=torch.int64, device=rsvd_block_idx.device
            )

        @torch.library.register_fake(
            "cann_ops_transformer::generic_block_sparse_attention_grad"
        )
        def generic_block_sparse_attention_grad_meta(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            dout: torch.Tensor,
            attn_out: torch.Tensor,
            softmax_lse: torch.Tensor,
            rsvd_block_idx: torch.Tensor,
            rsvd_block_count: torch.Tensor,
            metadata: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            cu_seqlens_q: Optional[torch.Tensor] = None,
            cu_seqlens_kv: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None,
            seqused_kv: Optional[torch.Tensor] = None,
            block_shape: Optional[List[int]] = None,
            is_packed_gqa: int = 1,
            layout_q: str = "TND",
            layout_kv: str = "TND",
            softmax_scale: float = 1.0,
            mask_mode: int = 1,
            softmax_precision: int = 0,
            win_left: int = -1,
            win_right: int = -1,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            del (
                dout,
                attn_out,
                softmax_lse,
                rsvd_block_idx,
                rsvd_block_count,
                metadata,
                attn_mask,
                cu_seqlens_q,
                cu_seqlens_kv,
                seqused_q,
                seqused_kv,
                block_shape,
                is_packed_gqa,
                layout_q,
                layout_kv,
                softmax_scale,
                mask_mode,
                softmax_precision,
                win_left,
                win_right,
            )
            return (
                torch.empty_like(q, device="meta"),
                torch.empty_like(k, device="meta"),
                torch.empty_like(v, device="meta"),
            )


generic_block_sparse_attention_grad_op_builder = (
    GenericBlockSparseAttentionGradOpBuilder()
)
generic_block_sparse_attention_grad_op_builder._ensure_initialized()


@impl(get_as_library(), GSAG_METADATA_OP_NAME, "PrivateUse1")
def generic_block_sparse_attention_grad_metadata(
    rsvd_block_idx: torch.Tensor,
    rsvd_block_count: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    max_seqlen_q: int = 0,
    max_seqlen_kv: int = 0,
    num_heads_q: int = 0,
    num_heads_kv: int = 0,
    head_dim: int = 128,
    block_shape: Optional[List[int]] = None,
    is_packed_gqa: int = 1,
    layout_q: str = "TND",
    layout_kv: str = "TND",
    mask_mode: Union[int, MaskMode, str] = MaskMode.RIGHT_DOWN_CAUSAL,
    softmax_precision: int = 0,
    win_left: int = -1,
    win_right: int = -1,
) -> torch.Tensor:
    if block_shape is None:
        block_shape = [1, 128]
    mask_mode_i = _resolve_mask_mode(mask_mode)
    op_module = generic_block_sparse_attention_grad_op_builder.load()
    return op_module.generic_block_sparse_attention_grad_metadata(
        rsvd_block_idx,
        rsvd_block_count,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        max_seqlen_q,
        max_seqlen_kv,
        num_heads_q,
        num_heads_kv,
        head_dim,
        block_shape,
        is_packed_gqa,
        layout_q,
        layout_kv,
        mask_mode_i,
        softmax_precision,
        win_left,
        win_right,
    )


@impl(get_as_library(), "generic_block_sparse_attention_grad", "PrivateUse1")
def generic_block_sparse_attention_grad(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    attn_out: torch.Tensor,
    softmax_lse: torch.Tensor,
    rsvd_block_idx: torch.Tensor,
    rsvd_block_count: torch.Tensor,
    metadata: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    is_packed_gqa: int = 1,
    layout_q: str = "TND",
    layout_kv: str = "TND",
    softmax_scale: float = 1.0,
    mask_mode: Union[int, MaskMode, str] = MaskMode.RIGHT_DOWN_CAUSAL,
    softmax_precision: int = 0,
    win_left: int = -1,
    win_right: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if block_shape is None:
        block_shape = [0]
    mask_mode_i = _resolve_mask_mode(mask_mode)
    op_module = generic_block_sparse_attention_grad_op_builder.load()
    return op_module.generic_block_sparse_attention_grad(
        q,
        k,
        v,
        dout,
        attn_out,
        softmax_lse,
        rsvd_block_idx,
        rsvd_block_count,
        metadata,
        attn_mask,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        block_shape,
        is_packed_gqa,
        layout_q,
        layout_kv,
        softmax_scale,
        mask_mode_i,
        softmax_precision,
        win_left,
        win_right,
    )


@torch.library.register_kernel(
    "cann_ops_transformer::generic_block_sparse_attention_grad", None
)
def generic_block_sparse_attention_grad_fallback(*args, **kwargs):
    return generic_block_sparse_attention_grad(*args, **kwargs)


@torch.library.register_kernel("cann_ops_transformer::" + GSAG_METADATA_OP_NAME, None)
def generic_block_sparse_attention_grad_metadata_fallback(*args, **kwargs):
    return generic_block_sparse_attention_grad_metadata(*args, **kwargs)


torch.compiler.allow_in_graph(generic_block_sparse_attention_grad_metadata)
torch.compiler.allow_in_graph(generic_block_sparse_attention_grad)
