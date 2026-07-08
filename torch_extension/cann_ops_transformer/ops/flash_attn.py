# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional, Tuple
import torch
import torch_npu
from torch.library import custom_op
from cann_ops_transformer.op_builder.builder import OpBuilder

FA_METADATA_SIZE = 1024
FA_METADATA_OP_NAME = "flash_attn_metadata"


def _calculate_batch_size(batch_size, cu_seqlens_q, seqused_q):
    if batch_size is not None:
        return batch_size
    elif cu_seqlens_q is not None and cu_seqlens_q.size(0) > 0:
        return cu_seqlens_q.size(0) - 1
    elif seqused_q is not None:
        return seqused_q.size(0)
    return 0


def _calculate_metadata_size(batch_size, num_heads_kv):
    """计算 metadata tensor 的对齐后大小"""
    metadata_size = ((36 + 72) * batch_size * num_heads_kv + 1) * 16
    return ((metadata_size + 4095) // 4096) * 4096


class FlashAttenOpBuilder(OpBuilder):
    def __init__(self):
        super(FlashAttenOpBuilder, self).__init__("flash_attn")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/flash_attn.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        return None

    def register_meta(self):
        pass


flash_attn_op_builder = FlashAttenOpBuilder()


@custom_op("cann_ops_transformer.flash_attn_metadata", mutates_args=())
def flash_attn_metadata(
        num_heads_q: int, num_heads_kv: int, head_dim: int,
        cu_seqlens_q: Optional[torch.Tensor] = None, cu_seqlens_kv: Optional[torch.Tensor] = None,
        seqused_q: Optional[torch.Tensor] = None, seqused_kv: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        max_seqlen_q: Optional[int] = None, max_seqlen_kv: Optional[int] = None,
        mask_mode: Optional[int] = None, win_left: Optional[int] = None, win_right: Optional[int] = None,
        layout_q: Optional[str] = None, layout_kv: Optional[str] = None, layout_out: Optional[str] = None
) -> torch.Tensor:
    op_module = flash_attn_op_builder.load()
    batch_size = _calculate_batch_size(batch_size, cu_seqlens_q, seqused_q) if batch_size is None else batch_size
    max_seqlen_q = -1 if max_seqlen_q is None else max_seqlen_q
    max_seqlen_kv = -1 if max_seqlen_kv is None else max_seqlen_kv
    mask_mode = 1 if mask_mode is None else mask_mode
    win_left = -1 if win_left is None else win_left
    win_right = -1 if win_right is None else win_right
    layout_q = "BSND" if layout_q is None else layout_q
    layout_kv = "BSND" if layout_kv is None else layout_kv
    layout_out = "BSND" if layout_out is None else layout_out

    metadata_size = _calculate_metadata_size(batch_size, num_heads_kv)
    output = torch.empty((metadata_size,), dtype=torch.int32, device="npu")

    return op_module.flash_attn_metadata(cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv,
                                          num_heads_q, num_heads_kv, head_dim,
                                          batch_size, max_seqlen_q, max_seqlen_kv,
                                          mask_mode, win_left, win_right, layout_q, layout_kv, layout_out, output)


@flash_attn_metadata.register_fake
def flash_attn_metadata_fake(
        num_heads_q: int, num_heads_kv: int, head_dim: int,
        cu_seqlens_q: Optional[torch.Tensor] = None, cu_seqlens_kv: Optional[torch.Tensor] = None,
        seqused_q: Optional[torch.Tensor] = None, seqused_kv: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        max_seqlen_q: Optional[int] = None, max_seqlen_kv: Optional[int] = None,
        mask_mode: Optional[int] = None, win_left: Optional[int] = None, win_right: Optional[int] = None,
        layout_q: Optional[str] = None, layout_kv: Optional[str] = None, layout_out: Optional[str] = None
) -> torch.Tensor:
    metadata_size = _calculate_metadata_size(batch_size, num_heads_kv)
    return torch.empty((metadata_size,), dtype=torch.int32, device="meta")


@custom_op("cann_ops_transformer.flash_attn", mutates_args=())
def flash_attn(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        block_table: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None, cu_seqlens_kv: Optional[torch.Tensor] = None,
        seqused_q: Optional[torch.Tensor] = None, seqused_kv: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        metadata: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = 1.0,
        mask_mode: Optional[int] = 0, win_left: Optional[int] = -1, win_right: Optional[int] = -1,
        max_seqlen_q: Optional[int] = -1, max_seqlen_kv: Optional[int] = -1,
        layout_q: Optional[str] = "BSND", layout_kv: Optional[str] = "BSND", layout_out: Optional[str] = "BSND",
        return_softmax_lse: Optional[bool] = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    op_module = flash_attn_op_builder.load()
    return op_module.flash_attn(
        q, k, v,
        block_table,
        cu_seqlens_q, cu_seqlens_kv,
        seqused_q, seqused_kv,
        sinks,
        attn_mask,
        metadata,
        softmax_scale,
        mask_mode, win_left, win_right,
        max_seqlen_q, max_seqlen_kv,
        layout_q, layout_kv, layout_out,
        return_softmax_lse)


@flash_attn.register_fake
def flash_attn_fake(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        block_table: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None, cu_seqlens_kv: Optional[torch.Tensor] = None,
        seqused_q: Optional[torch.Tensor] = None, seqused_kv: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        metadata: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = 1.0,
        mask_mode: Optional[int] = 0, win_left: Optional[int] = -1, win_right: Optional[int] = -1,
        max_seqlen_q: Optional[int] = -1, max_seqlen_kv: Optional[int] = -1,
        layout_q: Optional[str] = "BSND", layout_kv: Optional[str] = "BSND", layout_out: Optional[str] = "BSND",
        return_softmax_lse: Optional[bool] = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    if layout_q == "TND":
        t_size = q.size(0)
        n_size = q.size(1)
        d_size = v.size(2)
        softmax_out_size = (n_size, t_size)
    elif layout_q == "BSND":
        b_size = q.size(0)
        s_size = q.size(1)
        n_size = q.size(2)
        d_size = v.size(3)
        softmax_out_size = (b_size, n_size, s_size)
    else:
        b_size = q.size(0)
        n_size = q.size(1)
        s_size = q.size(2)
        d_size = v.size(3)
        softmax_out_size = (b_size, n_size, s_size)

    if layout_out == "TND":
        torch._check(
            layout_q == "TND",
            lambda: f"When the layout of output is TND, the layout of query must be TND, but got {layout_q}",
        )
        attention_out_size = (t_size, n_size, d_size)
    elif layout_out == "BNSD":
        torch._check(
            layout_q == "BNSD",
            lambda: f"When the layout of output is BNSD, the layout of query must be BNSD, but got {layout_q}",
        )
        attention_out_size = (b_size, n_size, s_size, d_size)
    else:
        torch._check(
            layout_q != "TND",
            lambda: f"When the layout of output is BSND, the layout of query must be BNSD or BSND, but got {layout_q}",
        )
        attention_out_size = (b_size, s_size, n_size, d_size)

    return (
        torch.empty(attention_out_size, dtype=q.dtype, device='meta'),
        torch.empty(softmax_out_size, dtype=q.dtype, device='meta')
    )
