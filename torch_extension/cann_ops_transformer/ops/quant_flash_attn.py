# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional
import torch
import torch_npu
from torch.library import impl
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY

QFA_METADATA_OP_NAME = "quant_flash_attn_metadata"


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


class QuantFlashAttnOpBuilder(OpBuilder):
    def __init__(self):
        super(QuantFlashAttnOpBuilder, self).__init__("quant_flash_attn")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/quant_flash_attn.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        return [
            "quant_flash_attn_metadata(int num_heads_q, int num_heads_kv, int head_dim, int quant_mode, *, "
            "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_kv=None, Tensor? seqused_q=None, Tensor? seqused_kv=None, "
            "int? batch_size=None, int? max_seqlen_q=None, int? max_seqlen_kv=None, "
            "int? mask_mode=None, int? win_left=None, int? win_right=None, "
            "str? layout_q=None, str? layout_kv=None, str? layout_out=None) -> Tensor",

            "quant_flash_attn(Tensor q, Tensor k, Tensor v, "
            "Tensor q_descale, Tensor k_descale, Tensor v_descale, int quant_mode, "
            "Tensor? block_table=None, Tensor? p_scale=None, "
            "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_kv=None, "
            "Tensor? seqused_q=None, Tensor? seqused_kv=None, "
            "Tensor? sinks=None, Tensor? attn_mask=None, Tensor? metadata=None, "
            "float softmax_scale=1.0, int mask_mode=0, int win_left=-1, int win_right=-1, "
            "int max_seqlen_q=-1, int max_seqlen_kv=-1, "
            "str layout_q=\"BSND\", str layout_kv=\"BSND\", str layout_out=\"BSND\", "
            "int pa_block_size=0, "
            "bool return_softmax_lse=False) -> (Tensor, Tensor)"
        ]

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @torch.library.register_fake("cann_ops_transformer::" + QFA_METADATA_OP_NAME)
        def quant_flash_attn_metadata_meta(
            num_heads_q: int, num_heads_kv: int, head_dim: int, quant_mode: int,
            cu_seqlens_q: Optional[torch.Tensor] = None, cu_seqlens_kv: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None, seqused_kv: Optional[torch.Tensor] = None,
            batch_size: Optional[int] = None,
            max_seqlen_q: Optional[int] = None, max_seqlen_kv: Optional[int] = None,
            mask_mode: Optional[int] = None, win_left: Optional[int] = None, win_right: Optional[int] = None,
            layout_q: Optional[str] = None, layout_kv: Optional[str] = None, layout_out: Optional[str] = None
        ):
            eff_nkv = num_heads_q if quant_mode == 1 else num_heads_kv
            metadata_size = _calculate_metadata_size(batch_size, eff_nkv)
            return torch.empty((metadata_size,), dtype=torch.int32, device="npu")

        @impl(AS_LIBRARY, self.name, "Meta")
        def quant_flash_attn_meta(
            q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            q_descale: torch.Tensor, k_descale: torch.Tensor, v_descale: torch.Tensor,
            quant_mode: int,
            block_table: Optional[torch.Tensor] = None,
            p_scale: Optional[torch.Tensor] = None,
            cu_seqlens_q: Optional[torch.Tensor] = None, cu_seqlens_kv: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None, seqused_kv: Optional[torch.Tensor] = None,
            sinks: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            metadata: Optional[torch.Tensor] = None,
            softmax_scale: Optional[float] = 1.0,
            mask_mode: Optional[int] = 0, win_left: Optional[int] = -1, win_right: Optional[int] = -1,
            max_seqlen_q: Optional[int] = -1, max_seqlen_kv: Optional[int] = -1,
            layout_q: Optional[str] = "BSND", layout_kv: Optional[str] = "BSND", layout_out: Optional[str] = "BSND",
            pa_block_size: Optional[int] = 0,
            return_softmax_lse: Optional[bool] = False
        ):
            if layout_q == "TND":
                t_size = q.size(0)
                n_size = q.size(1)
                softmax_out_size = (n_size, t_size)
            elif layout_q == "BSND":
                b_size = q.size(0)
                s_size = q.size(1)
                n_size = q.size(2)
                softmax_out_size = (b_size, n_size, s_size)
            else:
                b_size = q.size(0)
                n_size = q.size(1)
                s_size = q.size(2)
                softmax_out_size = (b_size, n_size, s_size)

            if layout_kv == "PA_NZ":
                d_size = v.size(2) * v.size(4)
            elif layout_kv == "TND":
                d_size = v.size(2)
            else:
                d_size = v.size(3)

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
                torch.empty(attention_out_size, dtype=torch.bfloat16, device='meta'),
                torch.empty(softmax_out_size, dtype=torch.float32, device='meta')
            )


# Instantiate the builder
quant_flash_attn_op_builder = QuantFlashAttnOpBuilder()
op_module = quant_flash_attn_op_builder.load()


@impl(AS_LIBRARY, QFA_METADATA_OP_NAME, "PrivateUse1")
def quant_flash_attn_metadata(
        num_heads_q: int, num_heads_kv: int, head_dim: int,
        quant_mode: int,
        cu_seqlens_q: Optional[torch.Tensor] = None, cu_seqlens_kv: Optional[torch.Tensor] = None,
        seqused_q: Optional[torch.Tensor] = None, seqused_kv: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        max_seqlen_q: Optional[int] = None, max_seqlen_kv: Optional[int] = None,
        mask_mode: Optional[int] = None, win_left: Optional[int] = None, win_right: Optional[int] = None,
        layout_q: Optional[str] = None, layout_kv: Optional[str] = None, layout_out: Optional[str] = None
    ):
    """
    Dispatcher implementation: NPU.
    'PrivateUse1' is dispatch key for custom NPU backends.
    """
    batch_size = _calculate_batch_size(batch_size, cu_seqlens_q, seqused_q) if batch_size is None else batch_size
    max_seqlen_q = -1 if max_seqlen_q is None else max_seqlen_q
    max_seqlen_kv = -1 if max_seqlen_kv is None else max_seqlen_kv
    mask_mode = 1 if mask_mode is None else mask_mode
    win_left = -1 if win_left is None else win_left
    win_right = -1 if win_right is None else win_right
    layout_q = "BSND" if layout_q is None else layout_q
    layout_kv = "BSND" if layout_kv is None else layout_kv
    layout_out = "BSND" if layout_out is None else layout_out

    eff_nkv = num_heads_q if quant_mode == 1 else num_heads_kv
    metadata_size = _calculate_metadata_size(batch_size, eff_nkv)
    output = torch.empty((metadata_size,), dtype=torch.int32, device="npu")
 
    return op_module.quant_flash_attn_metadata(
        cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv,
        batch_size, max_seqlen_q, max_seqlen_kv,
        num_heads_q, num_heads_kv, head_dim,
        quant_mode,
        mask_mode, win_left, win_right,
        layout_q, layout_kv, layout_out, output)


@torch.library.register_kernel("cann_ops_transformer::" + QFA_METADATA_OP_NAME, None)
def quant_flash_attn_metadata_fallback(
        num_heads_q: int, num_heads_kv: int, head_dim: int,
        quant_mode: int,
        cu_seqlens_q: Optional[torch.Tensor] = None, cu_seqlens_kv: Optional[torch.Tensor] = None,
        seqused_q: Optional[torch.Tensor] = None, seqused_kv: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        max_seqlen_q: Optional[int] = None, max_seqlen_kv: Optional[int] = None,
        mask_mode: Optional[int] = None, win_left: Optional[int] = None, win_right: Optional[int] = None,
        layout_q: Optional[str] = None, layout_kv: Optional[str] = None, layout_out: Optional[str] = None
    ):
    # 处理所有 tensor 都为 None 的情况
    return quant_flash_attn_metadata(
        num_heads_q, num_heads_kv, head_dim,
        quant_mode,
        cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv,
        batch_size, max_seqlen_q, max_seqlen_kv,
        mask_mode, win_left, win_right,
        layout_q, layout_kv, layout_out)


@impl(AS_LIBRARY, quant_flash_attn_op_builder.name, "PrivateUse1")
def quant_flash_attn(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        q_descale: torch.Tensor, k_descale: torch.Tensor, v_descale: torch.Tensor,
        quant_mode: int,
        block_table: Optional[torch.Tensor] = None,
        p_scale: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None, cu_seqlens_kv: Optional[torch.Tensor] = None,
        seqused_q: Optional[torch.Tensor] = None, seqused_kv: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        metadata: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = 1.0,
        mask_mode: Optional[int] = 0, win_left: Optional[int] = -1, win_right: Optional[int] = -1,
        max_seqlen_q: Optional[int] = -1, max_seqlen_kv: Optional[int] = -1,
        layout_q: Optional[str] = "BSND", layout_kv: Optional[str] = "BSND", layout_out: Optional[str] = "BSND",
        pa_block_size: Optional[int] = 0,
        return_softmax_lse: Optional[bool] = False
    ):
    """
    dispatcher implementation for NPU.
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    return op_module.quant_flash_attn(
        q, k, v,
        q_descale, k_descale, v_descale,
        quant_mode,
        block_table,
        p_scale,
        cu_seqlens_q, cu_seqlens_kv,
        seqused_q, seqused_kv,
        sinks, attn_mask, metadata,
        softmax_scale, mask_mode, win_left, win_right,
        max_seqlen_q, max_seqlen_kv,
        layout_q, layout_kv, layout_out, pa_block_size,
        return_softmax_lse)