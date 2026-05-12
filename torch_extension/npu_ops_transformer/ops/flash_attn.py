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


class FlashAttenOpBuilder(OpBuilder):
    def __init__(self):
        super(FlashAttenOpBuilder, self).__init__("npu_flash_attn")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/flash_attn.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        # TODO: 接口替换，基于新接口表格
        print("schema entrance ========================\n")
        return "npu_flash_attn(Tensor q, Tensor k, Tensor v," \
            "Tensor?block_table=None, Tensor?cu_seqlens_q=None," \
            "Tensor?cu_seqlens_kv=None, Tensor?seqused_q=None," \
            "Tensor?seqused_kv=None, Tensor?sinks=None, Tensor?metadata=None," \
            "float softmax_scale=1.0, int mask_mode=1, int win_left=-1, int win_right=-1," \
            "int max_seqlen_q=-1, int max_seqlen_kv=-1," \
            "str layout_q=\"BSND\", str layout_kv=\"BSND\", str layout_out=\"BSND\"," \
            "int return_softmax_lse=0, int deterministic=0)->(Tensor, Tensor)"

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @impl(AS_LIBRARY, self.name, "Meta") #TODO：同步接口
        def npu_flash_attn_meta(q, k, v, block_table=None, cu_seqlens_q=None,
                                cu_seqlens_kv=None, seqused_q=None,
                                seqused_kv=None, sinks=None, attn_mask=None, metadata=None,
                                softmax_scale=1.0, mask_mode=1, win_left=-1, win_right=-1,
                                max_seqlen_q=-1, max_seqlen_kv=-1,
                                layout_q="BSND", layout_kv="BSND", layout_out="BSND",
                                return_softmax_lse=0, deterministic=0):
            print("meta entrance ========================\n")
            if layout_q == "TND":
                tSize = q.size(0)
                nSize = q.size(1)
                dSize = v.size(2)
                softmaxOutSize = (nSize, tSize)
            elif layout_q == "BSND":
                bSize = q.size(0)
                sSize = q.size(1)
                nSize = q.size(2)
                dSize = v.size(3)
                softmaxOutSize = (bSize, nSize, sSize)
            else:
                bSize = q.size(0)
                nSize = q.size(1)
                sSize = q.size(2)
                dSize = v.size(3)
                softmaxOutSize = (bSize, nSize, sSize)

            if layout_out == "TND":
                torch._check(
                    layout_q == "TND",
                    lambda: "When the layout of output is TND, the layout of query  must be TND, but got " + str(layout_q),
                )
                attentionOutSize = (tSize, nSize, dSize)
            elif layout_out == "BNSD":
                torch._check(
                    layout_q == "BNSD",
                    lambda: "When the layout of output is BNSD, the layout of query  must be BNSD, but got " + str(layout_q),
                )
                attentionOutSize = (bSize, nSize, sSize, dSize)
            else:
                torch._check(
                    layout_q != "TND",
                    lambda: "When the layout of output is BSND, the layout of query  must be BNSD or BSND, but got " + str(layout_q),
                )
                attentionOutSize = (bSize, sSize, nSize, dSize)

            
            return attentionOutSize, softmaxOutSize


# Instantiate the builder
flash_attn_op_builder = FlashAttenOpBuilder()
op_module = flash_attn_op_builder.load()  # Compiles/loads the .so file


@impl(AS_LIBRARY, flash_attn_op_builder.name, "PrivateUse1")# TODO 接口
def npu_flash_attn(q, k, v, block_table=None, cu_seqlens_q=None,
                                cu_seqlens_kv=None, seqused_q=None,
                                seqused_kv=None, sinks=None, attn_mask=None, metadata=None,
                                softmax_scale=1.0, mask_mode=1, win_left=-1, win_right=-1,
                                max_seqlen_q=-1, max_seqlen_kv=-1,
                                layout_q="BSND", layout_kv="BSND", layout_out="BSND",
                                return_softmax_lse=0, deterministic=0):
    """
    dispatcher implementation for NPU.
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    print("flash_attn entrance ========================\n")
    return op_module.npu_flash_attn(q,  k, v, block_table, cu_seqlens_q,
                                    cu_seqlens_kv, seqused_q,
                                    seqused_kv, sinks, attn_mask, metadata,
                                    softmax_scale, mask_mode, win_left, win_right,
                                    max_seqlen_q, max_seqlen_kv,
                                    layout_q, layout_kv, layout_out,
                                    return_softmax_lse, deterministic)