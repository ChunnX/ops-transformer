# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import os
from typing import List, Optional, Tuple

import torch
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library


def _output_shapes(
    query: torch.Tensor, num_heads: int, input_layout: str, softmax_lse_flag: bool
) -> Tuple[tuple, tuple]:
    layouts = (
        "BSH",
        "BSND",
        "BNSD",
        "TND",
        "BSH_NBSD",
        "BSND_NBSD",
        "BNSD_NBSD",
        "TND_NTD",
    )
    torch._check(
        input_layout in layouts, lambda: f"Unsupported input_layout: {input_layout}"
    )
    torch._check(0 < num_heads <= 128, lambda: "num_heads must be in [1, 128]")
    layout_q = input_layout.split("_")[0]
    rank = 3 if layout_q in ("BSH", "TND") else 4
    torch._check(
        query.dim() == rank, lambda: f"query must have rank {rank} for {input_layout}"
    )
    if layout_q == "BSH":
        batch, seq, hidden = query.shape
        torch._check(
            hidden == num_heads * 512,
            lambda: "query hidden size must be num_heads * 512",
        )
    else:
        torch._check(query.size(-1) == 512, lambda: "query head dimension must be 512")
        if layout_q == "BSND":
            batch, seq, heads, _ = query.shape
        elif layout_q == "BNSD":
            batch, heads, seq, _ = query.shape
        else:
            seq, heads, _ = query.shape
        torch._check(
            heads == num_heads, lambda: "query head count must match num_heads"
        )

    if input_layout.endswith("_NBSD"):
        output = (num_heads, batch, seq, 512)
    elif input_layout == "TND_NTD":
        output = (num_heads, seq, 512)
    else:
        output = tuple(query.shape)
    if not softmax_lse_flag:
        lse = (0,)
    elif layout_q == "TND":
        lse = (seq, num_heads, 1)
    else:
        lse = (batch, num_heads, seq, 1)
    return output, lse


class FlashMlaOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("flash_mla", category="attention")

    def sources(self) -> List[str]:
        return ["csrc/attention/flash_mla.cpp"]

    def include_paths(self) -> List[str]:
        # aclnn_common.h needs the SDK ACL enums before torch_npu's older headers.
        return [os.path.join(self.cann_path, "include")] + super().include_paths()

    def schema(self) -> str:
        return (
            "flash_mla(Tensor query, Tensor[] key, Tensor[] value, Tensor query_rope, "
            "Tensor key_rope, Tensor metadata, int num_heads, *, Tensor? atten_mask=None, "
            "Tensor? actual_seq_lengths_q=None, Tensor? actual_seq_lengths_kv=None, "
            "Tensor? block_table=None, Tensor? dequant_scale_query=None, "
            "Tensor? key_antiquant_scale=None, Tensor? value_antiquant_scale=None, "
            "float scale_value=1.0, int pre_tokens=2147483647, int next_tokens=2147483647, "
            'str input_layout="BSH", int num_key_value_heads=1, int sparse_mode=0, '
            "int inner_precise=0, int block_size=0, bool softmax_lse_flag=False, "
            "int key_antiquant_mode=0, int value_antiquant_mode=0, int query_quant_mode=0) "
            "-> (Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def flash_mla_meta(
            query,
            key,
            value,
            query_rope,
            key_rope,
            metadata,
            num_heads,
            *,
            atten_mask=None,
            actual_seq_lengths_q=None,
            actual_seq_lengths_kv=None,
            block_table=None,
            dequant_scale_query=None,
            key_antiquant_scale=None,
            value_antiquant_scale=None,
            scale_value=1.0,
            pre_tokens=2147483647,
            next_tokens=2147483647,
            input_layout="BSH",
            num_key_value_heads=1,
            sparse_mode=0,
            inner_precise=0,
            block_size=0,
            softmax_lse_flag=False,
            key_antiquant_mode=0,
            value_antiquant_mode=0,
            query_quant_mode=0,
        ):
            torch._check(
                len(key) == 1 and len(value) == 1,
                lambda: "key and value must each contain exactly one tensor",
            )
            torch._check(
                query.dtype in (torch.float16, torch.bfloat16, torch.int8),
                lambda: "query dtype must be float16, bfloat16 or int8",
            )
            torch._check(
                metadata.dtype == torch.int32 and metadata.dim() == 1,
                lambda: "metadata must be a one-dimensional int32 tensor",
            )
            torch._check(
                metadata.numel() == 480,
                lambda: "metadata must contain 480 int32 elements",
            )
            torch._check(
                metadata.is_contiguous() and metadata.storage_offset() == 0,
                lambda: "metadata must be contiguous with zero storage offset",
            )
            output, lse = _output_shapes(
                query, num_heads, input_layout, softmax_lse_flag
            )
            dtype = torch.bfloat16 if query.dtype == torch.int8 else query.dtype
            return query.new_empty(output, dtype=dtype), query.new_empty(
                lse, dtype=torch.float32
            )


_flash_mla_op_builder = FlashMlaOpBuilder()
_flash_mla_op_builder._ensure_initialized()


@impl(get_as_library(), "flash_mla", "PrivateUse1")
def _flash_mla(
    query,
    key,
    value,
    query_rope,
    key_rope,
    metadata,
    num_heads,
    *,
    atten_mask=None,
    actual_seq_lengths_q=None,
    actual_seq_lengths_kv=None,
    block_table=None,
    dequant_scale_query=None,
    key_antiquant_scale=None,
    value_antiquant_scale=None,
    scale_value=1.0,
    pre_tokens=2147483647,
    next_tokens=2147483647,
    input_layout="BSH",
    num_key_value_heads=1,
    sparse_mode=0,
    inner_precise=0,
    block_size=0,
    softmax_lse_flag=False,
    key_antiquant_mode=0,
    value_antiquant_mode=0,
    query_quant_mode=0,
):
    return _flash_mla_op_builder.load().flash_mla(
        query,
        key,
        value,
        query_rope,
        key_rope,
        metadata,
        num_heads,
        atten_mask,
        actual_seq_lengths_q,
        actual_seq_lengths_kv,
        block_table,
        dequant_scale_query,
        key_antiquant_scale,
        value_antiquant_scale,
        scale_value,
        pre_tokens,
        next_tokens,
        input_layout,
        num_key_value_heads,
        sparse_mode,
        inner_precise,
        block_size,
        softmax_lse_flag,
        key_antiquant_mode,
        value_antiquant_mode,
        query_quant_mode,
    )


def flash_mla(
    query: torch.Tensor,
    key: List[torch.Tensor],
    value: List[torch.Tensor],
    query_rope: torch.Tensor,
    key_rope: torch.Tensor,
    metadata: torch.Tensor,
    num_heads: int,
    *,
    atten_mask: Optional[torch.Tensor] = None,
    actual_seq_lengths_q: Optional[torch.Tensor] = None,
    actual_seq_lengths_kv: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    dequant_scale_query: Optional[torch.Tensor] = None,
    key_antiquant_scale: Optional[torch.Tensor] = None,
    value_antiquant_scale: Optional[torch.Tensor] = None,
    scale_value: float = 1.0,
    pre_tokens: int = 2147483647,
    next_tokens: int = 2147483647,
    input_layout: str = "BSH",
    num_key_value_heads: int = 1,
    sparse_mode: int = 0,
    inner_precise: int = 0,
    block_size: int = 0,
    softmax_lse_flag: bool = False,
    key_antiquant_mode: int = 0,
    value_antiquant_mode: int = 0,
    query_quant_mode: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """执行 Ascend 910B/C split-RoPE MLA 前向计算。

    Args:
        query: FP16/BF16/INT8 NPU 张量，head dim 为 512，布局由 input_layout 指定。
        key, value: 各含一个张量的列表；dtype 与 query 一致，有效 KV heads 为 1。
        query_rope, key_rope: RoPE 张量，head dim 为 64；INT8 时为 BF16，否则与 query 同 dtype。
        metadata: flash_mla_metadata 生成的连续、零偏移、64B 对齐 NPU INT32[480]。
        num_heads: Q 头数，范围 [1, 128]。
        atten_mask: 可选 BOOL/INT8/UINT8 mask，shape 由 sparse_mode 决定。
        actual_seq_lengths_q, actual_seq_lengths_kv: 可选一维连续 NPU INT64 实长；
            TND Q/dense KV 为累计长度（不含起始 0），PA KV 为各 batch 的绝对长度。
        block_table: 可选 INT32[batch, max_blocks] 分页表。
        dequant_scale_query, key_antiquant_scale, value_antiquant_scale: INT8 必选 FP32 scale。
        scale_value: attention score 缩放系数。
        pre_tokens, next_tokens: 左右窗口，与 metadata 生成参数保持一致。
        input_layout: BSH/BSND/BNSD/TND，或前三者的 _NBSD 后缀、TND_NTD。
        num_key_value_heads: 默认 1；非量化非空输入仅支持 1，INT8 且 num_heads=1 时允许 0。
        sparse_mode: 非量化支持 0/3/4/9，INT8 支持 0/3/9。
        inner_precise: 非量化为 0–3，INT8 为 0/1。
        block_size: 非 PA 为 0；PA 为不大于 1024 的正 16 倍数，INT8 为 128。
        softmax_lse_flag: 是否返回有效 FP32 LSE，默认 False。
        key_antiquant_mode, value_antiquant_mode: 当前必须为 0。
        query_quant_mode: 非量化为 0，INT8 为 3。

    Returns:
        attention_out 与 softmax_lse。attention_out 由 input_layout 决定布局，
        INT8 输入输出 BF16；LSE 为 [T,N,1] 或 [B,N,S,1]，关闭时为 FP32[0]。

    metadata 必须由同一 stream 上相同配置、seq 版本和有效核数的 producer 生成。
    本接口只提供前向 eager 调用；完整约束见 docs/torchapi_flash_mla.md。
    """
    return torch.ops.cann_ops_transformer.flash_mla(
        query,
        key,
        value,
        query_rope,
        key_rope,
        metadata,
        num_heads,
        atten_mask=atten_mask,
        actual_seq_lengths_q=actual_seq_lengths_q,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        block_table=block_table,
        dequant_scale_query=dequant_scale_query,
        key_antiquant_scale=key_antiquant_scale,
        value_antiquant_scale=value_antiquant_scale,
        scale_value=scale_value,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        input_layout=input_layout,
        num_key_value_heads=num_key_value_heads,
        sparse_mode=sparse_mode,
        inner_precise=inner_precise,
        block_size=block_size,
        softmax_lse_flag=softmax_lse_flag,
        key_antiquant_mode=key_antiquant_mode,
        value_antiquant_mode=value_antiquant_mode,
        query_quant_mode=query_quant_mode,
    )
