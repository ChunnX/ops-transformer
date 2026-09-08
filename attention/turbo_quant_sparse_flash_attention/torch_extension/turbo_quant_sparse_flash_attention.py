# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Optional, Tuple

import torch
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library

INT64_MAX = 9223372036854775807
QUERY_HEAD_DIM = 576
LATENT_HEAD_DIM = 512
KV_SLOT_BYTES = 386


def _check(condition, message):
    torch._check(condition, lambda: message)


def _check_inputs(
    query,
    key,
    value,
    sparse_indices,
    key_scale,
    value_scale,
    block_table,
    query_lengths,
    kv_lengths,
    layout_query,
    layout_kv,
    key_quant_mode,
    value_quant_mode,
    sparse_block_size,
    sparse_mode,
    pre_tokens,
    next_tokens,
    attention_mode,
    quant_scale_repo_mode,
    tile_size,
    rope_head_dim,
):
    """只校验张量元数据，不读取设备数据，也不创建 NPU 张量。"""
    _check(
        layout_query == "TND",
        f"layout_query[{layout_query}] is not supported, only TND",
    )
    _check(
        layout_kv in ("PA_BSND", "PA_BNBD"),
        f"layout_kv[{layout_kv}] is not supported, only PA_BSND and PA_BNBD",
    )
    _check(
        query.dtype in (torch.float16, torch.bfloat16),
        "query must be float16 or bfloat16",
    )
    _check(query.dim() == 3, "query must have shape [T, N, 576]")
    _check(
        query.shape[1] in (1, 2, 4, 8, 16, 32, 64, 128),
        "query head count must be a power of two in [1, 128]",
    )
    _check(query.shape[2] == QUERY_HEAD_DIM, "query head dimension must be 576")
    _check(key.dim() == 4 and value.dim() == 4, "key and value must have rank 4")
    _check(
        key.dtype == torch.int8 and value.dtype == torch.int8,
        "key and value must be int8",
    )
    for axis in range(4):
        _check(key.shape[axis] == value.shape[axis], "key and value shapes must match")
    head_axis, block_axis = (1, 2) if layout_kv == "PA_BNBD" else (2, 1)
    _check(key.shape[head_axis] == 1, "key must contain exactly one KV head")
    _check(key.shape[0] > 0, "KV physical block count must be positive")
    _check(key.shape[3] == KV_SLOT_BYTES, "KV slot width must be 386")
    block_size = key.shape[block_axis]
    _check(block_size >= 16, "KV block size must be at least 16")
    _check(block_size <= 1024, "KV block size must be at most 1024")
    _check(block_size % 16 == 0, "KV block size must be 16-aligned")
    for cache in (key, value):
        page_bytes = 1
        for axis in range(3, 0, -1):
            _check(
                cache.shape[axis] == 1 or cache.stride(axis) == page_bytes,
                "KV cache only supports non-contiguous tensor on axis 0",
            )
            page_bytes *= cache.shape[axis]
        _check(cache.stride(0) >= 0, "KV cache stride must be nonnegative")
        _check(
            cache.shape[0] <= 1 or cache.stride(0) >= page_bytes,
            "KV cache pages must not overlap",
        )
        _check(
            (cache.shape[0] - 1) * cache.stride(0) + page_bytes <= INT64_MAX,
            "KV cache address span exceeds INT64_MAX",
        )
    _check(sparse_indices.dim() == 3, "sparse_indices must have shape [T, 1, K]")
    _check(
        sparse_indices.shape[0] == query.shape[0], "sparse_indices must match query T"
    )
    _check(sparse_indices.shape[1] == 1, "sparse_indices must have one KV head")
    _check(
        query_lengths.dim() == 1 and kv_lengths.dim() == 1,
        "sequence lengths must be rank 1",
    )
    _check(
        query_lengths.shape[0] > 0, "sequence lengths must contain at least one batch"
    )
    _check(
        query_lengths.shape[0] == kv_lengths.shape[0],
        "sequence batch dimensions must match",
    )
    _check(block_table.dim() == 2, "block_table must have rank 2")
    _check(
        block_table.shape[0] == query_lengths.shape[0],
        "block_table batch dimension must match",
    )
    _check(block_table.shape[1] > 0, "block_table must contain at least one column")
    for tensor in (sparse_indices, block_table, query_lengths, kv_lengths):
        _check(
            tensor.dtype == torch.int32,
            "indices, block table and lengths must be int32",
        )
    for scale in (key_scale, value_scale):
        _check(
            scale is None or scale.dtype == torch.float32,
            "dequant scales must be float32",
        )
    for tensor in (
        key,
        value,
        sparse_indices,
        block_table,
        query_lengths,
        kv_lengths,
        key_scale,
        value_scale,
    ):
        _check(
            tensor is None or tensor.device == query.device,
            "all inputs must be on the query device",
        )
    _check(
        key_quant_mode == 3 and value_quant_mode == 3,
        "key/value quant modes must both be 3",
    )
    _check(
        sparse_block_size in (1, 2, 4, 8, 16),
        "sparse_block_size must be a power of two in [1, 16]",
    )
    _check(
        block_size % sparse_block_size == 0,
        "block size must be divisible by sparse_block_size",
    )
    _check(sparse_mode in (0, 3), "sparse_mode must be 0 or 3")
    _check(
        pre_tokens == INT64_MAX and next_tokens == INT64_MAX,
        "pre_tokens and next_tokens must be INT64_MAX",
    )
    _check(
        attention_mode == 2
        and quant_scale_repo_mode == 1
        and tile_size == 128
        and rope_head_dim == 64,
        "attention_mode, quant_scale_repo_mode, tile_size and rope_head_dim must be 2, 1, 128 and 64",
    )


def _turbo_quant_sparse_flash_attention_meta(
    query,
    key,
    value,
    sparse_indices,
    *,
    key_dequant_scale=None,
    value_dequant_scale=None,
    block_table,
    actual_seq_lengths_query,
    actual_seq_lengths_kv,
    scale_value=1.0,
    key_quant_mode=3,
    value_quant_mode=3,
    sparse_block_size=1,
    layout_query="TND",
    layout_kv="PA_BSND",
    sparse_mode=3,
    pre_tokens=INT64_MAX,
    next_tokens=INT64_MAX,
    attention_mode=2,
    quant_scale_repo_mode=1,
    tile_size=128,
    rope_head_dim=64,
    return_softmax_lse=False,
):
    _check_inputs(
        query,
        key,
        value,
        sparse_indices,
        key_dequant_scale,
        value_dequant_scale,
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        layout_query,
        layout_kv,
        key_quant_mode,
        value_quant_mode,
        sparse_block_size,
        sparse_mode,
        pre_tokens,
        next_tokens,
        attention_mode,
        quant_scale_repo_mode,
        tile_size,
        rope_head_dim,
    )
    softmax_shape = (1, query.shape[0], query.shape[1]) if return_softmax_lse else (0,)
    return (
        query.new_empty((query.shape[0], query.shape[1], LATENT_HEAD_DIM)),
        query.new_empty(softmax_shape, dtype=torch.float32),
        query.new_empty(softmax_shape, dtype=torch.float32),
    )


class _TurboQuantSparseFlashAttentionOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("turbo_quant_sparse_flash_attention", category="attention")

    def sources(self):
        return ["csrc/attention/turbo_quant_sparse_flash_attention.cpp"]

    def schema(self):
        return (
            "turbo_quant_sparse_flash_attention(Tensor query, Tensor key, Tensor value, Tensor sparse_indices, *, "
            "Tensor? key_dequant_scale=None, Tensor? value_dequant_scale=None, "
            "Tensor block_table, Tensor actual_seq_lengths_query, Tensor actual_seq_lengths_kv, "
            "float scale_value=1.0, int key_quant_mode=3, int value_quant_mode=3, int sparse_block_size=1, "
            "str layout_query='TND', str layout_kv='PA_BSND', int sparse_mode=3, "
            "int pre_tokens=9223372036854775807, int next_tokens=9223372036854775807, int attention_mode=2, "
            "int quant_scale_repo_mode=1, int tile_size=128, int rope_head_dim=64, "
            "bool return_softmax_lse=False) -> (Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        impl(get_as_library(), self.name, "Meta")(
            _turbo_quant_sparse_flash_attention_meta
        )


_op_builder = _TurboQuantSparseFlashAttentionOpBuilder()
_op_builder._ensure_initialized()


@impl(get_as_library(), _op_builder.name, "PrivateUse1")
def _turbo_quant_sparse_flash_attention(
    query,
    key,
    value,
    sparse_indices,
    *,
    key_dequant_scale=None,
    value_dequant_scale=None,
    block_table,
    actual_seq_lengths_query,
    actual_seq_lengths_kv,
    scale_value=1.0,
    key_quant_mode=3,
    value_quant_mode=3,
    sparse_block_size=1,
    layout_query="TND",
    layout_kv="PA_BSND",
    sparse_mode=3,
    pre_tokens=INT64_MAX,
    next_tokens=INT64_MAX,
    attention_mode=2,
    quant_scale_repo_mode=1,
    tile_size=128,
    rope_head_dim=64,
    return_softmax_lse=False,
):
    return _op_builder.load().turbo_quant_sparse_flash_attention(
        query,
        key,
        value,
        sparse_indices,
        key_dequant_scale,
        value_dequant_scale,
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        scale_value,
        key_quant_mode,
        value_quant_mode,
        sparse_block_size,
        layout_query,
        layout_kv,
        sparse_mode,
        pre_tokens,
        next_tokens,
        attention_mode,
        quant_scale_repo_mode,
        tile_size,
        rope_head_dim,
        return_softmax_lse,
    )


def turbo_quant_sparse_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_indices: torch.Tensor,
    *,
    key_dequant_scale: Optional[torch.Tensor] = None,
    value_dequant_scale: Optional[torch.Tensor] = None,
    block_table: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_kv: torch.Tensor,
    scale_value: float = 1.0,
    key_quant_mode: int = 3,
    value_quant_mode: int = 3,
    sparse_block_size: int = 1,
    layout_query: str = "TND",
    layout_kv: str = "PA_BSND",
    sparse_mode: int = 3,
    pre_tokens: int = INT64_MAX,
    next_tokens: int = INT64_MAX,
    attention_mode: int = 2,
    quant_scale_repo_mode: int = 1,
    tile_size: int = 128,
    rope_head_dim: int = 64,
    return_softmax_lse: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """面向 DeepSeek Sparse Attention（DSA）推理场景，计算 TQ4 量化 KV cache 的稀疏注意力。

    Args:
        query: TND [T, N, 576]，float16 或 bfloat16；N 为 [1, 128] 内的 2 的幂。
        key: int8 KV cache，PA_BSND [Bn, Bs, 1, 386] 或 PA_BNBD [Bn, 1, Bs, 386]。
            每个 slot 包含 nibble、与 query 同型的 rope/scale、float16 scale。支持首轴非连续。
        value: 请传入与 key 相同的张量。
        sparse_indices: int32 [T, 1, K]，batch 内局部稀疏块索引，无效后缀填 -1。
        key_dequant_scale: 可选 float32 张量；COMBINE 模式不读取。
        value_dequant_scale: 可选 float32 张量；COMBINE 模式不读取。
        block_table: int32 [B, max_blocks]，物理页映射。
        actual_seq_lengths_query: int32 [B]，query 累计长度，允许重复值。
        actual_seq_lengths_kv: int32 [B]，每个 batch 的实际 KV 长度，允许为零。
        scale_value: QK 分数缩放系数，默认 1.0。
        key_quant_mode: 必须为 3（TQ4）。
        value_quant_mode: 必须为 3（TQ4）。
        sparse_block_size: 稀疏块大小，支持 1、2、4、8、16。
        layout_query: 必须为 TND。
        layout_kv: PA_BSND 或 PA_BNBD；不接受别名。
        sparse_mode: 0 为无因果掩码，3 为右下对齐因果掩码。
        pre_tokens: 必须为 INT64_MAX。
        next_tokens: 必须为 INT64_MAX。
        attention_mode: 当前 DSA 路径的底层计算模式，必须为 2（MLA-absorb）。
        quant_scale_repo_mode: 必须为 1（COMBINE）。
        tile_size: 必须为 128。
        rope_head_dim: 必须为 64。
        return_softmax_lse: 是否返回 softmax_max、softmax_sum，默认 False。

    Returns:
        attention_out [T, N, 512]，dtype 与 query 相同；softmax_max、softmax_sum 为
        float32 [1, T, N]，不请求时为 [0]。零 KV 行输出零，max=-2e38、sum=0。

    所有输入必须在同一 NPU；索引和长度张量的数值合法性由调用方保证。
    仅支持前向推理；支持 Meta/FakeTensor 和 ACL graph，GE Converter 暂未提供。
    """
    return torch.ops.cann_ops_transformer.turbo_quant_sparse_flash_attention(
        query,
        key,
        value,
        sparse_indices,
        key_dequant_scale=key_dequant_scale,
        value_dequant_scale=value_dequant_scale,
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        scale_value=scale_value,
        key_quant_mode=key_quant_mode,
        value_quant_mode=value_quant_mode,
        sparse_block_size=sparse_block_size,
        layout_query=layout_query,
        layout_kv=layout_kv,
        sparse_mode=sparse_mode,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        attention_mode=attention_mode,
        quant_scale_repo_mode=quant_scale_repo_mode,
        tile_size=tile_size,
        rope_head_dim=rope_head_dim,
        return_softmax_lse=return_softmax_lse,
    )
