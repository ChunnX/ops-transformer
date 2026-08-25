# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file in compliance with the License.
# THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from typing import Optional

import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library

# aclDataType (acl_base.h): ACL_BF16 = 27
ACL_BF16 = 27


class GatherPaKvCacheWithDequantOpBuilder(OpBuilder):
    """
    torch_npu适配：gather_pa_kv_cache_with_dequant算子

    功能：从分页 KV cache 中按 block_tables/seq_lens gather 指定 token 并反量化为 BF16，
         原地写入 key_ref/value_ref（调用方预分配输出 tensor）。

    输入：
        key_cache:        INT8  [num_blocks, block_size, num_heads_k, head_dim_k]
        value_cache:      FP8   [num_blocks, block_size, num_heads_v, head_dim_v]
        key_scale_cache:  FLOAT Norm: [nb, bs, nh_k] / HBS: [nb, nh_k, bs] / C8: [nh_k, hd_k]
        value_scale:      FLOAT [num_heads_v, head_dim_v]
        block_tables:     INT32 [batch, max_blocks_per_seq]
        seq_lens:         INT32 [batch] (或 [batch+1] cumsum)

    输出（inplace更新）：
        key_ref:          BF16 [num_tokens, num_heads_k, head_dim_k]
        value_ref:        BF16 [num_tokens, num_heads_v, head_dim_v]
    """

    def __init__(self):
        super(GatherPaKvCacheWithDequantOpBuilder, self).__init__(
            "gather_pa_kv_cache_with_dequant", category="attention"
        )

    def sources(self):
        """C++源代码路径"""
        return ["csrc/attention/gather_pa_kv_cache_with_dequant.cpp"]

    def schema(self) -> str:
        """PyTorch算子签名"""
        return (
            "gather_pa_kv_cache_with_dequant("
            "Tensor key_cache, Tensor value_cache, Tensor key_scale_cache, "
            "Tensor value_scale, Tensor block_tables, Tensor seq_lens, "
            "Tensor(a!) key_ref, Tensor(b!) value_ref, *, "
            "Tensor? seq_offset=None, str cache_mode='Norm', "
            "bool is_seq_lens_cumsum=False, int key_dst_type=27, "
            "int value_dst_type=27) -> ()"
        )

    def register_meta(self):
        """
        Meta实现：Shape/Dtype推导

        原地更新 key_ref/value_ref，无返回值。
        torch.compile / FakeTensor 需要准确的 mutation 信息（schema 中 Tensor(a!)/Tensor(b!)）。
        """

        @impl(get_as_library(), self.name, "Meta")
        def gather_pa_kv_cache_with_dequant_meta(
            key_cache,
            value_cache,
            key_scale_cache,
            value_scale,
            block_tables,
            seq_lens,
            key_ref,
            value_ref,
            *,
            seq_offset=None,
            cache_mode="Norm",
            is_seq_lens_cumsum=False,
            key_dst_type=ACL_BF16,
            value_dst_type=ACL_BF16,
        ) -> None:
            return None


# 实例化OpBuilder
gather_pa_kv_cache_with_dequant_op_builder = GatherPaKvCacheWithDequantOpBuilder()
gather_pa_kv_cache_with_dequant_op_builder._ensure_initialized()


@impl(get_as_library(), gather_pa_kv_cache_with_dequant_op_builder.name, "PrivateUse1")
def gather_pa_kv_cache_with_dequant(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    key_scale_cache: torch.Tensor,
    value_scale: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    key_ref: torch.Tensor,
    value_ref: torch.Tensor,
    *,
    seq_offset: Optional[torch.Tensor] = None,
    cache_mode: str = "Norm",
    is_seq_lens_cumsum: bool = False,
    key_dst_type: int = ACL_BF16,
    value_dst_type: int = ACL_BF16,
) -> None:
    """
    torch_npu实现：gather_pa_kv_cache_with_dequant

    从分页 KV cache 中 gather 指定 token 并反量化为 BF16，原地写入 key_ref/value_ref。
    封装 aclnnGatherPaKvCacheWithDequant。

    Args:
        key_cache (Tensor): INT8 key cache，shape [num_blocks, block_size, num_heads_k, head_dim_k]。
            NZ 物理布局时逻辑 shape 不变，format 由框架自动传递给 aclnn（零拷贝）。
        value_cache (Tensor): FP8 (float8_e4m3fn) value cache，
            shape [num_blocks, block_size, num_heads_v, head_dim_v]。
        key_scale_cache (Tensor): FLOAT32 key scale cache。
            Norm: [num_blocks, block_size, num_heads_k] 或 HBS: [num_blocks, num_heads_k, block_size]；
            C8: [num_heads_k, head_dim_k]（per-channel）。strided allowed。
        value_scale (Tensor): FLOAT32 value scale，[num_heads_v, head_dim_v]（per-head-per-channel, static）。
        block_tables (Tensor): INT32 [num_batches, max_blocks_per_seq]，分页 block 索引。
        seq_lens (Tensor): INT32 [num_batches]，每个 batch 的真实序列长度；
            若 is_seq_lens_cumsum=True 则为 [num_batches+1] 前缀和形式。
        key_ref (Tensor): OUT BF16 [num_tokens, num_heads_k, head_dim_k]，被原地写入。
        value_ref (Tensor): OUT BF16 [num_tokens, num_heads_v, head_dim_v]，被原地写入。
        seq_offset (Tensor, optional): INT32，token 写入的起始偏移，默认 None。
        cache_mode (str): scale 布局与 gather 逻辑模式，合法值 "Norm"/"C8"/"Norm_NZ"/"C8_NZ"，
            默认 "Norm"。含 "C8" 子串 → per-channel scale；含 "NZ" 子串 → NZ 物理 KV 布局。
        is_seq_lens_cumsum (bool): seq_lens 是否为前缀和形式，默认 False。
        key_dst_type (int): key 目标 dtype（aclDataType 枚举，BF16=27），默认 27。
        value_dst_type (int): value 目标 dtype（aclDataType 枚举，BF16=27），默认 27。

    Returns:
        None: 算子原地更新 key_ref/value_ref，无返回值。

    Note:
        1. key_ref/value_ref 必须由调用方预分配（shape/dtype 正确），算子原地 mutate。
        2. NZ 物理 KV cache 支持零拷贝：框架通过 at_npu::native::get_npu_format() 自动传递
           ACL_FORMAT_FRACTAL_NZ 与 storage_sizes，无需调用方额外处理。
        3. value_cache 锁死 FP8 (float8_e4m3fn)；若误传 int8 会被 C++ 校验拦截。
        4. NZ 布局要求 head_dim 32 对齐（生产 head_dim=128 满足）。
    """
    op_module = gather_pa_kv_cache_with_dequant_op_builder.load()
    op_module.gather_pa_kv_cache_with_dequant(
        key_cache,
        value_cache,
        key_scale_cache,
        value_scale,
        block_tables,
        seq_lens,
        key_ref,
        value_ref,
        seq_offset,
        cache_mode,
        is_seq_lens_cumsum,
        key_dst_type,
        value_dst_type,
    )
