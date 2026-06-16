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


class IndexerQuantCacheOpBuilder(OpBuilder):
    def __init__(self):
        super(IndexerQuantCacheOpBuilder, self).__init__("npu_indexer_quant_cache")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/indexer_quant_cache.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        return "npu_indexer_quant_cache(Tensor cache, Tensor cache_scale, Tensor x, " \
               "Tensor slot_mapping, *, int quant_mode=1, bool round_scale=True, " \
               "float x_scale=1.0) -> (Tensor, Tensor)"

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for FakeTensor / graph mode support.
        """
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_indexer_quant_cache_meta(cache, cache_scale, x, slot_mapping, *,
                                         quant_mode=1, round_scale=True, x_scale=1.0):
            # 输出为更新后的 cache 与 cache_scale，shape/dtype 与各自输入相同
            return (torch.empty_like(cache), torch.empty_like(cache_scale))


# Instantiate the builder
npu_indexer_quant_cache_op_builder = IndexerQuantCacheOpBuilder()
op_module = npu_indexer_quant_cache_op_builder.load()  # Compiles/loads the .so file


@impl(AS_LIBRARY, npu_indexer_quant_cache_op_builder.name, "PrivateUse1")
def npu_indexer_quant_cache(cache, cache_scale, x, slot_mapping, *,
                            quant_mode=1, round_scale=True, x_scale=1.0):
    """在 Indexer 注意力机制的 Epilog 阶段对 KV Cache 进行逐块动态量化压缩更新，封装 aclnnIndexerQuantCache。

    将 float16/bfloat16 激活值 x 按尾轴每 128 个元素一组（MX-FP4 模式为每 32 个元素一组）动态量化为
    FP8（E4M3/E5M2）、INT8（uint8）或 MX-FP4，并按 slot_mapping 将量化结果与对应 scale 散写到
    cache/cache_scale，slot_mapping 中值为 -1 的 token 跳过不处理。返回更新后的 (cache, cache_scale)
    （函数式语义，底层 aclnn 接口为原地实现）。

    Args:
        cache (Tensor): 当前层的 KV Cache 向量缓存，shape 为 [num_slots, d]。dtype 支持
            float8_e4m3fn/float8_e5m2/uint8/float4_e2m1/float4_e1m2（MX-FP4 模式下为 uint8，每字节打包 2 个 fp4 值）。
        cache_scale (Tensor): 每块量化的 scale 因子，shape 为 [num_slots, CeilDiv(d, 128)]。dtype 支持
            float32（Normal/HiFloat8 模式）或 float8_e8m0（MX-FP8/MX-FP4 模式）。
        x (Tensor): 待量化的激活输入，dtype 为 float16 或 bfloat16，shape 为 [bs, d]，d 需满足 d % 32 == 0。
        slot_mapping (Tensor): token 到 cache slot 的索引映射，dtype 为 int32，shape 为 [bs]，元素取值范围
            [-1, num_slots - 1]，-1 表示跳过该 token。维度需等于 x 的维度减 1。
        quant_mode (int): 量化模式，0=MX-FP8、1=Normal、2=HiFloat8、3=MX-FP4，默认 1。
        round_scale (bool): MX-FP8 模式（quant_mode=0）下是否对 scale 进行舍入，默认 True。
        x_scale (float): HiFloat8 模式（quant_mode=2）下的全局 scale 乘数，默认 1.0。

    Returns:
        Tuple[Tensor, Tensor]: 更新后的 (cache, cache_scale)，shape/dtype 与各自输入相同。

    仅支持 Ascend 950PR/Ascend 950DT。
    """
    return op_module.npu_indexer_quant_cache(cache, cache_scale, x, slot_mapping,
                                             quant_mode, round_scale, x_scale)