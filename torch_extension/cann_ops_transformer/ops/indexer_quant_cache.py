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
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY


class IndexerQuantCacheOpBuilder(OpBuilder):
    def __init__(self):
        super(IndexerQuantCacheOpBuilder, self).__init__("indexer_quant_cache")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/indexer_quant_cache.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        return "indexer_quant_cache(Tensor cache, Tensor cache_scale, Tensor x, " \
               "Tensor slot_mapping, *, int quant_mode=1, bool round_scale=True, " \
               "float x_scale=1.0) -> (Tensor, Tensor)"

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for FakeTensor / graph mode support.
        """
        @impl(AS_LIBRARY, self.name, "Meta")
        def indexer_quant_cache_meta(cache, cache_scale, x, slot_mapping, *,
                                     quant_mode=1, round_scale=True, x_scale=1.0):
            # 输出为更新后的 cache 与 cache_scale，shape/dtype 与各自输入相同
            return (torch.empty_like(cache), torch.empty_like(cache_scale))


# Instantiate the builder
indexer_quant_cache_op_builder = IndexerQuantCacheOpBuilder()
op_module = indexer_quant_cache_op_builder.load()  # Compiles/loads the .so file


@impl(AS_LIBRARY, indexer_quant_cache_op_builder.name, "PrivateUse1")
def indexer_quant_cache(cache, cache_scale, x, slot_mapping, *,
                        quant_mode=1, round_scale=True, x_scale=1.0):
    return op_module.indexer_quant_cache(cache, cache_scale, x, slot_mapping,
                                         quant_mode, round_scale, x_scale)