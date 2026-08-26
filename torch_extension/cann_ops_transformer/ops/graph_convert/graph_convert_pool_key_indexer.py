# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE Converter for Graph Mode

try:
    import torch
    import torch_npu
    import torchair
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )
    from torchair.ge._ge_graph import Tensor, TensorSpec
    from torchair.ge import attr
    from typing import Optional, List

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.pool_key_indexer.default
    )
    def convert_pool_key_indexer(
        query: Tensor,
        pool_key: Tensor,
        weights: Tensor,
        pool_tail_k: Tensor,
        *,
        actual_seq_q: Optional[Tensor] = None,
        actual_seq_k: Optional[Tensor] = None,
        block_table: Optional[Tensor] = None,
        q_descale: Optional[Tensor] = None,
        k_descale: Optional[Tensor] = None,
        layout_q: str = "BSND",
        layout_k: str = "BSND",
        topk: int = 128,
        pool_size: int = 1,
        mask_mode: int = 0,
        quant_mode: int = -1,
        return_value: bool = False,
        meta_outputs: List[TensorSpec] = None,
    ):
        return torchair.ge.custom_op(
            "PoolKeyIndexer",
            inputs={
                "query": query,
                "pool_key": pool_key,
                "weights": weights,
                "pool_tail_k": pool_tail_k,
                "actual_seq_q": actual_seq_q,
                "actual_seq_k": actual_seq_k,
                "block_table": block_table,
                "q_descale": q_descale,
                "k_descale": k_descale,
            },
            attrs={
                "layout_q": attr.Str(layout_q),
                "layout_k": attr.Str(layout_k),
                "topk": attr.Int(topk),
                "pool_size": attr.Int(pool_size),
                "mask_mode": attr.Int(mask_mode),
                "quant_mode": attr.Int(quant_mode),
                "return_value": attr.Bool(return_value),
            },
            outputs=["sparse_indices_out", "sparse_values_out"],
        )
