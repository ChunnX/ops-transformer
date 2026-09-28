# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE Converter for Graph Mode

try:
    import torch
    import torch_npu
    import torchair
    from torch.library import impl
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair.ge._ge_graph import Tensor, TensorSpec
    from torchair._ge_concrete_graph.fx2ge_converter import (
        declare_supported,
        register_fx_node_ge_converter,
    )
    from torchair._ge_concrete_graph.supported_declaration import Support
    from typing import Any, Dict, List, Tuple, Union, Callable, Optional
    from torchair._ge_concrete_graph.ge_ir_pb2 import (
        GraphDef,
        OpDef,
        TensorDescriptor,
        TensorDef,
    )
    from torchair.ge._ge_graph import get_default_ge_graph, next_unique_name
    from torchair.ge._ge_graph import auto_convert_to_tensor
    from torchair.ge._ge_graph import DataType, TensorType
    from torchair.ge._ge_graph import compat_as_bytes, compat_as_bytes_list
    from torchair.ge._ge_graph import trans_to_list_list_int, trans_to_list_list_float
    from torchair.ge._ge_graph import get_invalid_desc
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair.ge import attr

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.sparse_flash_attention.default
    )
    def convert_sparse_flash_attention(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        sparse_indices: Tensor,
        scale_value: float = 1.0,
        *,
        block_table: Optional[Tensor] = None,
        actual_seq_lengths_query: Optional[Tensor] = None,
        actual_seq_lengths_kv: Optional[Tensor] = None,
        query_rope: Optional[Tensor] = None,
        key_rope: Optional[Tensor] = None,
        sinks: Optional[Tensor] = None,
        sparse_block_size: int = 1,
        layout_query: str = "BSND",
        layout_kv: str = "BSND",
        sparse_mode: int = 3,
        pre_tokens: int = 9223372036854775807,
        next_tokens: int = 9223372036854775807,
        attention_mode: int = 0,
        return_softmax_lse: bool = False,
        meta_outputs: TensorSpec = None,
    ):
        raise RuntimeError(
            "The cann_ops_transformer.sparse_flash_attention does not support dynamic or static graph execution."
        )
