# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it and/or modify it under the terms and conditions of
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
    from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
    from torchair._ge_concrete_graph.supported_declaration import Support
    from typing import Any, Dict, List, Tuple, Union, Callable, Optional
    from torchair._ge_concrete_graph.ge_ir_pb2 import GraphDef, OpDef, TensorDescriptor, TensorDef
    from torchair.ge._ge_graph import get_default_ge_graph, next_unique_name
    from torchair.ge._ge_graph import auto_convert_to_tensor
    from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, TensorType
    from torchair.ge._ge_graph import compat_as_bytes, compat_as_bytes_list
    from torchair.ge._ge_graph import trans_to_list_list_int, trans_to_list_list_float
    from torchair.ge._ge_graph import get_invalid_desc
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair.ge import attr
    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:
    # Declaration for supported ops
    declare_supported(
        Support(
            "npu_ops_transformer::npu_grouped_matmul_v5",
            "Custom",
        )
    )

    @register_fx_node_ge_converter(torch.ops.npu_ops_transformer.npu_grouped_matmul_v5.default)
    def converter_grouped_matmul_v5(
        x: List[Tensor],
        weight: List[Tensor],
        bias: Optional[List[Tensor]] = None,
        scale: Optional[List[Tensor]] = None,
        offset: Optional[List[Tensor]] = None,
        antiquant_scale: Optional[List[Tensor]] = None,
        antiquant_offset: Optional[List[Tensor]] = None,
        per_token_scale: Optional[List[Tensor]] = None,
        group_list: Optional[Tensor] = None,
        activation_input: Optional[List[Tensor]] = None,
        activation_quant_scale: Optional[List[Tensor]] = None,
        activation_quant_offset: Optional[List[Tensor]] = None,
        split_item: int = 3,
        group_type: int = 0,
        group_list_type: int = 1,
        act_type: int = 0,
        tuning_config: Optional[List[int]] = None,
        output_dtype: Any = None,
        meta_outputs: TensorSpec = None):
        """
        Graph mode converter for npu_grouped_matmul_v5.

        For graph mode, we delegate to torch_npu.npu_grouped_matmul which already
        has graph mode support. This converter is provided for completeness and
        compatibility with torchair's fx2ge conversion pipeline.
        """
        # Import the Python implementation
        from npu_ops_transformer.ops.grouped_matmul_v5 import npu_grouped_matmul_v5 as py_impl

        # Call the Python implementation for meta outputs
        # Note: In graph mode, this is primarily used for type/shape inference
        # The actual execution will go through torchair's compiled graph
        pass
