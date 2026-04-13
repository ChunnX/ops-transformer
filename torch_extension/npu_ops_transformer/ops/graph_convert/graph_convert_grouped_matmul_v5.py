# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Graph mode converter for GroupedMatmulV5.
Converts PyTorch FX graph nodes to TorChair GE graph nodes.
"""
import torch
import torch_npu

try:
    import torchair
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair.ge._ge_graph import Tensor, TensorSpec
    from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
    from torchair._ge_concrete_graph.supported_declaration import Support
    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False


if _TORCHAIR_AVAILABLE:
    @declare_supported([
        Support(torch.float16, [(8, 128, 256), (8, 256, 512)]),
        Support(torch.bfloat16, [(8, 128, 256), (8, 256, 512)]),
        Support(torch.float32, [(8, 128, 256), (8, 256, 512)]),
    ])
    @register_fx_node_ge_converter(torch.ops.npu_ops_transformer.npu_grouped_matmul_v5.default)
    def convert_npu_grouped_matmul_v5(
        x: list,
        weight: list,
        bias: list = None,
        scale: list = None,
        offset: list = None,
        antiquant_scale: list = None,
        antiquant_offset: list = None,
        per_token_scale: list = None,
        group_list: Tensor = None,
        activation_input: list = None,
        activation_quant_scale: list = None,
        activation_quant_offset: list = None,
        split_item: int = 0,
        group_type: int = -1,
        group_list_type: int = 0,
        act_type: int = 0,
        tuning_config: list = None,
        meta_outputs: TensorSpec = None):
        """
        Convert npu_grouped_matmul_v5 to GE graph node.

        Args:
            x: Input tensor list
            weight: Weight tensor list
            bias: Optional bias tensor list
            scale: Optional quantization scale tensor list
            offset: Optional quantization offset tensor list
            antiquant_scale: Optional antiquant scale tensor list
            antiquant_offset: Optional antiquant offset tensor list
            per_token_scale: Optional per-token scale tensor list
            group_list: Optional group list tensor
            activation_input: Optional activation input tensor list
            activation_quant_scale: Optional activation quant scale tensor list
            activation_quant_offset: Optional activation quant offset tensor list
            split_item: Split item mode
            group_type: Group type
            group_list_type: Group list type
            act_type: Activation type
            tuning_config: Optional tuning config
            meta_outputs: Meta output specification

        Returns:
            GE graph node for GroupedMatmulV5
        """
        return ge.GroupedMatmulV5(
            x, weight,
            bias=bias,
            scale=scale,
            offset=offset,
            antiquant_scale=antiquant_scale,
            antiquant_offset=antiquant_offset,
            per_token_scale=per_token_scale,
            group_list=group_list,
            activation_input=activation_input,
            activation_quant_scale=activation_quant_scale,
            activation_quant_offset=activation_quant_offset,
            split_item=split_item,
            group_type=group_type,
            group_list_type=group_list_type,
            act_type=act_type,
            tuning_config=tuning_config
        )
