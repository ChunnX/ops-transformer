# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
from torch_npu.utils._error_code import ErrCode, ops_error
from npu_ops_transformer.op_builder.builder import OpBuilder
from npu_ops_transformer.op_builder.builder import AS_LIBRARY


class GroupedMatmulV5OpBuilder(OpBuilder):
    def __init__(self):
        super(GroupedMatmulV5OpBuilder, self).__init__("npu_grouped_matmul_v5")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/grouped_matmul_v5.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        return "npu_grouped_matmul_v5(Tensor[] x, Tensor[] weight, Tensor[]? bias=None, " \
               "Tensor[]? scale=None, Tensor[]? offset=None, Tensor[]? antiquant_scale=None, " \
               "Tensor[]? antiquant_offset=None, Tensor[]? per_token_scale=None, " \
               "Tensor? group_list=None, Tensor[]? activation_input=None, " \
               "Tensor[]? activation_quant_scale=None, Tensor[]? activation_quant_offset=None, " \
               "int split_item=0, int group_type=-1, int group_list_type=0, int act_type=0, " \
               "int[]? tuning_config=None) -> (Tensor[], Tensor[]?, Tensor[]?)"

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_grouped_matmul_v5_meta(x, weight, bias=None, scale=None, offset=None,
                                       antiquant_scale=None, antiquant_offset=None,
                                       per_token_scale=None, group_list=None,
                                       activation_input=None, activation_quant_scale=None,
                                       activation_quant_offset=None, split_item=0,
                                       group_type=-1, group_list_type=0, act_type=0,
                                       tuning_config=None):
            torch._check(
                len(x) > 0,
                lambda: f"x tensor list cannot be empty{ops_error(ErrCode.VALUE)}."
            )
            torch._check(
                len(weight) > 0,
                lambda: f"weight tensor list cannot be empty{ops_error(ErrCode.VALUE)}."
            )
            torch._check(
                len(x) == len(weight),
                lambda: (
                    f"x and weight must have the same number of tensors, "
                    f"but got len(x)={len(x)}, len(weight)={len(weight)}"
                    f"{ops_error(ErrCode.VALUE)}."
                )
            )
            torch._check(
                split_item in [0, 1, 2, 3],
                lambda: f"split_item must be in [0, 3], but got {split_item}{ops_error(ErrCode.VALUE)}."
            )
            torch._check(
                group_type in [-1, 0, 1, 2],
                lambda: f"group_type must be in [-1, 2], but got {group_type}{ops_error(ErrCode.VALUE)}."
            )
            torch._check(
                group_list_type in [0, 1],
                lambda: f"group_list_type must be in [0, 1], but got {group_list_type}{ops_error(ErrCode.VALUE)}."
            )
            torch._check(
                act_type in [0, 1, 2, 3, 4, 5],
                lambda: f"act_type must be in [0, 5], but got {act_type}{ops_error(ErrCode.VALUE)}."
            )

            # Calculate output shapes
            out_list = []
            if split_item in [0, 1]:
                # Multi-tensor output
                for i in range(len(x)):
                    x_shape = x[i].shape
                    weight_shape = weight[i].shape
                    torch._check(
                        len(x_shape) >= 2 and len(weight_shape) >= 2,
                        lambda: f"x and weight must be at least 2D tensors{ops_error(ErrCode.VALUE)}."
                    )

                    m = x_shape[-2]
                    n = weight_shape[-1]
                    out_shape = list(x_shape[:-1]) + [n]
                    out_list.append(x[i].new_empty(out_shape))
            else:
                # Single-tensor output
                total_m = sum(x_tensor.shape[-2] for x_tensor in x)
                n = weight[0].shape[-1]
                out_list.append(x[0].new_empty([total_m, n]))

            # Optional outputs (not implemented yet)
            activation_feature_out = None
            dyn_quant_scale_out = None

            return (out_list, activation_feature_out, dyn_quant_scale_out)


# Instantiate the builder
grouped_matmul_v5_op_builder = GroupedMatmulV5OpBuilder()
op_module = grouped_matmul_v5_op_builder.load()  # Compiles/loads the .so file


@impl(AS_LIBRARY, grouped_matmul_v5_op_builder.name, "PrivateUse1")
def npu_grouped_matmul_v5(x, weight, bias=None, scale=None, offset=None,
                          antiquant_scale=None, antiquant_offset=None,
                          per_token_scale=None, group_list=None,
                          activation_input=None, activation_quant_scale=None,
                          activation_quant_offset=None, split_item=0,
                          group_type=-1, group_list_type=0, act_type=0,
                          tuning_config=None):
    """
    Dispatcher implementation for NPU.
    'PrivateUse1' is the dispatch key for custom NPU backends.
    """
    return op_module.npu_grouped_matmul_v5(
        x, weight, bias, scale, offset,
        antiquant_scale, antiquant_offset, per_token_scale,
        group_list, activation_input, activation_quant_scale,
        activation_quant_offset, split_item, group_type,
        group_list_type, act_type, tuning_config
    )


# GE Converter for Graph Mode
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
        Support(torch.float16, [(8, 128), (8, 128)]),
        Support(torch.bfloat16, [(8, 128), (8, 128)]),
    ])
    @register_fx_node_ge_converter(torch.ops.npu_ops_transformer.npu_grouped_matmul_v5.default)
    def converter_grouped_matmul_v5(
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
