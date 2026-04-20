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
grouped_matmul_v5 - Python interface for aclnnGroupedMatmulV5

This module provides a torch.ops.npu_ops_transformer interface for aclnnGroupedMatmulV5,
simplified for DeepGEMM-style grouped matmul operations.
"""

import torch
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
        return "npu_grouped_matmul_v5(Tensor[] x, Tensor[] weight, " \
            "*, Tensor[]? bias=None, Tensor[]? scale=None, Tensor[]? offset=None, " \
            "Tensor[]? antiquant_scale=None, Tensor[]? antiquant_offset=None, " \
            "Tensor[]? per_token_scale=None, Tensor? group_list=None, " \
            "Tensor[]? activation_input=None, Tensor[]? activation_quant_scale=None, " \
            "Tensor[]? activation_quant_offset=None, " \
            "int split_item=3, int group_type=0, int group_list_type=1, int act_type=0, " \
            "int[]? tuning_config=None, ScalarType? output_dtype=None) -> Tensor[]"

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """

        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_grouped_matmul_v5_meta(
            x, weight, bias=None, scale=None, offset=None,
            antiquant_scale=None, antiquant_offset=None, per_token_scale=None,
            group_list=None, activation_input=None, activation_quant_scale=None,
            activation_quant_offset=None, split_item=3, group_type=0,
            group_list_type=1, act_type=0, tuning_config=None, output_dtype=None
        ):
            torch._check(len(x) > 0, lambda: "x list must not be empty" + ops_error(ErrCode.VALUE))
            torch._check(len(weight) > 0, lambda: "weight list must not be empty" + ops_error(ErrCode.VALUE))
            torch._check(
                split_item >= 0 and split_item <= 3,
                lambda: f"split_item must be in [0, 3], got {split_item}" + ops_error(ErrCode.VALUE)
            )

            num_x = len(x)
            num_weight = len(weight)
            num_group_list = group_list.numel() if group_list is not None and group_list.dim() > 0 else 0

            # Validate dimensions based on split_item
            if split_item == 0 or split_item == 1:
                if num_group_list > 0:
                    torch._check(
                        num_x == 1 and num_weight == num_group_list,
                        lambda: (f"When split_item is {split_item} and group_list is not None, "
                                 f"num of x must be 1 and num of weight must equal group_list length. "
                                 f"Got x={num_x}, weight={num_weight}, group_list={num_group_list}."
                                 + ops_error(ErrCode.VALUE))
                    )
                else:
                    torch._check(
                        num_x == num_weight,
                        lambda: (f"When split_item is {split_item} and group_list is None, "
                                 f"num of x must equal num of weight. "
                                 f"Got x={num_x}, weight={num_weight}."
                                 + ops_error(ErrCode.VALUE))
                    )

            out_dtype = output_dtype if output_dtype is not None else x[0].dtype
            options = x[0].new_empty((0,)).options().dtype(out_dtype)

            dim_num_w = weight[0].dim()
            n0 = weight[0].size(dim_num_w - 1)
            results = []

            if split_item == 0 or split_item == 1:
                if num_group_list > 0:
                    glr_value_0 = group_list[0].item()
                    results.append(x[0].new_empty((glr_value_0, n0), dtype=out_dtype))
                    glr_value_pre = glr_value_0
                    for i in range(1, num_group_list):
                        glr_value_cur = group_list[i].item()
                        ni = weight[i].size(0) if dim_num_w == 3 else weight[i].size(dim_num_w - 1)
                        results.append(x[0].new_empty((glr_value_cur - glr_value_pre, ni), dtype=out_dtype))
                        glr_value_pre = glr_value_cur
                else:
                    for i in range(num_x):
                        ni = weight[i].size(0) if dim_num_w == 3 else weight[i].size(dim_num_w - 1)
                        x_size = list(x[i].size())
                        x_size[-1] = ni
                        results.append(x[0].new_empty(tuple(x_size), dtype=out_dtype))
            else:  # split_item == 2 or 3
                if num_x > 1:
                    dim_m = sum(x[i].size(0) for i in range(num_x))
                    n_out = n0 * 8 if weight[0].dtype == torch.int32 else n0
                    results.append(x[0].new_empty((dim_m, n_out), dtype=out_dtype))
                elif num_x == 1:
                    if group_type == 2:
                        # K_SPLIT: output is 3D
                        results.append(x[0].new_empty((num_group_list, x[0].size(0), n0), dtype=out_dtype))
                    else:
                        n_out = n0 * 8 if weight[0].dtype == torch.int32 else n0
                        results.append(x[0].new_empty((x[0].size(0), n_out), dtype=out_dtype))

            return results


# Instantiate the builder
grouped_matmul_v5_op_builder = GroupedMatmulV5OpBuilder()
op_module = grouped_matmul_v5_op_builder.load()  # Compiles/loads the .so file


@impl(AS_LIBRARY, grouped_matmul_v5_op_builder.name, "PrivateUse1")
def npu_grouped_matmul_v5(
    x, weight, bias=None, scale=None, offset=None,
    antiquant_scale=None, antiquant_offset=None, per_token_scale=None,
    group_list=None, activation_input=None, activation_quant_scale=None,
    activation_quant_offset=None, split_item=3, group_type=0,
    group_list_type=1, act_type=0, tuning_config=None, output_dtype=None
):
    """
    Grouped Matmul V5 implementation for NPU.

    'PrivateUse1' is the dispatch key for custom NPU backends.
    """
    return op_module.npu_grouped_matmul_v5(
        x, weight, bias, scale, offset, antiquant_scale, antiquant_offset,
        per_token_scale, group_list, activation_input, activation_quant_scale,
        activation_quant_offset, split_item, group_type, group_list_type,
        act_type, tuning_config, output_dtype
    )
