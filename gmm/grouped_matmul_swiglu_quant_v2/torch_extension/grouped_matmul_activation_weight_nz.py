# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from typing import List, Optional

import torch
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library


MX_DEQUANT_MODE = 2
FLOAT_DEQUANT_DTYPE = 0
SWIGLU_ACT_TYPE = 0
SWIGLU_MODE_NO_CLAMP = 0


def _require_tensor_list(value, name):
    if not isinstance(value, list) or not all(isinstance(item, torch.Tensor) for item in value):
        raise TypeError(f"{name} must be a TensorList (list of Tensor).")
    return value


class GroupedMatmulActivationWeightNzOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("grouped_matmul_activation_weight_nz", category="gmm")

    def sources(self):
        return ["csrc/gmm/grouped_matmul_activation_weight_nz.cpp"]

    def schema(self):
        return (
            "grouped_matmul_activation_weight_nz("
            "Tensor x, Tensor x_scale, Tensor group_list, Tensor[] weight, Tensor[] weight_scale, "
            "*, Tensor? bias=None, int dequant_mode=2, int dequant_dtype=0, int act_type=0, "
            "bool transpose_weight=False, int group_list_type=0, int[]? tuning_config=None, "
            "int swiglu_mode=0, float clamp_limit=0.0, float glu_alpha=1.0, float glu_bias=0.0) -> Tensor"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def grouped_matmul_activation_weight_nz_meta(
            x,
            x_scale,
            group_list,
            weight,
            weight_scale,
            bias=None,
            dequant_mode=MX_DEQUANT_MODE,
            dequant_dtype=FLOAT_DEQUANT_DTYPE,
            act_type=SWIGLU_ACT_TYPE,
            transpose_weight=False,
            group_list_type=0,
            tuning_config=None,
            swiglu_mode=SWIGLU_MODE_NO_CLAMP,
            clamp_limit=0.0,
            glu_alpha=1.0,
            glu_bias=0.0,
        ):
            if not weight_scale or weight_scale[0] is None:
                raise ValueError("weight_scale must contain one tensor for output-shape inference.")
            if x.dim() == 0 or weight_scale[0].dim() != 4:
                raise ValueError("x must be at least 1D and weight_scale[0] must be 4D.")
            logical_n = weight_scale[0].shape[2]
            if logical_n <= 0 or logical_n % 2 != 0:
                raise ValueError("The logical N dimension must be positive and even.")
            return torch.empty((x.shape[0], logical_n // 2), dtype=torch.bfloat16, device="meta")


_op_builder = GroupedMatmulActivationWeightNzOpBuilder()
_op_builder._ensure_initialized()


@impl(get_as_library(), _op_builder.name, "PrivateUse1")
def _grouped_matmul_activation_weight_nz(
    x,
    x_scale,
    group_list,
    weight,
    weight_scale,
    bias=None,
    dequant_mode=MX_DEQUANT_MODE,
    dequant_dtype=FLOAT_DEQUANT_DTYPE,
    act_type=SWIGLU_ACT_TYPE,
    transpose_weight=False,
    group_list_type=0,
    tuning_config=None,
    swiglu_mode=SWIGLU_MODE_NO_CLAMP,
    clamp_limit=0.0,
    glu_alpha=1.0,
    glu_bias=0.0,
):
    return _op_builder.load().grouped_matmul_activation_weight_nz(
        x, x_scale, group_list, weight, weight_scale, bias, dequant_mode, dequant_dtype, act_type,
        transpose_weight, group_list_type, tuning_config, swiglu_mode, clamp_limit, glu_alpha, glu_bias
    )


def grouped_matmul_activation_weight_nz(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    group_list: torch.Tensor,
    weight: List[torch.Tensor],
    weight_scale: List[torch.Tensor],
    *,
    bias: Optional[torch.Tensor] = None,
    dequant_mode: int = MX_DEQUANT_MODE,
    dequant_dtype: int = FLOAT_DEQUANT_DTYPE,
    act_type: int = SWIGLU_ACT_TYPE,
    transpose_weight: bool = False,
    group_list_type: int = 0,
    tuning_config: Optional[List[int]] = None,
    swiglu_mode: int = SWIGLU_MODE_NO_CLAMP,
    clamp_limit: float = 0.0,
    glu_alpha: float = 1.0,
    glu_bias: float = 0.0,
) -> torch.Tensor:
    """A4W4 WeightNZ grouped matmul followed by SwiGLU, with direct BF16 output."""
    return torch.ops.cann_ops_transformer.grouped_matmul_activation_weight_nz(
        x,
        x_scale,
        group_list,
        _require_tensor_list(weight, "weight"),
        _require_tensor_list(weight_scale, "weight_scale"),
        bias=bias,
        dequant_mode=dequant_mode,
        dequant_dtype=dequant_dtype,
        act_type=act_type,
        transpose_weight=transpose_weight,
        group_list_type=group_list_type,
        tuning_config=tuning_config,
        swiglu_mode=swiglu_mode,
        clamp_limit=clamp_limit,
        glu_alpha=glu_alpha,
        glu_bias=glu_bias,
    )
