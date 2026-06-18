# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import math

import torch
import torch_npu
from torch.library import impl
from cann_ops_transformer.op_builder.builder import AS_LIBRARY
from cann_ops_transformer.op_builder.builder import OpBuilder


GE_DTYPE_FLOAT = 0
GE_DTYPE_FLOAT8_E5M2 = 35
GE_DTYPE_FLOAT8_E4M3FN = 36
GE_DTYPE_FLOAT8_E8M0 = 37
ACL_DTYPE_OFFSET = 256
DEFAULT_Y_DTYPE = GE_DTYPE_FLOAT8_E4M3FN

FLOAT8_E8M0_DTYPE = getattr(torch_npu, "float8_e8m0fnu", getattr(torch, "float8_e8m0fnu", torch.uint8))

_TORCH_DTYPE_TO_GE_DTYPE = {
    torch.float32: GE_DTYPE_FLOAT,
    torch.float8_e5m2: GE_DTYPE_FLOAT8_E5M2,
    torch.float8_e4m3fn: GE_DTYPE_FLOAT8_E4M3FN,
    FLOAT8_E8M0_DTYPE: GE_DTYPE_FLOAT8_E8M0,
}

_GE_DTYPE_TO_TORCH_DTYPE = {
    GE_DTYPE_FLOAT8_E5M2: torch.float8_e5m2,
    GE_DTYPE_FLOAT8_E4M3FN: torch.float8_e4m3fn,
    GE_DTYPE_FLOAT8_E8M0: FLOAT8_E8M0_DTYPE,
}


def _normalize_tensor_list(value, name):
    if isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a TensorList (list or tuple of Tensor), but got a single Tensor.")
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        for index, tensor in enumerate(value):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name}[{index}] must be a Tensor, but got {type(tensor)}.")
        return value
    raise TypeError(f"{name} must be a TensorList (list or tuple of Tensor), but got {type(value)}.")


def _normalize_bias(bias):
    if bias is None:
        return None
    return _normalize_tensor_list(bias, "bias")


def _normalize_attr_dtype(dtype, default=None):
    if dtype is None:
        return default
    if isinstance(dtype, torch.dtype):
        if dtype not in _TORCH_DTYPE_TO_GE_DTYPE:
            raise TypeError(f"Unsupported dtype attr: {dtype}.")
        return _TORCH_DTYPE_TO_GE_DTYPE[dtype]
    if isinstance(dtype, int):
        if dtype >= ACL_DTYPE_OFFSET:
            return dtype - ACL_DTYPE_OFFSET
        return dtype
    raise TypeError(f"dtype attr must be an int or torch.dtype, but got {type(dtype)}.")


def _normalize_wrapper_dtype(dtype):
    if dtype is None:
        return None
    ge_dtype = _normalize_attr_dtype(dtype)
    if ge_dtype in (GE_DTYPE_FLOAT, GE_DTYPE_FLOAT8_E5M2, GE_DTYPE_FLOAT8_E4M3FN, GE_DTYPE_FLOAT8_E8M0):
        return ge_dtype + ACL_DTYPE_OFFSET
    return dtype


def _to_torch_dtype(dtype):
    ge_dtype = _normalize_attr_dtype(dtype, DEFAULT_Y_DTYPE)
    if ge_dtype not in _GE_DTYPE_TO_TORCH_DTYPE:
        raise TypeError(f"Unsupported y_dtype: {dtype}.")
    return _GE_DTYPE_TO_TORCH_DTYPE[ge_dtype]


def _get_effective_x_ge_dtype(x, x_dtype):
    if x_dtype is not None:
        return _normalize_attr_dtype(x_dtype)
    return _TORCH_DTYPE_TO_GE_DTYPE.get(x.dtype)


def _resolve_quant_mode(quant_mode, x, x_dtype):
    x_ge_dtype = _get_effective_x_ge_dtype(x, x_dtype)
    if quant_mode in (None, "") and x_ge_dtype in (GE_DTYPE_FLOAT8_E4M3FN, GE_DTYPE_FLOAT8_E5M2):
        return "MX"
    if x_ge_dtype in (GE_DTYPE_FLOAT8_E4M3FN, GE_DTYPE_FLOAT8_E5M2) and quant_mode != "MX":
        raise ValueError("当xdtype为float8_e4m3 或 float8_e5m2时，quant_mode必须为MX")
    return quant_mode


def _infer_weight_layout(weight, weight_scale, e, k):
    weight_shape = tuple(weight.shape)
    weight_scale_shape = tuple(weight_scale.shape)

    non_trans_n = weight_scale.shape[2]
    non_trans_weight_shape = (e, math.ceil(non_trans_n / 32), math.ceil(k / 16), 16, 32)
    non_trans_weight_scale_shape = (e, math.ceil(k / 64), non_trans_n, 2)
    if weight_shape == non_trans_weight_shape and weight_scale_shape == non_trans_weight_scale_shape:
        return non_trans_n

    trans_n = weight_scale.shape[1]
    trans_weight_shape = (e, math.ceil(k / 32), math.ceil(trans_n / 16), 16, 32)
    trans_weight_scale_shape = (e, trans_n, math.ceil(k / 64), 2)
    if weight_shape == trans_weight_shape and weight_scale_shape == trans_weight_scale_shape:
        return trans_n

    raise AssertionError(
        "weight and weight_scale shape must match one of the supported WeightNz layouts: "
        f"non-transposed weight {non_trans_weight_shape}, weight_scale {non_trans_weight_scale_shape}; "
        f"transposed weight {trans_weight_shape}, weight_scale {trans_weight_scale_shape}. "
        f"Got weight {weight_shape}, weight_scale {weight_scale_shape}."
    )


class GroupedMatmulActivationQuantOpBuilder(OpBuilder):
    def __init__(self):
        super(GroupedMatmulActivationQuantOpBuilder, self).__init__("npu_grouped_matmul_activation_quant")

    def sources(self):
        return ["ops/csrc/grouped_matmul_activation_quant.cpp"]

    def schema(self) -> str:
        return "npu_grouped_matmul_activation_quant(" \
            "Tensor x, Tensor[] weight, Tensor[] weight_scale, Tensor group_list, *, " \
            "Tensor? x_scale=None, Tensor[]? bias=None, str activation_type=\"gelu_tanh\", " \
            "int group_list_type=0, int[]? tuning_config=None, " \
            "str quant_mode=\"\", int y_dtype=36, str round_mode=\"rint\", int scale_alg=0, " \
            "float dst_type_max=0.0, int? x_dtype=None, int? weight_dtype=None, " \
            "int? weight_scale_dtype=None, int? x_scale_dtype=None) -> (Tensor, Tensor)"

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_grouped_matmul_activation_quant_meta(x, weight, weight_scale, group_list,
                                                     x_scale=None, bias=None, activation_type="gelu_tanh",
                                                     group_list_type=0, tuning_config=None,
                                                     quant_mode="", y_dtype=DEFAULT_Y_DTYPE,
                                                     round_mode="rint", scale_alg=0,
                                                     dst_type_max=0.0, x_dtype=None,
                                                     weight_dtype=None, weight_scale_dtype=None,
                                                     x_scale_dtype=None):
            torch._check(activation_type == "gelu_tanh",
                         lambda: f"activation_type only supports gelu_tanh, but got {activation_type}.")
            quant_mode = _resolve_quant_mode(quant_mode, x, x_dtype)
            torch._check(round_mode == "rint", lambda: f"round_mode only supports rint, but got {round_mode}.")
            torch._check(scale_alg in (0, 1), lambda: f"scale_alg only supports 0 or 1, but got {scale_alg}.")
            torch._check(group_list_type in (0, 1),
                         lambda: f"group_list_type only supports 0 or 1, but got {group_list_type}.")
            torch._check(len(weight) == 1 and weight[0] is not None,
                         lambda: "weight only supports one non-null tensor in current MXFP8 mode.")
            torch._check(len(weight_scale) == 1 and weight_scale[0] is not None,
                         lambda: "weight_scale only supports one non-null tensor in current MXFP8 mode.")
            torch._check(bias is None or len(bias) == 0,
                         lambda: "bias must be None or an empty tensor list in current MXFP8 mode.")
            torch._check(x_scale is not None, lambda: "x_scale must be provided in current MXFP8 mode.")
            torch._check(x.dim() == 2, lambda: f"x must be 2D, but got {x.dim()}D.")
            torch._check(weight[0].dim() == 5,
                         lambda: f"weight must be a 5D WeightNz tensor, but got {weight[0].dim()}D.")
            torch._check(weight_scale[0].dim() == 4,
                         lambda: f"weight_scale must be 4D, but got {weight_scale[0].dim()}D.")
            torch._check(x_scale.dim() == 3, lambda: f"x_scale must be 3D, but got {x_scale.dim()}D.")
            torch._check(group_list.dim() == 1, lambda: f"group_list must be 1D, but got {group_list.dim()}D.")

            m = x.shape[0]
            k = x.shape[1]
            e = group_list.shape[0]
            torch._check(1 <= e <= 1024, lambda: f"group_list first dim must be in [1, 1024], but got {e}.")
            torch._check(weight[0].shape[0] == e,
                         lambda: f"weight dim 0 must equal group number {e}, but got {weight[0].shape[0]}.")
            torch._check(weight_scale[0].shape[0] == e,
                         lambda: f"weight_scale dim 0 must equal group number {e}, but got {weight_scale[0].shape[0]}.")
            n = _infer_weight_layout(weight[0], weight_scale[0], e, k)
            torch._check(m == 0 or n == 0 or k > 0, lambda: "K must be greater than 0 unless M or N is 0.")
            torch._check(n % 64 == 0, lambda: f"N must be divisible by 64, but got {n}.")
            torch._check(x_scale.shape == (m, math.ceil(k / 64), 2),
                         lambda: f"x_scale shape must be {(m, math.ceil(k / 64), 2)}, but got {tuple(x_scale.shape)}.")

            y_dtype_value = _to_torch_dtype(y_dtype)
            y = torch.empty((m, n), dtype=y_dtype_value, device="meta")
            y_scale = torch.empty((m, math.ceil(n / 64), 2), dtype=FLOAT8_E8M0_DTYPE, device="meta")
            return (y, y_scale)


_grouped_matmul_activation_quant_op_builder = GroupedMatmulActivationQuantOpBuilder()
_op_module = _grouped_matmul_activation_quant_op_builder.load()


@impl(AS_LIBRARY, _grouped_matmul_activation_quant_op_builder.name, "PrivateUse1")
def _npu_grouped_matmul_activation_quant(x, weight, weight_scale, group_list,
                                         x_scale=None, bias=None, activation_type="gelu_tanh",
                                         group_list_type=0, tuning_config=None,
                                         quant_mode="", y_dtype=DEFAULT_Y_DTYPE,
                                         round_mode="rint", scale_alg=0,
                                         dst_type_max=0.0, x_dtype=None,
                                         weight_dtype=None, weight_scale_dtype=None,
                                         x_scale_dtype=None):
    return _op_module.npu_grouped_matmul_activation_quant(
        x, weight, weight_scale, group_list, x_scale, bias, activation_type, group_list_type, tuning_config,
        quant_mode, y_dtype, round_mode,
        scale_alg, dst_type_max, x_dtype, weight_dtype, weight_scale_dtype, x_scale_dtype)


def npu_grouped_matmul_activation_quant(x, weight, weight_scale, group_list, *,
                                        x_scale=None, bias=None, activation_type="gelu_tanh",
                                        group_list_type=0, tuning_config=None,
                                        quant_mode="", y_dtype=torch.float8_e4m3fn,
                                        round_mode="rint", scale_alg=0, dst_type_max=0.0,
                                        x_dtype=None, weight_dtype=None,
                                        weight_scale_dtype=None, x_scale_dtype=None):
    weight = _normalize_tensor_list(weight, "weight")
    weight_scale = _normalize_tensor_list(weight_scale, "weight_scale")
    bias = _normalize_bias(bias)
    y_dtype = _normalize_attr_dtype(y_dtype, DEFAULT_Y_DTYPE)
    x_dtype = _normalize_wrapper_dtype(x_dtype)
    weight_dtype = _normalize_wrapper_dtype(weight_dtype)
    weight_scale_dtype = _normalize_wrapper_dtype(weight_scale_dtype)
    x_scale_dtype = _normalize_wrapper_dtype(x_scale_dtype)
    quant_mode = _resolve_quant_mode(quant_mode, x, x_dtype)
    return torch.ops.cann_ops_transformer.npu_grouped_matmul_activation_quant(
        x, weight, weight_scale, group_list, x_scale=x_scale, bias=bias,
        activation_type=activation_type, group_list_type=group_list_type,
        tuning_config=tuning_config, quant_mode=quant_mode, y_dtype=y_dtype,
        round_mode=round_mode, scale_alg=scale_alg, dst_type_max=dst_type_max,
        x_dtype=x_dtype, weight_dtype=weight_dtype, weight_scale_dtype=weight_scale_dtype,
        x_scale_dtype=x_scale_dtype)
