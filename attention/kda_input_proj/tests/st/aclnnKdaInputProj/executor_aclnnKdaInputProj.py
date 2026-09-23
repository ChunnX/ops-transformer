#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import sys
from pathlib import Path

import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.api_execute.base_api import BaseApi

_PYTEST_DIR = Path(__file__).resolve().parents[2] / "pytest"
if str(_PYTEST_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTEST_DIR))

from kda_input_proj_bgg_golden import bgg_golden
from kda_input_proj_mx_quant_golden import mx_quant_golden
from kda_input_proj_qkv_golden import qkv_golden

ACLNN_GET_WORKSPACE_SIGNATURE = (
    "aclnnStatus aclnnKdaInputProjGetWorkspaceSize("
    "const aclTensor *x, const aclTensor *weightQkv, const aclTensor *weightBeta, "
    "const aclTensor *weightGate, const aclTensor *weightG, const aclTensor *weightQkvScale, "
    "const aclTensor *qkvOut, const aclTensor *betaOut, const aclTensor *gateOut, "
    "const aclTensor *gOut, uint64_t *workspaceSize, aclOpExecutor **executor)"
)

_PREPARED_KEY = "_kda_prepared"


def _to_fp8_e4m3(weight: torch.Tensor) -> torch.Tensor:
    if weight.dtype == torch.float8_e4m3fn:
        return weight
    if weight.dtype == torch.uint8:
        return weight.view(torch.float8_e4m3fn)
    return weight.to(torch.float32).to(torch.float8_e4m3fn)


def _to_e8m0(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype == torch.float8_e8m0fnu:
        return scale
    if scale.dtype == torch.uint8:
        return scale.view(torch.float8_e8m0fnu)
    codes = scale.to(torch.int32).clamp(0, 255).to(torch.uint8)
    return codes.view(torch.float8_e8m0fnu)


def _maybe_trans_kn(weight: torch.Tensor, hidden_size: int) -> torch.Tensor:
    """公开 view 是 [K, N]。若 ATK 按 Linear [N, K] 造数，转成 stride[-2]==1 的 [K, N] view。"""
    if (
        weight.dim() == 2
        and weight.shape[0] != hidden_size
        and weight.shape[1] == hidden_size
    ):
        return weight.transpose(-2, -1)
    return weight


def prepare_kda_inputs(kwargs: dict) -> dict:
    """把 ATK 可生成的 bf16/int32 转成算子真实 dtype，并按 stride 表达 trans。"""
    if kwargs.get(_PREPARED_KEY):
        return kwargs

    x = kwargs["x"].to(torch.bfloat16)
    hidden_size = int(x.shape[-1])

    weight_qkv = _to_fp8_e4m3(_maybe_trans_kn(kwargs["weightQkv"], hidden_size))
    weight_beta = _maybe_trans_kn(kwargs["weightBeta"], hidden_size).to(torch.bfloat16)
    weight_gate = _maybe_trans_kn(kwargs["weightGate"], hidden_size).to(torch.bfloat16)
    weight_g = _maybe_trans_kn(kwargs["weightG"], hidden_size).to(torch.bfloat16)
    weight_qkv_scale = _to_e8m0(kwargs["weightQkvScale"])

    kwargs["x"] = x
    kwargs["weightQkv"] = weight_qkv
    kwargs["weightBeta"] = weight_beta
    kwargs["weightGate"] = weight_gate
    kwargs["weightG"] = weight_g
    kwargs["weightQkvScale"] = weight_qkv_scale
    kwargs[_PREPARED_KEY] = True
    return kwargs


def _scale_to_kn(
    weight_qkv: torch.Tensor, weight_qkv_scale: torch.Tensor
) -> torch.Tensor:
    """qkv golden 要 KN 排布的 scale：[K/64, N, 2]。NK 存储则 permute。"""
    hidden_size = int(weight_qkv.shape[0])
    n_qkv = int(weight_qkv.shape[1])
    group_k = (hidden_size + 63) // 64
    if tuple(weight_qkv_scale.shape) == (n_qkv, group_k, 2):
        return weight_qkv_scale.permute(1, 0, 2).contiguous()
    return weight_qkv_scale


def kda_input_proj_golden(kwargs: dict):
    kwargs = prepare_kda_inputs(kwargs)
    x = kwargs["x"]
    weight_qkv = kwargs["weightQkv"]
    weight_beta = kwargs["weightBeta"]
    weight_gate = kwargs["weightGate"]
    weight_g = kwargs["weightG"]
    weight_qkv_scale = kwargs["weightQkvScale"]

    quant_bytes, scale_bytes = mx_quant_golden(x)
    quant_x = quant_bytes.view(torch.float8_e4m3fn)
    x_scale = scale_bytes.view(torch.float8_e8m0fnu)
    scale_kn = _scale_to_kn(weight_qkv, weight_qkv_scale)
    qkv = qkv_golden(quant_x, x_scale, weight_qkv, scale_kn)
    beta, gate, g = bgg_golden(x, weight_beta, weight_gate, weight_g)
    return qkv, beta, gate, g


@register("ascend_kda_input_proj")
class AclnnKdaInputProjApi(BaseApi):
    def init_by_input_data(self, input_data: InputDataset, with_output: bool = False):
        prepare_kda_inputs(input_data.kwargs)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "cpu":
            return kda_input_proj_golden(input_data.kwargs)
        return None

    def get_cpp_func_signature_type(self):
        return ACLNN_GET_WORKSPACE_SIGNATURE


@register("ascend_kda_input_proj_npu")
class AclnnKdaInputProjApiNpu(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        prepare_kda_inputs(input_data.kwargs)
        input_data.kwargs.pop(_PREPARED_KEY, None)
        return super().init_by_input_data(input_data)

    def get_cpp_func_signature_type(self):
        return ACLNN_GET_WORKSPACE_SIGNATURE
