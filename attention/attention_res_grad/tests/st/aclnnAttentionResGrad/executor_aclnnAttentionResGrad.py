#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# ----------------------------------------------------------------------------

from typing import Tuple

import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.api_execute.base_api import BaseApi


def _attn_res_bwd_core(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    grad_output: torch.Tensor,
    inv_rms: torch.Tensor,
    probs: torch.Tensor,
    use_matmul_out: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """CPU reference implementation for AttentionResGrad backward."""
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    hidden_size = v_float.shape[-1]

    inv_rms = inv_rms.float()
    if inv_rms.dim() == 2:
        inv_rms = inv_rms.unsqueeze(-1)
    probs = probs.float()
    if probs.dim() == 3:
        probs = probs.squeeze(1)

    k = v_float * inv_rms

    norm_w = norm_weight.float()
    proj_w = proj_weight.squeeze(0).float()
    score_weight = norm_w * proj_w

    grad_out = grad_output.float()

    if use_matmul_out:
        grad_out_u = grad_out.unsqueeze(1)
        probs_u = probs.unsqueeze(1)
        grad_probs = torch.matmul(grad_out_u, v_float.transpose(-1, -2)).squeeze(1)
        grad_v_from_out = torch.matmul(probs_u.transpose(-1, -2), grad_out_u)
    else:
        grad_probs = (grad_out.unsqueeze(1) * v_float).sum(-1)
        grad_v_from_out = grad_out.unsqueeze(1) * probs.unsqueeze(-1)

    grad_scores = torch.ops.aten._softmax_backward_data(
        grad_probs, probs, -1, grad_probs.dtype
    )

    grad_k = grad_scores.unsqueeze(-1) * score_weight
    grad_score_weight = (grad_scores.unsqueeze(-1) * k).sum(dim=(0, 1))

    grad_norm_weight = grad_score_weight * proj_w
    grad_proj_weight = (grad_score_weight * norm_w).unsqueeze(0)

    grad_v_from_k = grad_k * inv_rms
    grad_inv_rms = (grad_k * v_float).sum(-1, keepdim=True)

    grad_variance = grad_inv_rms * (-0.5) * inv_rms.pow(3)
    grad_v_from_var = (grad_variance / hidden_size) * (2.0 * v_float)

    grad_v = grad_v_from_out + grad_v_from_k + grad_v_from_var

    num_blocks = block_residual.shape[1]
    grad_block_residual = grad_v[:, :num_blocks, :].to(block_residual.dtype)
    grad_prefix_sum = grad_v[:, num_blocks, :].to(prefix_sum.dtype)

    return (
        grad_prefix_sum,
        grad_block_residual,
        grad_proj_weight.to(proj_weight.dtype),
        grad_norm_weight.to(norm_weight.dtype),
    )


@register("ascend_function_attention_res_grad")
class AttentionResGradCpuApi(BaseApi):
    """CPU benchmark executor for AttentionResGrad."""

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        return _attn_res_bwd_core(
            input_data.kwargs["prefix_sum"],
            input_data.kwargs["block_residual"],
            input_data.kwargs["proj_weight"],
            input_data.kwargs["norm_weight"],
            input_data.kwargs["grad_output"],
            input_data.kwargs["inv_rms"],
            input_data.kwargs["probs"],
        )


@register("aclnn_attention_res_grad")
class AttentionResGradAclnnApi(AclnnBaseApi):
    """NPU executor for aclnnAttentionResGrad."""

    def init_by_input_data(self, input_data: InputDataset):
        input_args = []
        output_packages = []

        for i, arg in enumerate(input_data.args):
            data = self.backend.convert_input_data(arg, index=i)
            input_args.extend(data)
        for name, kwarg in input_data.kwargs.items():
            data = self.backend.convert_input_data(kwarg, name=name)
            input_args.extend(data)

        dtype = input_data.kwargs["prefix_sum"].dtype
        for index, output_data in enumerate(self.task_result.output_info_list):
            output_data.dtype = str(dtype)
            output = self.backend.convert_output_data(output_data, index)
            output_packages.extend(output)

        input_args.extend(output_packages)
        return input_args, output_packages

    def call(self):
        super().call()

    def after_call(self, output_packages):
        outputs = []
        for output_pack in output_packages:
            output = self.acl_tensor_to_torch(output_pack)
            outputs.append(output)
        return tuple(outputs)

    def get_cpp_func_signature_type(self):
        return (
            "aclnnStatus aclnnAttentionResGradGetWorkspaceSize("
            "const aclTensor* prefix_sum, const aclTensor* block_residual, "
            "const aclTensor* proj_weight, const aclTensor* norm_weight, "
            "const aclTensor* grad_output, const aclTensor* inv_rms, "
            "const aclTensor* probs, const aclTensor* grad_prefix_sum, "
            "const aclTensor* grad_block_residual, "
            "const aclTensor* grad_proj_weight, "
            "const aclTensor* grad_norm_weight, "
            "uint64_t* workspaceSize, aclOpExecutor** executor)"
        )
