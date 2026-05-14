#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import random

import torch

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset


PADDED = "padded"
NON_PADDED = "non_padded"
NON_PADDED_MINUS_ONE = "non_padded_minus_one"


def case_mode(case_id):
    modes = (PADDED, NON_PADDED, NON_PADDED_MINUS_ONE)
    return modes[case_id % len(modes)]


def safe_div(a, b):
    if b == 0:
        return 0
    return a // b


def generate_tensor(shape, data_type, data_max=5):
    tensor = torch.rand(shape) * (2 * data_max) - data_max
    return tensor.to(data_type)


def has_special_values(tensor):
    if tensor is None:
        return False
    return torch.any(torch.isinf(tensor)) or torch.any(torch.isnan(tensor))


def generate_routing_map_padded(capacity, num_tokens, num_experts):
    routing_map = torch.zeros((num_tokens, num_experts), dtype=torch.bool)
    for expert_idx in range(num_experts):
        indices = torch.randperm(num_tokens)[:capacity]
        routing_map[indices, expert_idx] = True
    return routing_map


def generate_non_padded_case(num_tokens, num_experts, topk, prob_dtype, with_minus_one):
    routing_map = torch.zeros((num_tokens, num_experts), dtype=torch.bool)
    probs = torch.zeros((num_tokens, num_experts), dtype=prob_dtype)
    sorted_indices = torch.full((num_tokens * topk,), -1, dtype=torch.int32)

    permuted_row = 0
    for token_idx in range(num_tokens):
        selected_experts = torch.randperm(num_experts)[:topk].sort().values
        for topk_idx, expert_idx_tensor in enumerate(selected_experts):
            expert_idx = int(expert_idx_tensor.item())
            slot = token_idx * topk + topk_idx
            routing_map[token_idx, expert_idx] = True

            make_invalid = with_minus_one and (
                (token_idx % 4 == 0 and topk_idx == 0)
                or (token_idx % 4 == 1 and topk_idx == topk - 1)
                or (token_idx % 4 == 3 and topk_idx % 2 == 0)
            )
            if make_invalid:
                probs[token_idx, expert_idx] = torch.tensor(0, dtype=prob_dtype)
                continue

            sorted_indices[slot] = permuted_row
            probs[token_idx, expert_idx] = (torch.rand((), dtype=torch.float32) * 2 - 1).to(prob_dtype)
            permuted_row += 1

    return routing_map, probs, sorted_indices


def permute_for_padded(tokens, routing_map, probs, num_out_tokens):
    num_tokens = tokens.shape[0]
    num_experts = routing_map.shape[1]
    capacity = safe_div(num_out_tokens, num_experts)
    routing_map_int8 = routing_map.to(dtype=torch.int8).T.contiguous()
    sorted_indices = routing_map_int8.argsort(dim=-1, descending=True, stable=True)[:, :capacity]
    sorted_indices = sorted_indices.contiguous().view(-1)
    permuted_tokens = tokens.index_select(0, sorted_indices)
    permuted_probs = probs.T.contiguous().masked_select(routing_map_int8.bool())
    return permuted_tokens, permuted_probs, sorted_indices, routing_map


def unpermute_padded(permuted_tokens, sorted_indices, probs, restore_shape, is_mixed, compute_dtype, output_dtype):
    original_dtype = permuted_tokens.dtype
    if is_mixed:
        permuted_tokens = permuted_tokens.to(compute_dtype)
        probs = probs.to(compute_dtype)

    num_tokens, _ = restore_shape
    num_experts = probs.size(1)
    capacity = sorted_indices.size(0) // num_experts

    indices_token = sorted_indices.view(num_experts, capacity)
    expert_indices = torch.arange(num_experts, device=permuted_tokens.device).unsqueeze(-1)
    indices_1d = (expert_indices * num_tokens + indices_token).view(-1)
    indices_1d = torch.clamp(indices_1d, 0, num_experts * num_tokens - 1)

    permute_probs = probs.T.contiguous().view(-1)[indices_1d]
    weighted_tokens = permuted_tokens * permute_probs.unsqueeze(-1)

    sorted_values, sorted_order = torch.sort(sorted_indices, dim=-1, stable=True, descending=False)
    output_tokens = torch.zeros(restore_shape, dtype=compute_dtype, device=permuted_tokens.device)
    output_tokens.index_add_(0, sorted_values, weighted_tokens[sorted_order])

    if is_mixed:
        output_tokens = output_tokens.to(output_dtype)
        permute_probs = permute_probs.to(output_dtype)

    return (
        output_tokens.to(dtype=original_dtype),
        sorted_order.to(dtype=sorted_indices.dtype),
        sorted_values.to(dtype=sorted_indices.dtype),
        permute_probs.to(dtype=probs.dtype),
    )


def unpermute_non_padded(permuted_tokens, sorted_indices, routing_map, probs, restore_shape):
    num_tokens = restore_shape[0]
    topk = safe_div(sorted_indices.numel(), num_tokens)
    compute_dtype = torch.float32 if permuted_tokens.dtype in (torch.float16, torch.bfloat16) else permuted_tokens.dtype
    output_tokens = torch.zeros(restore_shape, dtype=compute_dtype, device=permuted_tokens.device)

    permute_probs = probs.masked_select(routing_map.to(torch.bool)).to(compute_dtype)
    for token_idx in range(num_tokens):
        for topk_idx in range(topk):
            slot = token_idx * topk + topk_idx
            permuted_idx = int(sorted_indices[slot].item())
            if permuted_idx < 0 or permuted_idx >= permuted_tokens.shape[0]:
                continue
            output_tokens[token_idx] += permuted_tokens[permuted_idx].to(compute_dtype) * permute_probs[slot]

    dummy_index = torch.tensor([1], dtype=sorted_indices.dtype, device=permuted_tokens.device)
    return output_tokens.to(permuted_tokens.dtype), dummy_index, dummy_index, permute_probs.to(probs.dtype)


@register("ascend_function_moe_token_unpermute_with_routing_map")
class FunctionMoeTokenUnpermuteWithRoutingMapApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)
        self.mode = PADDED
        self.is_mixed = False
        self.compute_dtype = None
        self.output_dtype = None

    def init_by_input_data(self, input_data: InputDataset):
        case_id = self.task_result.case_config.id
        OpsDataset.seed_everything(case_id)
        random.seed(case_id)
        self.mode = case_mode(case_id)

        if self.mode == PADDED:
            self.init_padded(input_data)
        else:
            self.init_non_padded(input_data, self.mode == NON_PADDED_MINUS_ONE)

    def init_padded(self, input_data: InputDataset):
        permuted_tokens_input = input_data.kwargs["permutedTokens"]
        probs_input = input_data.kwargs["probsOptional"]
        sorted_indices_input = input_data.kwargs["sortedIndices"]
        restore_shape_input = input_data.kwargs["restoreShapeOptional"]

        token_dtype = permuted_tokens_input.dtype
        prob_dtype = probs_input.dtype
        hidden_size = restore_shape_input[1]
        num_tokens, num_experts = probs_input.shape
        num_out_tokens = sorted_indices_input.shape[0]
        capacity = safe_div(num_out_tokens, num_experts)

        self.is_mixed = token_dtype in (torch.bfloat16, torch.float16) and prob_dtype == torch.float32
        self.compute_dtype = torch.float32 if self.is_mixed else token_dtype
        self.output_dtype = token_dtype

        if has_special_values(permuted_tokens_input) or has_special_values(probs_input):
            permuted_tokens = permuted_tokens_input
            probs = probs_input.to(torch.float32 if self.is_mixed else token_dtype)
            sorted_indices = sorted_indices_input.to(torch.int32)
            routing_map = input_data.kwargs["routingMapOptional"]
            restore_shape = restore_shape_input
        else:
            original_tokens = generate_tensor((num_tokens, hidden_size), token_dtype, data_max=5)
            routing_map = generate_routing_map_padded(capacity, num_tokens, num_experts)
            probs = generate_tensor(
                (num_tokens, num_experts),
                torch.float32 if self.is_mixed else prob_dtype,
                data_max=1.0,
            )
            permuted_tokens, _, sorted_indices, routing_map = permute_for_padded(
                original_tokens, routing_map, probs, num_out_tokens
            )
            restore_shape = [num_tokens, hidden_size]

        probs_dtype_for_acl = torch.float32 if self.is_mixed else token_dtype
        if self.device == "pyaclnn":
            input_data.kwargs["permutedTokens"] = permuted_tokens.to(token_dtype).npu()
            input_data.kwargs["sortedIndices"] = sorted_indices.to(torch.int32).npu()
            input_data.kwargs["routingMapOptional"] = routing_map.npu()
            input_data.kwargs["probsOptional"] = probs.to(probs_dtype_for_acl).npu()
        else:
            input_data.kwargs["permutedTokens"] = permuted_tokens.to(token_dtype)
            input_data.kwargs["sortedIndices"] = sorted_indices.to(torch.int32)
            input_data.kwargs["routingMapOptional"] = routing_map
            input_data.kwargs["probsOptional"] = probs.to(probs_dtype_for_acl)

        input_data.kwargs["paddedMode"] = True
        input_data.kwargs["restoreShapeOptional"] = restore_shape

    def init_non_padded(self, input_data: InputDataset, with_minus_one):
        permuted_tokens_input = input_data.kwargs["permutedTokens"]
        routing_map_input = input_data.kwargs["routingMapOptional"]
        probs_input = input_data.kwargs["probsOptional"]

        token_dtype = permuted_tokens_input.dtype
        prob_dtype = probs_input.dtype
        routing_dtype = routing_map_input.dtype
        num_tokens = max(1, min(int(routing_map_input.shape[0]), 128))
        num_experts = max(1, min(int(routing_map_input.shape[1]), 64))
        hidden_size = max(1, min(int(permuted_tokens_input.shape[1]), 512))
        topk = min(16, num_experts)
        num_out_tokens = num_tokens * topk

        self.compute_dtype = torch.float32 if token_dtype in (torch.float16, torch.bfloat16) else token_dtype

        routing_map, probs, sorted_indices = generate_non_padded_case(
            num_tokens, num_experts, topk, prob_dtype, with_minus_one
        )
        permuted_tokens = generate_tensor((num_out_tokens, hidden_size), token_dtype, data_max=5)
        restore_shape = [num_tokens, hidden_size]

        if self.device == "pyaclnn":
            input_data.kwargs["permutedTokens"] = permuted_tokens.npu()
            input_data.kwargs["sortedIndices"] = sorted_indices.npu()
            input_data.kwargs["routingMapOptional"] = routing_map.to(routing_dtype).npu()
            input_data.kwargs["probsOptional"] = probs.npu()
        else:
            input_data.kwargs["permutedTokens"] = permuted_tokens
            input_data.kwargs["sortedIndices"] = sorted_indices
            input_data.kwargs["routingMapOptional"] = routing_map.to(routing_dtype)
            input_data.kwargs["probsOptional"] = probs

        input_data.kwargs["paddedMode"] = False
        input_data.kwargs["restoreShapeOptional"] = restore_shape

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "gpu":
            device = f"cuda:{self.device_id}"
        elif self.device == "npu":
            device = f"{self.device}:{self.device_id}"
        else:
            device = "cpu"

        permuted_tokens = input_data.kwargs["permutedTokens"].to(device)
        sorted_indices = input_data.kwargs["sortedIndices"].to(device)
        routing_map = input_data.kwargs["routingMapOptional"].to(device)
        probs = input_data.kwargs["probsOptional"].to(device)
        restore_shape = input_data.kwargs["restoreShapeOptional"]

        if self.mode == PADDED:
            return unpermute_padded(
                permuted_tokens, sorted_indices, probs, restore_shape,
                self.is_mixed, self.compute_dtype, self.output_dtype
            )
        return unpermute_non_padded(permuted_tokens, sorted_indices, routing_map, probs, restore_shape)


@register("aclnn_function")
class AclnnMoeTokenUnpermuteWithRoutingMapApi(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        self.mode = case_mode(self.task_result.case_config.id)
        return super().init_by_input_data(input_data)

    def after_call(self, output_packages):
        if self.mode == PADDED:
            return [self.acl_tensor_to_torch(output_package) for output_package in output_packages]

        output_tokens = self.acl_tensor_to_torch(output_packages[0])
        return [
            output_tokens,
            torch.tensor([1], dtype=torch.int32, device=output_tokens.device),
            torch.tensor([1], dtype=torch.int32, device=output_tokens.device),
            self.acl_tensor_to_torch(output_packages[3]),
        ]
