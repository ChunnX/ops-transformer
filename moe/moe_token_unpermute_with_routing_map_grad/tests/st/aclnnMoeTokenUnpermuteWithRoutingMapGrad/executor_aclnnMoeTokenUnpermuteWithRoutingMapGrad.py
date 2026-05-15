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

import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
import random
import ctypes
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor
from typing import Optional

@register("ascend_aclnn_moe_token_unpermute_with_routing_map_grad")
class FunctionMoeTokenUnpermuteWithRoutingMapGradApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionMoeTokenUnpermuteWithRoutingMapGradApi, self).__init__(task_result)
        self.sorted_indices = None
        self.torch_type = None
        self.prob_is_none = True

    def unpermute_grad_with_routing_map(
        self,
        device,
        unpermuted_tokens_grad, # unpermute正向结果梯度   [num_tokens, hidden]
        permuted_tokens, # permute输出结果 
        sorted_indices: torch.Tensor, # permute输出结果
        restore_shape: torch.Size,
        probs, # [num_tokens, num_experts]
        routing_map: torch.Tensor = None, # [num_tokens, num_experts]
        drop_and_pad: bool = False,
    ):
        if not self.prob_is_none:
            if drop_and_pad:
                num_tokens = probs.size(0)
                num_experts = probs.size(1)
                capacity = sorted_indices.size(0) // num_experts

                # get 1D indices of the probs selected by routing_map
                indices_dim0 = torch.arange(num_experts, device=routing_map.device).unsqueeze(-1).to(device)  # [num_experts, 1]
                indices_dim1 = sorted_indices.view(num_experts, capacity)  # [num_experts, capacity]
                indices_1D = (indices_dim0 * num_tokens + indices_dim1).view(-1)
                mul_grad = unpermuted_tokens_grad.index_select(0, sorted_indices)  # [num_experts * capacity, hidden]
                permuted_probs_grad = permuted_tokens * mul_grad  # [num_experts * capacity, hidden]
                permuted_probs_grad = permuted_probs_grad.sum(-1) # [num_experts * capacity,]
                probs_grad = torch.zeros([num_experts * num_tokens,], dtype=probs.dtype).to(device)
                probs_grad.index_copy_(0, indices_1D, permuted_probs_grad) # indices_1D值一定没有重复，这里直接分核不会有确定性问题
                probs_grad = probs_grad.reshape(num_experts, num_tokens).T # 转置
                # [num_tokens, num_experts] -> num_tokens * num_experts
                probs_T_1D = probs.T.contiguous().view(-1)  # [num_tokens * num_experts,]
                permuted_probs = probs_T_1D.index_select(0, indices_1D)  # [num_experts * capacity,]
                permuted_tokens_grad = mul_grad * permuted_probs.unsqueeze(-1)  # [num_experts * capacity, hidden]
                return permuted_tokens_grad, probs_grad
            else:
                num_tokens = probs.size(0)
                num_experts = probs.size(1)
                mul_grad = unpermuted_tokens_grad.index_select(0, sorted_indices) # [routing_map.sum(), hidden]
                permuted_probs_grad = permuted_tokens * mul_grad # [routing_map.sum(), hidden]
                permuted_probs_grad = permuted_probs_grad.sum(-1)
                probs_grad = torch.zeros((num_experts, num_tokens), dtype=probs.dtype).to(device)
                probs_grad.masked_scatter_(routing_map.T, permuted_probs_grad)
                probs_grad = probs_grad.T

                permuted_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous()) # [routing_map.sum(),]
                permuted_tokens_grad = mul_grad * permuted_probs.unsqueeze(-1) # [routing_map.sum(), hidden]
                return permuted_tokens_grad, probs_grad
        else: # prob为none，index_select
            permuted_tokens_grad = unpermuted_tokens_grad.index_select(0, sorted_indices)
            return permuted_tokens_grad, None

    def permute(
        self,
        tokens, # [num_tokens, hidden]
        routing_map, # [num_tokens, num_experts], bool
        probs: Optional[torch.Tensor] = None,
        num_out_tokens: Optional[int] = None,
        fused: bool = False,
        drop_and_pad: bool = False,
    ):
        num_tokens, hidden = tokens.shape
        num_experts = routing_map.shape[1]
        if drop_and_pad and not (num_out_tokens is None):
            capacity = num_out_tokens // num_experts
            assert not routing_map.requires_grad
            routing_map = routing_map.to(dtype=torch.int8).T.contiguous() # [num_experts, num_tokens]
            sorted_indices = routing_map.argsort(dim=-1, descending=True, stable=True)[
                :, :capacity
            ].contiguous() # [num_experts, capacity]
            sorted_indices = sorted_indices.view(-1) # [num_experts * capacity,]
            if not self.prob_is_none:
                routing_map = routing_map.bool() # [num_experts, num_tokens]
        else:
            routing_map = routing_map.bool().T.contiguous() # [num_experts, num_tokens]
            token_indices = (
                torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
            )  # [num_experts, num_tokens]
            sorted_indices = token_indices.masked_select(routing_map) # 1-D [routing_map.sum(0).sum(-1),]
        permuted_input = tokens.index_select(0, sorted_indices)
        if not self.prob_is_none:
            permuted_probs = probs.T.contiguous().masked_select(routing_map)
        else:
            permuted_probs = None
        return permuted_input, permuted_probs, sorted_indices
    
    # 生成值域在 [-data_max，data_max]之间，数据类型为data_type的torch tensor
    def generate_tensor(self,shape, data_type, data_max):
        tensor = torch.rand(shape) * (data_max * 2) - data_max
        return tensor.to(data_type)
    
    def generate_routing_map_drop_pad_false(self,topk,num_tokens,num_experts):
        tensor = torch.zeros((num_tokens, num_experts), dtype=torch.bool)
        for i in range(num_tokens):
            indices = torch.randperm(num_experts)[:topk]
            tensor[i, indices] = True
        return tensor
    

    def generate_routing_map_drop_pad_true(self,capacity,num_tokens,num_experts):
        tensor = torch.zeros((num_tokens, num_experts), dtype=torch.bool)
        for i in range(num_experts):
            indices = torch.randperm(num_tokens)[:capacity]
            tensor[indices, i] = True
        return tensor

    def init_by_input_data(self, input_data: InputDataset):
        num_tokens = input_data.kwargs['unpermutedTokensGrad'].shape[0]
        hidden_size = input_data.kwargs['unpermutedTokensGrad'].shape[1]
        num_experts = input_data.kwargs["routingMapOptional"].shape[1]
        drop_and_pad = input_data.kwargs["padded_mode"]
        num_out_tokens = input_data.kwargs["outIndex"].shape[0]
        self.torch_type = input_data.kwargs['unpermutedTokensGrad'].dtype

        case_id = self.task_result.case_config.id
        OpsDataset.seed_everything(case_id)
        prob_is_none_list = [True,False]
        self.prob_is_none = prob_is_none_list[case_id%2]

        routing_map = None
        if drop_and_pad:
            capacity = num_out_tokens // num_experts
            routing_map = self.generate_routing_map_drop_pad_true(capacity,num_tokens,num_experts)
        else:
            topK =  num_out_tokens // num_tokens
            routing_map = self.generate_routing_map_drop_pad_false(topK,num_tokens,num_experts)
        tokens = self.generate_tensor((num_tokens, hidden_size), torch.bfloat16, 5)
        probs = None
        if not self.prob_is_none:
            probs = self.generate_tensor((num_tokens, num_experts), torch.bfloat16, 5)
        permuted_tokens, _, sorted_indices = self.permute(tokens, routing_map, probs, num_out_tokens, False, drop_and_pad)
        self.sorted_indices = sorted_indices
        sorted_twice_indices, sorted_twice_index = torch.sort(sorted_indices, stable=True)
        unpermuted_tokens_grad = self.generate_tensor((num_tokens, hidden_size), torch.bfloat16, 5)
        restore_shape = tokens.shape

        if self.device == "pyaclnn":
            input_data.kwargs['unpermutedTokensGrad'] = unpermuted_tokens_grad.to(self.torch_type).npu()
            input_data.kwargs["outIndex"] = sorted_twice_index.to(torch.int32).npu()
            input_data.kwargs["permuteTokenId"] = sorted_twice_indices.to(torch.int32).npu()
            input_data.kwargs["routingMapOptional"] = routing_map.npu()
            input_data.kwargs["permutedTokensOptional"] = permuted_tokens.to(self.torch_type).npu()
            if not self.prob_is_none:
                input_data.kwargs["probsOptional"] = probs.to(self.torch_type).npu()
            else:
                input_data.kwargs["probsOptional"] = None
            input_data.kwargs["padded_mode"] = drop_and_pad
            input_data.kwargs["restore_shape"] = restore_shape
        else:
            input_data.kwargs['unpermutedTokensGrad'] = unpermuted_tokens_grad.to(torch.float32)
            input_data.kwargs["outIndex"] = sorted_twice_index.to(torch.int32)
            input_data.kwargs["permuteTokenId"] = sorted_twice_indices.to(torch.int32)
            input_data.kwargs["routingMapOptional"] = routing_map
            input_data.kwargs["permutedTokensOptional"] = permuted_tokens.to(torch.float32)
            if not self.prob_is_none:
                input_data.kwargs["probsOptional"] = probs.to(torch.float32)
            else:
                input_data.kwargs["probsOptional"] = None
            input_data.kwargs["padded_mode"] = drop_and_pad
            input_data.kwargs["restore_shape"] = restore_shape
    
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        device = "cpu"
        unpermuted_tokens_grad = input_data.kwargs['unpermutedTokensGrad'].to(device)
        sorted_twice_index = input_data.kwargs["outIndex"].to(device)
        sorted_twice_indices = input_data.kwargs["permuteTokenId"].to(device)
        routing_map = input_data.kwargs["routingMapOptional"].to(device)
        permuted_tokens = input_data.kwargs["permutedTokensOptional"].to(device)
        probs = None
        if not self.prob_is_none:
            probs = input_data.kwargs["probsOptional"].to(device)
        padded_mode = input_data.kwargs["padded_mode"]
        restore_shape = input_data.kwargs["restore_shape"] 

        if not self.prob_is_none:
            probs = probs.to(torch.float32)
            golden_res = self.unpermute_grad_with_routing_map(device, unpermuted_tokens_grad,permuted_tokens,self.sorted_indices.to(device),restore_shape,probs,routing_map,padded_mode)
            return golden_res[0].to(self.torch_type), golden_res[1].to(self.torch_type).contiguous()
        else:
            golden_res = self.unpermute_grad_with_routing_map(device, unpermuted_tokens_grad,permuted_tokens,self.sorted_indices.to(device),restore_shape,probs,routing_map,padded_mode)
            return golden_res[0].to(self.torch_type),torch.tensor([0]).to(torch.float32).contiguous().to(device)

@register("AclnnBaseApi_aclnn_moe_token_unpermute_with_routing_map_grad")
class AclnnMoeTokenUnpermuteWithRoutingMapGradApi(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        case_id = self.task_result.case_config.id
        prob_is_none_list = [True,False]
        self.prob_is_none = prob_is_none_list[case_id%2]
        if self.prob_is_none:
            AclTensorPtr = ctypes.POINTER(AclTensor)
            null_tensor_ptr = ctypes.cast(input_args[5], AclTensorPtr)
            input_args[5] = null_tensor_ptr
        return input_args, output_packages

    
    def after_call(self, output_packages):
        output = []
        torch_tensor = self.acl_tensor_to_torch(output_packages[0])
        output.append(torch_tensor)
        if not self.prob_is_none:
            output.append(self.acl_tensor_to_torch(output_packages[1]))
        else:
            output.append(torch.tensor([0]).to(torch.float32).contiguous())
        return output