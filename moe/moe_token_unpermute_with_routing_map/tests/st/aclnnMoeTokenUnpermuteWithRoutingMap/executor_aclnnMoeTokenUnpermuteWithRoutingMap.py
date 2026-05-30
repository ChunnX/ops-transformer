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
import numpy as np
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset


@register("ascend_function_moe_token_unpermute_with_routing_map")
class FunctionMoeTokenUnpermuteWithRoutingMapApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionMoeTokenUnpermuteWithRoutingMapApi, self).__init__(task_result)

    def unpermute_with_routing_map(self, shuffled_tokens, ordered_indices, routing_info, weights, padding_mode, original_shape):
        ordered_index, ordered_indices1 = ordered_indices.sort(dim=-1, stable=True, descending=False)
        if weights is not None:
            if padding_mode:
                num_experts = weights.size(1)
                num_tokens = weights.size(0)

                weight_trans = weights.T.contiguous().view(-1)
                indices_expert = torch.arange(num_experts, device=routing_info.device).unsqueeze(-1)
                
                feature_dim = shuffled_tokens.size(1)
                capacity = ordered_indices.size(0) // num_experts
                indices_token = ordered_indices.view(num_experts, capacity)
                indices = (indices_expert * num_tokens + indices_token).view(-1)
                shuffled_weights = weight_trans.index_select(0, indices)
            else:
                shuffled_weights = weights.T.contiguous().masked_select(routing_info.T.contiguous())

            shuffled_tokens = shuffled_tokens * shuffled_weights.unsqueeze(-1)

        output_tokens = torch.zeros(
            original_shape, dtype=shuffled_tokens.dtype, device=shuffled_tokens.device
        )
        for i in range(ordered_index.size(-1)):
            output_tokens[ordered_index[i]] += shuffled_tokens[ordered_indices1[i]]

        return output_tokens.to(dtype=shuffled_tokens.dtype), ordered_indices1.to(dtype=ordered_indices.dtype), ordered_index.to(dtype=ordered_indices.dtype), shuffled_weights.to(dtype=shuffled_tokens.dtype)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        permuted_tokens = input_data.kwargs["permutedTokens"]
        sorted_indices = input_data.kwargs["sortedIndices"]
        routing_map = input_data.kwargs["routingMapOptional"]
        probs = input_data.kwargs["probsOptional"]
        paddedMode = True
        restoreShapeOptional = input_data.kwargs["restoreShapeOptional"]
        cpu_res = self.unpermute_with_routing_map(permuted_tokens, sorted_indices, routing_map, probs, paddedMode, restoreShapeOptional)

        return cpu_res[0], cpu_res[1], cpu_res[2], cpu_res[3]