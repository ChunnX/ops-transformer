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

from typing import Union, List
import random
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import InputCaseConfig, CaseConfig

@GENERATOR_REGISTRY.register("ascend_generate_moe_token_unpermute_with_routing_map_grad")
class MoeTokenUnpermuteWithRoutingMapGradGenerator(CaseGenerator):
    def __init__(self, config):
        super().__init__(config)

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        token_num = random.randint(16,4096)
        hidden_size = random.randint(1024,7168)
        expert_num = random.randint(16,256)
        top_k = random.randint(1,16)
        capacity = random.randint(1,16)

        case_config.inputs[0].shape = [token_num, hidden_size]
        case_config.inputs[3].shape = [token_num, expert_num]
        case_config.inputs[5].shape = [token_num, expert_num]

        padded_mode = case_config.inputs[6].range_values
        is_mix = case_config.inputs[8].range_values
        token_type = case_config.inputs[0].dtype
        case_config.inputs[9].range_values = token_type
            
        if is_mix and token_type == "bf16":
            case_config.inputs[5].dtype = "fp32"
        else:
            case_config.inputs[5].dtype = token_type
        case_config.inputs[4].dtype = token_type
        if padded_mode:
            case_config.inputs[1].shape = [expert_num * capacity]
            case_config.inputs[2].shape = [expert_num * capacity]
            case_config.inputs[4].shape = [expert_num * capacity, hidden_size]
        else:
            case_config.inputs[1].shape = [token_num * top_k]
            case_config.inputs[2].shape = [token_num * top_k]
            case_config.inputs[4].shape = [token_num * top_k, hidden_size]
        return case_config