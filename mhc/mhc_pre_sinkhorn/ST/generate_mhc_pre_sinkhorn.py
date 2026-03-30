# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
import random
from typing import Union, List

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import InputCaseConfig, CaseConfig


@GENERATOR_REGISTRY.register("ascend_generate_mhc_pre_sinkhorn")
class MhcPreSinkhornGenerator(CaseGenerator):

    def __init__(self, config):
        super().__init__(config)
        self.tensor_dim = 0
        self.range_is_null = False

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        h_res_shape = case_config.inputs[0].shape
        
        # Ensure the last two dimensions are equal and in {4, 6, 8}
        n = h_res_shape[-1]
        if n not in [4, 6, 8]:
            # Round to nearest valid n value
            valid_n_values = [4, 6, 8]
            n = min(valid_n_values, key=lambda x: abs(x - n))
            h_res_shape = list(h_res_shape)
            h_res_shape[-1] = n
            h_res_shape[-2] = n
            h_res_shape = tuple(h_res_shape)
        
        case_config.inputs[0].shape = h_res_shape
        
        return case_config
