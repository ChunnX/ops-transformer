# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import random
from typing import Union, List
import numpy as np
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


@GENERATOR_REGISTRY.register("ascend_generate_mhc_pre_sinkhorn")
class MhcPreSinkhornGenerator(CaseGenerator):

    def __init__(self, config):
        super().__init__(config)

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        x_shape = case_config.inputs[0].shape
        
        bs = x_shape[0]
        seq_len = x_shape[1]
        n = x_shape[2]
        c = x_shape[3]
        
        n = 4 

        c = np.random.choice([1280, 1920, 2560])
        
        hc_mix = n * n + 2 * n
        
        case_config.inputs[0].shape = [bs, seq_len, n, c]

        case_config.inputs[1].shape = [hc_mix, n * c]
        
        case_config.inputs[2].shape = [3]
        
        case_config.inputs[3].shape = [hc_mix]
        
        case_config.inputs[4].range_values = n
        
        return case_config
