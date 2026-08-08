#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig


@GENERATOR_REGISTRY.register("attention_res_grad")
class AttentionResGradGenerator(CaseGenerator):
    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        b = case_config.inputs[0].shape[0]
        n = case_config.inputs[1].shape[1]
        h = case_config.inputs[0].shape[1]

        dtype = case_config.inputs[0].dtype
        # inv_rms/probs 在算子定义中固定为 fp32，不能跟随 prefix_sum 的 fp16/bf16。
        for input_config in case_config.inputs[:5]:
            input_config.dtype = dtype

        case_config.inputs[0].shape = [b, h]
        case_config.inputs[1].shape = [b, n, h]
        case_config.inputs[2].shape = [1, h]
        case_config.inputs[3].shape = [h]
        case_config.inputs[4].shape = [b, h]
        case_config.inputs[5].shape = [b, n + 1]
        case_config.inputs[6].shape = [b, n + 1]
        return case_config
