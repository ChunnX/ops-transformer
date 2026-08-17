#!/usr/bin/python
# -*- coding: utf-8 -*-
import random

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


@GENERATOR_REGISTRY.register("generate_lightning_indexer_kl_loss")
class LightningIndexerKLLossGenerator(CaseGenerator):
    # 每维独立区间（YAML 只能配单区间，逐维差异化在此实现）
    # 2 维：dim0 ∈ [1, 80000], dim1 ∈ [1, 8192]
    # 3 维：dim0 ∈ [1, 512],    dim1 ∈ [1, 80000], dim2 ∈ [1, 8192]

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        shape = list(case_config.inputs[0].shape)
        if len(shape) == 2:
            # 2 维：第 0 维保持 YAML 值，第 1 维 clamp 到 [1, 8192]
            shape[1] = min(shape[1], 8192)
        elif len(shape) == 3:
            # 3 维：第 0 维 <= 512，第 1 维保持，第 2 维 <= 8192
            shape[0] = min(shape[0], 512)
            shape[2] = min(shape[2], 8192)
        
        # indexProbs shape 必须与 targetScore 完全一致
        case_config.inputs[1].shape = list(case_config.inputs[0].shape)
        # indexProbs dtype 与 targetScore 一致
        case_config.inputs[1].dtype = case_config.inputs[0].dtype
        return case_config









