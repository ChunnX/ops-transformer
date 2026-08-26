"""ATK constraints for the fixed default BlockAttentionResiduals case."""

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig


@GENERATOR_REGISTRY.register("block_attention_residuals_constraint")
class BlockAttentionResidualsConstraintGenerator(CaseGenerator):
    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        shapes = {
            "prefix_sum": [544, 7168],
            "block_res": [544, 12, 7168],
            "proj_weight": [1, 7168],
            "norm_weight": [7168],
        }
        for inp in case_config.inputs:
            if inp.name in shapes:
                inp.shape = shapes[inp.name]
        return case_config
