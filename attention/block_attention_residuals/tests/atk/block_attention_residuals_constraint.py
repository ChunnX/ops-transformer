"""ATK constraints for the fixed default BlockAttentionResiduals case."""

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig


@GENERATOR_REGISTRY.register("block_attention_residuals_constraint")
class BlockAttentionResidualsConstraintGenerator(CaseGenerator):
    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        return case_config
