"""ATK constraints for BlockAttentionResiduals supported default scenarios.

The operator currently exposes the forward/default path only.  This generator
keeps the ATK-generated dtype and shape equivalence classes, then makes the
cross-input relationships explicit before cases are serialized.
"""

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig


_TENSOR_NAMES = {"prefix_sum", "block_res", "proj_weight", "norm_weight"}


def _dim(shape, index, default):
    return shape[index] if len(shape) > index and isinstance(shape[index], int) else default


@GENERATOR_REGISTRY.register("block_attention_residuals_constraint")
class BlockAttentionResidualsConstraintGenerator(CaseGenerator):
    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        source_shapes = {
            inp.name: list(inp.shape or [])
            for inp in case_config.inputs
            if inp.name in _TENSOR_NAMES
        }
        prefix_shape = source_shapes.get("prefix_sum", [1, 64])
        block_shape = source_shapes.get("block_res", [1, 1, 64])
        proj_shape = source_shapes.get("proj_weight", [1, 64])

        # C++ shape checks: prefix_sum=[T,H], block_res=[T,N,H].
        tokens = max(0, _dim(prefix_shape, 0, 1))
        hidden = max(1, _dim(prefix_shape, 1, 64))
        blocks = min(100, max(1, _dim(block_shape, 1, 1)))
        proj_rank = 2 if len(proj_shape) == 2 else 1
        shapes = {
            "prefix_sum": [tokens, hidden],
            "block_res": [tokens, blocks, hidden],
            "proj_weight": [1, hidden] if proj_rank == 2 else [hidden],
            "norm_weight": [hidden],
        }

        # All primary tensors must have the same dtype.
        dtype = next(
            (inp.dtype for inp in case_config.inputs
             if inp.name in _TENSOR_NAMES and inp.dtype),
            "bf16",
        )
        for inp in case_config.inputs:
            if inp.name in shapes:
                inp.shape = shapes[inp.name]
                inp.dtype = dtype
            elif inp.name == "valid_block_num":
                inp.range_values = blocks
            elif inp.name == "norm_eps":
                inp.range_values = 1.0e-5
            elif inp.name == "need_backward":
                inp.range_values = False
        return case_config
