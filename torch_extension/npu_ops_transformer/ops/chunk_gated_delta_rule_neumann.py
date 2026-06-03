import torch
import torch_npu
from torch.library import impl

from npu_ops_transformer.op_builder.builder import AS_LIBRARY
from npu_ops_transformer.op_builder.builder import OpBuilder


class ChunkGatedDeltaRuleNeumannOpBuilder(OpBuilder):
    def __init__(self):
        super(ChunkGatedDeltaRuleNeumannOpBuilder, self).__init__("npu_chunk_gated_delta_rule_neumann")

    def sources(self):
        return ["ops/csrc/chunk_gated_delta_rule_neumann.cpp"]

    def schema(self) -> str:
        return (
            "npu_chunk_gated_delta_rule_neumann(Tensor query, Tensor key, Tensor value, Tensor beta, "
            "Tensor initial_state, Tensor actual_seq_lengths, Tensor? g=None, float scale=1.0) "
            "-> (Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_chunk_gated_delta_rule_neumann_meta(query, key, value, beta, initial_state, actual_seq_lengths, g=None,
                                                    scale=1.0):
            out = value.new_empty(value.shape)
            final_state = initial_state.new_empty(initial_state.shape)
            return out, final_state


chunk_gated_delta_rule_neumann_op_builder = ChunkGatedDeltaRuleNeumannOpBuilder()
op_module = chunk_gated_delta_rule_neumann_op_builder.load()


@impl(AS_LIBRARY, chunk_gated_delta_rule_neumann_op_builder.name, "PrivateUse1")
def _npu_chunk_gated_delta_rule_neumann_impl(query, key, value, beta, initial_state, actual_seq_lengths, g=None,
                                             scale=1.0):
    return op_module.npu_chunk_gated_delta_rule_neumann(query, key, value, beta, initial_state, actual_seq_lengths, g,
                                                        scale)


def npu_chunk_gated_delta_rule_neumann(query, key, value, *, beta, initial_state, actual_seq_lengths, scale=1.0,
                                       g=None):
    return torch.ops.npu_ops_transformer.npu_chunk_gated_delta_rule_neumann(
        query, key, value, beta, initial_state, actual_seq_lengths, g, scale
    )


torch_npu.npu_chunk_gated_delta_rule_neumann = npu_chunk_gated_delta_rule_neumann
