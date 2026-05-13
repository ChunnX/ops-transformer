import torch
import torch_npu
from torch.library import impl

from npu_ops_transformer.op_builder.builder import AS_LIBRARY
from npu_ops_transformer.op_builder.builder import OpBuilder


class TriInverseOpBuilder(OpBuilder):
    def __init__(self):
        super(TriInverseOpBuilder, self).__init__("npu_tri_inverse")

    def sources(self):
        return ["ops/csrc/tri_inverse.cpp"]

    def schema(self) -> str:
        return "npu_tri_inverse(Tensor cc_in) -> Tensor"

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_tri_inverse_meta(cc_in):
            return cc_in.new_empty(cc_in.shape)


tri_inverse_op_builder = TriInverseOpBuilder()
op_module = tri_inverse_op_builder.load()


@impl(AS_LIBRARY, tri_inverse_op_builder.name, "PrivateUse1")
def _npu_tri_inverse_impl(cc_in):
    return op_module.npu_tri_inverse(cc_in)


def npu_tri_inverse(cc_in):
    return torch.ops.npu_ops_transformer.npu_tri_inverse(cc_in)


torch_npu.npu_tri_inverse = npu_tri_inverse
