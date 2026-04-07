#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor moe_finalize_routing_v2(at::Tensor expandedX, at::Tensor expandedRowIdx, 
                                   at::Tensor x1Optional, at::Tensor x2Optional, 
                                   at::Tensor biasOptional, at::Tensor scalesOptional, 
                                   at::Tensor expertIdxOptional, int64_t dropPadMode) {
    auto output_size = x1Optional.sizes();
    at::Tensor out = torch::empty(output_size, x1Optional.options());
    EXEC_NPU_CMD(aclnnMoeFinalizeRoutingV2, expandedX, expandedRowIdx, x1Optional, x2Optional, 
                 biasOptional, scalesOptional, expertIdxOptional, dropPadMode, out);

    return out;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("moe_finalize_routing_v2", &moe_finalize_routing_v2, "");
}