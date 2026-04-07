#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */
at::Tensor moe_finalize_routing(
    const at::Tensor &expandedX, const at::Tensor &expandedRowIdx, const at::Tensor &x1Optional, 
    const at::Tensor &x2Optional, const at::Tensor &biasOptional, const at::Tensor &scalesOptional, 
    const at::Tensor &expertIdxOptional, int64_t dropPadMode)
{
    auto output_size = x1Optional.sizes();
    at::Tensor out = torch::empty(output_size, x1Optional.options());
    EXEC_NPU_CMD(aclnnMoeFinalizeRouting, expandedX, x1Optional, x2Optional, biasOptional, 
                 scalesOptional, expandedRowIdx, expertIdxOptional, out);

    return out;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("moe_finalize_routing", &moe_finalize_routing, "");
}