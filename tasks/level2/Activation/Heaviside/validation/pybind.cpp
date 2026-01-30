#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */
at::Tensor heaviside(at::Tensor &input, at::Tensor &values)
{
    auto broadcastShape = at::infer_size(input.sizes(), values.sizes());
    at::Tensor result = torch::empty(broadcastShape, input.options());

    EXEC_NPU_CMD(aclnnHeaviside, input, values, result);
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("heaviside", &heaviside, "");
}