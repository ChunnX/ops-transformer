#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */
at::Tensor squared_relu(at::Tensor &input)
{
    at::Tensor result = torch::empty_like(input);

    EXEC_NPU_CMD(aclnnSquaredRelu, input, result);
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("squared_relu", &squared_relu, "");
}