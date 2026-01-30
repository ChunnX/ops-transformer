#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */
at::Tensor pows(at::Tensor &x1, at::Tensor &x2)
{
    at::Tensor result = torch::empty_like(x1);

    EXEC_NPU_CMD(aclnnPows, x1, x2, result);
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("pows", &pows, "");
}