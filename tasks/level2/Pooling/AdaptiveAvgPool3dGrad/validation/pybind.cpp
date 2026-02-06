#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */

at::Tensor adaptive_avg_pool3d_grad(at::Tensor &y_grad, at::Tensor &x)
{
    at::Tensor result = torch::empty_like(x);

    EXEC_NPU_CMD(aclnnAdaptiveAvgPool3dBackward, y_grad, x, result);

    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("adaptive_avg_pool3d_grad", &adaptive_avg_pool3d_grad, "");
}