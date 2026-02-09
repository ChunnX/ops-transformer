#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */
at::Tensor avg_pool3_d_grad(const at::Tensor &grad_output, const at::Tensor &x, at::IntArrayRef kernel_size,
                            at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
                            int64_t divisor_override)
{
    at::Tensor result = torch::empty_like(x);
    EXEC_NPU_CMD(aclnnAvgPool3dBackward, grad_output, x, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, result);
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("avg_pool3_d_grad", &avg_pool3_d_grad, "");
}