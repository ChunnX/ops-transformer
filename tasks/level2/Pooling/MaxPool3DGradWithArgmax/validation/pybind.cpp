#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */

at::Tensor max_pool3d_grad_with_argmax(
    const at::Tensor &x, const at::Tensor &grad, const at::Tensor &indices, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode)
{
    auto input_size = x.sizes();
    at::Tensor grad_input = torch::empty(input_size, x.options());
    EXEC_NPU_CMD(aclnnMaxPool3dWithArgmaxBackward, grad, x, indices, kernel_size, stride, padding,
                 dilation, ceil_mode, grad_input);

    return grad_input;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("max_pool3d_grad_with_argmax", &max_pool3d_grad_with_argmax, "");
}