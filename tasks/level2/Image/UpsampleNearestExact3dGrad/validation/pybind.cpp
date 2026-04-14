#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */

at::Tensor upsample_nearest_exact3d_grad(at::Tensor &grad_output,
                                          at::IntArrayRef input_size,
                                          at::IntArrayRef output_size,
                                          double scales_d,
                                          double scales_h,
                                          double scales_w)
{
    at::Tensor grad_input = torch::empty(input_size, grad_output.options());
    EXEC_NPU_CMD(aclnnCustomOp, grad_output, input_size, output_size, scales_d, scales_h, scales_w, grad_input);
    return grad_input;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("upsample_nearest_exact3d_grad", &upsample_nearest_exact3d_grad, "");
}
