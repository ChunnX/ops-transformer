#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */

at::Tensor upsample_nearest2d_grad(at::Tensor &grad_output,
                                    at::IntArrayRef output_size,
                                    at::IntArrayRef input_size,
                                    float scales_h,
                                    float scales_w)
{
    at::Tensor grad_input = torch::empty(input_size, grad_output.options());
    EXEC_NPU_CMD(aclnnCustomOp, grad_output, output_size, input_size, scales_h, scales_w, grad_input);
    return grad_input;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("upsample_nearest2d_grad", &upsample_nearest2d_grad, "");
}
