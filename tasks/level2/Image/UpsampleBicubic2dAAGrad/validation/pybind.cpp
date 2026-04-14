#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * 为 NPU 设备注册前向实现
 */
at::Tensor upsample_bicubic2d_aa_grad(at::Tensor &grad_output, at::IntArrayRef output_size,
    at::IntArrayRef input_size, bool align_corners, float scales_h, float scales_w)
{
    at::Tensor result = torch::empty({input_size[0], input_size[1], input_size[2], input_size[3]},
        grad_output.options());
    EXEC_NPU_CMD(aclnnUpsampleBicubic2dAAGrad, grad_output, output_size, input_size, align_corners, scales_h, scales_w, result);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("upsample_bicubic2d_aa_grad", &upsample_bicubic2d_aa_grad, "");
}
