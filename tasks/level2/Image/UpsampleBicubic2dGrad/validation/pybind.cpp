#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
* 为 NPU 设备注册前向实现
*/

at::Tensor upsample_bicubic2d_grad(at::Tensor &grad_output, bool align_corners, double scales_h, double scales_w, at::IntArrayRef input_size)
{
    // Allocate output with grad_input spatial dimensions
    auto sizes = grad_output.sizes().vec();
    sizes[2] = input_size[0];
    sizes[3] = input_size[1];
    at::Tensor result = torch::empty(sizes, grad_output.options());

    // EXEC_NPU_CMD: input tensors → attrs → output tensor (last)
    std::printf("DEBUG pybind: align_corners=%d, scales_h=%f, scales_w=%f\n", align_corners, scales_h, scales_w);
    EXEC_NPU_CMD(aclnnUpsampleBicubic2dGrad, grad_output, align_corners, scales_h, scales_w, result);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("upsample_bicubic2d_grad", &upsample_bicubic2d_grad, "");
}
