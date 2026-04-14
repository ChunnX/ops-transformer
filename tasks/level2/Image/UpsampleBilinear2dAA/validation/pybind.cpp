#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * 为 NPU 设备注册前向实现
 */
at::Tensor upsample_bilinear2d_aa(at::Tensor &input, at::IntArrayRef output_size, bool align_corners, float scales_h, float scales_w)
{
    at::Tensor result = torch::empty({input.size(0), input.size(1), output_size[0], output_size[1]}, input.options());
    EXEC_NPU_CMD(aclnnUpsampleBilinear2dAA, input, output_size, align_corners, scales_h, scales_w, result);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("upsample_bilinear2d_aa", &upsample_bilinear2d_aa, "");
}
