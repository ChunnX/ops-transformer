#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * 为 NPU 设备注册前向实现
 */
at::Tensor upsample_bicubic2d_aa(at::Tensor &x, at::IntArrayRef output_size, bool align_corners, float scales_h, float scales_w)
{
    at::Tensor result = torch::empty({x.size(0), x.size(1), output_size[0], output_size[1]}, x.options());
    EXEC_NPU_CMD(aclnnUpsampleBicubic2dAA, x, output_size, align_corners, scales_h, scales_w, result);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("upsample_bicubic2d_aa", &upsample_bicubic2d_aa, "");
}
