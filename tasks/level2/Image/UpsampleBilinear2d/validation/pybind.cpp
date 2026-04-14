#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
* 为 NPU 设备注册前向实现
*/
at::Tensor upsample_bilinear2d(at::Tensor &input, at::Tensor &size, bool align_corners, float scales_h, float scales_w)
{
    auto size_cpu = size.to(torch::kCPU).contiguous();
    auto size_acc = size_cpu.accessor<int32_t, 1>();
    int64_t out_h = size_acc[0];
    int64_t out_w = size_acc[1];

    at::Tensor result = torch::empty({input.size(0), input.size(1), out_h, out_w}, input.options());

    EXEC_NPU_CMD(aclnnUpsampleBilinear2d, input, size, align_corners, scales_h, scales_w, result);
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("upsample_bilinear2d", &upsample_bilinear2d, "");
}
