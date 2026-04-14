#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * 为 NPU 设备注册前向实现
 */

at::Tensor resize_upsample_trilinear(
    at::Tensor &input,
    at::IntArrayRef output_size,
    bool align_corners,
    double scales_d,
    double scales_h,
    double scales_w)
{
    // Compute output shape: [N, C, D_out, H_out, W_out]
    auto input_sizes = input.sizes();
    std::vector<int64_t> out_shape = {
        input_sizes[0], input_sizes[1],
        output_size[0], output_size[1], output_size[2]
    };
    at::Tensor result = torch::empty(out_shape, input.options());

    EXEC_NPU_CMD(aclnnResizeUpsampleTrilinear, input, output_size, align_corners,
                 scales_d, scales_h, scales_w, result);
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("resize_upsample_trilinear", &resize_upsample_trilinear, "");
}
