#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */

at::Tensor upsample_nearest(at::Tensor &input, at::IntArrayRef output_size,
                             float scales_h, float scales_w, bool exact_mode)
{
    // Permute NCHW -> NHWC (kernel uses NHWC memory layout)
    auto input_nhwc = input.permute({0, 2, 3, 1}).contiguous();

    auto sizes = input.sizes();
    int64_t N = sizes[0];
    int64_t C = sizes[1];
    int64_t out_h = output_size[0];
    int64_t out_w = output_size[1];

    at::Tensor result_nhwc = torch::empty({N, out_h, out_w, C}, input.options());

    EXEC_NPU_CMD(aclnnUpsampleNearest, input_nhwc, output_size, scales_h, scales_w, exact_mode, result_nhwc);

    // Permute NHWC -> NCHW
    return result_nhwc.permute({0, 3, 1, 2}).contiguous();
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("upsample_nearest", &upsample_nearest, "");
}
