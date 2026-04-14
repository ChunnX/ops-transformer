#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * NPU forward implementation for UpsampleBicubic2d
 */
at::Tensor upsample_bicubic2d(at::Tensor &input, at::IntArrayRef output_size,
                             bool align_corners, float scales_h, float scales_w)
{
    at::Tensor result = torch::empty({input.size(0), input.size(1), output_size[0], output_size[1]},
                                      input.options());

    EXEC_NPU_CMD(aclnnUpsampleBicubic2d, input, output_size, align_corners, scales_h, scales_w, result);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("upsample_bicubic2d", &upsample_bicubic2d, "");
}
