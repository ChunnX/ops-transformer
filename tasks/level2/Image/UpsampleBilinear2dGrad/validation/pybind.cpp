#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * UpsampleBilinear2dGrad: compute gradient of bilinear 2d upsampling.
 * params order: input tensor -> attrs -> output tensor (last)
 */
at::Tensor upsample_bilinear2d_grad(at::Tensor &grad_output,
                             at::IntArrayRef output_size,
                             at::IntArrayRef input_size,
                             bool align_corners,
                             double scales_h,
                             double scales_w)
{
    at::Tensor grad_input = torch::empty({input_size[0], input_size[1], input_size[2], input_size[3]},
                                         grad_output.options());

    EXEC_NPU_CMD(aclnnUpsampleBilinear2dGrad, grad_output, output_size, input_size, align_corners, scales_h, scales_w, grad_input);
    return grad_input;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("upsample_bilinear2d_grad", &upsample_bilinear2d_grad, "");
}
