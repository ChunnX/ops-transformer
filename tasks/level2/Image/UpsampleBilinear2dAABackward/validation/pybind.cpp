#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * UpsampleBilinear2dAABackward: backward pass of bilinear 2D upsampling with anti-aliasing
 * grad_output: gradient from next layer, shape (N, C, H_out, W_out)
 * output_size: the upsampled output spatial size [H_out, W_out]
 * input_size: the original input spatial size [N, C, H_in, W_in]
 * align_corners: whether to align corners
 * scales_h, scales_w: scale factors (derived from shapes)
 * grad_input: gradient at original input, shape (N, C, H_in, W_in)
 */

at::Tensor upsample_bilinear2d_aa_backward(
    at::Tensor &grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    double scales_h,
    double scales_w)
{
    at::Tensor grad_input = torch::empty({input_size[0], input_size[1], input_size[2], input_size[3]},
        grad_output.options());

    EXEC_NPU_CMD(aclnnUpsampleBilinear2dAABackward, grad_output, output_size, input_size, align_corners, scales_h, scales_w, grad_input);
    return grad_input;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("upsample_bilinear2d_aa_backward", &upsample_bilinear2d_aa_backward, "");
}
