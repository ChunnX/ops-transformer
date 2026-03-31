#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <cstdint>
#include <vector>
#include "pytorch_npu_helper.hpp"

at::Tensor custom_pybind_api(
    at::Tensor &grad_output,
    at::Tensor &rois,
    std::vector<int64_t> y_grad_shape_vec,
    int64_t pooled_height,
    int64_t pooled_width,
    double spatial_scale,
    int64_t sampling_ratio,
    bool aligned,
    bool clockwise)
{
    TORCH_CHECK(grad_output.dim() == 4, "grad_output must be 4D (N, pooled_h, pooled_w, C)");
    TORCH_CHECK(rois.dim() == 2 && rois.size(0) == 6, "rois must be shape (6, num_rois)");
    TORCH_CHECK(
        y_grad_shape_vec.size() == 4,
        "y_grad_shape must be length-4 list [B, H, W, C] for forward feature x");
    TORCH_CHECK(
        grad_output.size(0) == rois.size(1),
        "grad_output dim0 (num_rois) must match rois num columns");

    int64_t b = y_grad_shape_vec[0];
    int64_t h = y_grad_shape_vec[1];
    int64_t w = y_grad_shape_vec[2];
    int64_t c = y_grad_shape_vec[3];
    TORCH_CHECK(grad_output.size(3) == c, "grad_output channels must match y_grad_shape[3]");

    at::IntArrayRef y_grad_shape(y_grad_shape_vec);
    at::Tensor y_grad = at::empty({b, h, w, c}, grad_output.options());
    EXEC_NPU_CMD(
        aclnnCustomOp,
        grad_output,
        rois,
        y_grad_shape,
        pooled_height,
        pooled_width,
        spatial_scale,
        sampling_ratio,
        aligned,
        clockwise,
        y_grad);
    return y_grad;
}

PYBIND11_MODULE(kernel_gen_ops, m)
{
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "RoiAlignRotatedGrad / aclnnRoiAlignRotatedGrad");
}
