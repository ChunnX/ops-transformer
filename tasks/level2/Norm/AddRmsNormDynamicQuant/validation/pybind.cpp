/**
 * @file pybind.cpp
 * AddRmsNormDynamicQuant: x1, x2, gamma, optional smooth_scale1/2, epsilon -> y1, y2, x, scale1, scale2
 * Aligned with CANN aclnn_add_rms_norm_dynamic_quant.h: y1/y2 shape=x1, scale1/scale2 shape=x1除最后一维, epsilon=double.
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"

std::vector<at::Tensor> add_rms_norm_dynamic_quant(
    at::Tensor x1, at::Tensor x2, at::Tensor gamma,
    c10::optional<at::Tensor> smooth_scale1, c10::optional<at::Tensor> smooth_scale2,
    double epsilon)
{
    x1 = x1.contiguous();
    x2 = x2.contiguous();
    gamma = gamma.contiguous();
    torch::IntArrayRef shapes = x1.sizes();
    std::vector<int64_t> scale_shape(shapes.begin(), shapes.end());
    size_t ndim = gamma.sizes().size();
    for (size_t i = 0; i < ndim; i++) {
        scale_shape[scale_shape.size() - 1 - i] = 1;
    }
    // CANN: when no smooth, internal y2Out/scale2Out use shape [1]; ViewCopy requires same shape.
    bool has_smooth = (smooth_scale1.has_value() && smooth_scale1->defined())
        && (smooth_scale2.has_value() && smooth_scale2->defined());
    at::Tensor y1 = torch::empty(shapes, torch::TensorOptions().dtype(torch::kInt8).device(x1.device()));
    at::Tensor y2 = has_smooth
        ? torch::empty(shapes, torch::TensorOptions().dtype(torch::kInt8).device(x1.device()))
        : torch::empty({1}, torch::TensorOptions().dtype(torch::kInt8).device(x1.device()));
    at::Tensor x_out = torch::empty_like(x1);
    at::Tensor scale1 = torch::empty(scale_shape, torch::TensorOptions().dtype(torch::kFloat32).device(x1.device()));
    at::Tensor scale2 = has_smooth
        ? torch::empty(scale_shape, torch::TensorOptions().dtype(torch::kFloat32).device(x1.device()))
        : torch::empty({1}, torch::TensorOptions().dtype(torch::kFloat32).device(x1.device()));

    c10::optional<at::Tensor> smooth1_opt = smooth_scale1.has_value() && smooth_scale1->defined()
        ? c10::optional<at::Tensor>(smooth_scale1->contiguous()) : c10::nullopt;
    c10::optional<at::Tensor> smooth2_opt = smooth_scale2.has_value() && smooth_scale2->defined()
        ? c10::optional<at::Tensor>(smooth_scale2->contiguous()) : c10::nullopt;

    EXEC_NPU_CMD(aclnnAddRmsNormDynamicQuant, x1, x2, gamma, smooth1_opt, smooth2_opt, epsilon,
                 y1, y2, x_out, scale1, scale2);

    return {y1, y2, x_out, scale1, scale2};
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("add_rms_norm_dynamic_quant", &add_rms_norm_dynamic_quant, "");
}
