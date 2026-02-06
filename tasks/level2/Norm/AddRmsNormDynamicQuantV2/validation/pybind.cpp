/**
 * @file pybind.cpp
 * AddRmsNormDynamicQuantV2: x1, x2, gamma, optional smooth_scale1, optional smooth_scale2, epsilon -> y1,y2,y3,y4,x,scale1,scale2
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"

std::vector<at::Tensor> custom_pybind_api(
    at::Tensor x1, at::Tensor x2, at::Tensor gamma,
    c10::optional<at::Tensor> smooth_scale1, c10::optional<at::Tensor> smooth_scale2,
    float epsilon)
{
    torch::IntArrayRef shapes = x1.sizes();
    std::vector<int64_t> scale_shape(shapes.begin(), shapes.end());
    size_t ndim = gamma.sizes().size();
    for (size_t i = 0; i < ndim; i++) {
        scale_shape[scale_shape.size() - 1 - i] = 1;
    }
    at::Tensor y1 = torch::empty(shapes, torch::TensorOptions().dtype(torch::kInt8).device(x1.device()));
    at::Tensor y2 = torch::empty(shapes, torch::TensorOptions().dtype(torch::kInt8).device(x1.device()));
    at::Tensor y3 = torch::empty(shapes, torch::TensorOptions().dtype(torch::kFloat32).device(x1.device()));
    at::Tensor y4 = torch::empty_like(x1);
    at::Tensor x_out = torch::empty_like(x1);
    at::Tensor scale1 = torch::empty(scale_shape, torch::TensorOptions().dtype(torch::kFloat32).device(x1.device()));
    at::Tensor scale2 = torch::empty(scale_shape, torch::TensorOptions().dtype(torch::kFloat32).device(x1.device()));

    at::Tensor smooth1 = smooth_scale1.has_value() && smooth_scale1.value().defined()
                             ? smooth_scale1.value().contiguous()
                             : at::Tensor();
    at::Tensor smooth2 = smooth_scale2.has_value() && smooth_scale2.value().defined()
                             ? smooth_scale2.value().contiguous()
                             : at::Tensor();

    EXEC_NPU_CMD(aclnnCustomOp, x1, x2, gamma, smooth1, smooth2, epsilon,
                 y1, y2, y3, y4, x_out, scale1, scale2);

    return {y1, y2, y3, y4, x_out, scale1, scale2};
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
