#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

std::vector<at::Tensor> custom_pybind_api(
    at::Tensor &value,
    at::Tensor &value_spatial_shapes,
    at::Tensor &value_level_start_index,
    at::Tensor &sampling_locations,
    at::Tensor &attention_weights,
    at::Tensor &grad_output)
{
    at::Tensor grad_value = torch::zeros_like(value);
    at::Tensor grad_sampling_locations = torch::zeros_like(sampling_locations);
    at::Tensor grad_attention_weights = torch::zeros_like(attention_weights);
    EXEC_NPU_CMD(aclnnCustomOp, value, value_spatial_shapes, value_level_start_index,
                 sampling_locations, attention_weights, grad_output,
                 grad_value, grad_sampling_locations, grad_attention_weights);
    return {grad_value, grad_sampling_locations, grad_attention_weights};
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
