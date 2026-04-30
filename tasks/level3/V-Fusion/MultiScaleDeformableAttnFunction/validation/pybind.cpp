#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor custom_pybind_api(at::Tensor &value, at::Tensor &value_spatial_shapes,
    at::Tensor &value_level_start_index, at::Tensor &sampling_locations, at::Tensor &attention_weights)
{
    int64_t batch = value.size(0);
    int64_t embed_dims = value.size(3);
    bool is_transpose = sampling_locations.size(1) < 32;
    int64_t num_queries, num_heads;
    if (is_transpose) {
        num_heads = sampling_locations.size(1);
        num_queries = sampling_locations.size(4);
    } else {
        num_queries = sampling_locations.size(1);
        num_heads = sampling_locations.size(2);
    }
    auto output_sizes = std::vector<int64_t>{batch, num_queries, num_heads * embed_dims};
    at::Tensor result = torch::zeros(output_sizes, value.options());
    EXEC_NPU_CMD(aclnnCustomOp, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, result);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "kernel_gen_ops custom operator");
}