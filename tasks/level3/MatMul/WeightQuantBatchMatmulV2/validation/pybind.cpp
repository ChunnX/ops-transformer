#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor custom_pybind_api(
    at::Tensor &x,
    at::Tensor &weight,
    at::Tensor &antiquant_scale,
    c10::optional<at::Tensor> antiquant_offset,
    c10::optional<at::Tensor> quant_scale,
    c10::optional<at::Tensor> quant_offset,
    c10::optional<at::Tensor> bias,
    bool transpose_x,
    bool transpose_weight,
    int64_t antiquant_group_size,
    int64_t dtype,
    int64_t inner_precise)
{
    auto x_sizes = x.sizes();
    int64_t m = transpose_x ? x_sizes[x_sizes.size() - 1] : x_sizes[x_sizes.size() - 2];
    int64_t k = transpose_x ? x_sizes[x_sizes.size() - 2] : x_sizes[x_sizes.size() - 1];

    auto weight_sizes = weight.sizes();
    int64_t n = transpose_weight ? weight_sizes[weight_sizes.size() - 2] : weight_sizes[weight_sizes.size() - 1];

    std::vector<int64_t> out_sizes = {m, n};

    at::Tensor result = torch::empty(out_sizes, x.options().dtype(torch::kFloat16));
    EXEC_NPU_CMD(aclnnCustomOp, x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias, transpose_x, transpose_weight, antiquant_group_size, dtype, inner_precise, result);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
