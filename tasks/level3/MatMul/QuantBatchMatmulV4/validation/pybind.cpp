#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor custom_pybind_api(
    at::Tensor &x1,
    at::Tensor &x2,
    c10::optional<at::Tensor> bias,
    at::Tensor &x1_scale,
    at::Tensor &x2_scale,
    at::Tensor &y_scale,
    c10::optional<at::Tensor> x1_offset,
    c10::optional<at::Tensor> x2_offset,
    c10::optional<at::Tensor> y_offset,
    c10::optional<at::Tensor> x2_table,
    int64_t dtype,
    int64_t compute_type,
    bool transpose_x1,
    bool transpose_x2,
    int64_t group_size)
{
    auto x1_sizes = x1.sizes();
    auto x2_sizes = x2.sizes();
    int64_t m = transpose_x1 ? x1_sizes[x1_sizes.size() - 1] : x1_sizes[x1_sizes.size() - 2];
    int64_t n = transpose_x2 ? x2_sizes[x2_sizes.size() - 2] : x2_sizes[x2_sizes.size() - 1];

    std::vector<int64_t> out_sizes;
    auto batch_sizes = x1_sizes.vec();
    batch_sizes.pop_back();
    batch_sizes.pop_back();
    out_sizes = batch_sizes;
    out_sizes.push_back(m);
    out_sizes.push_back(n);

    at::Tensor result = torch::empty(out_sizes, x1.options().dtype(torch::kBFloat16));
    EXEC_NPU_CMD(aclnnCustomOp, x1, x2, bias, x1_scale, x2_scale, y_scale, x1_offset, x2_offset, y_offset, x2_table, dtype, compute_type, transpose_x1, transpose_x2, group_size, result);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
