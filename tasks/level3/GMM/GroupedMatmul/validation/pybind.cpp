#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor custom_pybind_api(
    at::Tensor &x,
    at::Tensor &weight,
    at::Tensor &bias,
    at::Tensor &scale,
    at::Tensor &offset,
    at::Tensor &antiquant_scale,
    at::Tensor &antiquant_offset,
    at::Tensor &group_list,
    at::Tensor &per_token_scale,
    int64_t split_item,
    int64_t group_type,
    int64_t group_list_type,
    int64_t act_type,
    at::IntArrayRef tuning_config)
{
    auto x_shape = x.sizes();
    auto weight_shape = weight.sizes();
    int64_t M = x_shape[0];
    int64_t K = x_shape[1];
    int64_t N = weight_shape[0];
    at::Tensor y = torch::empty({M, N}, x.options());
    EXEC_NPU_CMD(aclnnCustomOp, x, weight, bias, scale, offset, antiquant_scale,
                 antiquant_offset, group_list, per_token_scale,
                 split_item, group_type, group_list_type, act_type, tuning_config, y);
    return y;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "kernel_gen_ops custom operator");
}
