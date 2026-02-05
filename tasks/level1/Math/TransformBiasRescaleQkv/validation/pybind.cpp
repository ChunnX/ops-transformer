#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * transform_bias_rescale_qkv: (qkv, qkv_bias, num_heads) -> (q, k, v)
 * Output shapes: [batch, num_heads, token, dim_per_head] each.
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor> transform_bias_rescale_qkv(
    at::Tensor& qkv, at::Tensor& qkv_bias, int64_t num_heads)
{
    int64_t batch = qkv.size(0);
    int64_t token = qkv.size(1);
    int64_t triple_dim = qkv.size(2);
    int64_t dim = triple_dim / 3;
    int64_t dim_per_head = dim / num_heads;

    auto opts = qkv.options();
    at::Tensor q = torch::empty({batch, num_heads, token, dim_per_head}, opts);
    at::Tensor k = torch::empty({batch, num_heads, token, dim_per_head}, opts);
    at::Tensor v = torch::empty({batch, num_heads, token, dim_per_head}, opts);

    EXEC_NPU_CMD(aclnnTransformBiasRescaleQkv, qkv, qkv_bias, num_heads, q, k, v);
    return std::make_tuple(q, k, v);
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("transform_bias_rescale_qkv", &transform_bias_rescale_qkv, "");
}
