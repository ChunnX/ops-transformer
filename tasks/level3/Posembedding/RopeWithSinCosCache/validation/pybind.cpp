#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * rope_with_sin_cos_cache: Apply Rotary Position Embedding with sin/cos cache.
 * Inputs: positions, queryIn, keyIn, cosSinCache
 * Outputs: queryOut, keyOut
 */
std::tuple<at::Tensor, at::Tensor> rope_with_sin_cos_cache(
    at::Tensor &positions, at::Tensor &query_in, at::Tensor &key_in, at::Tensor &cos_sin_cache,
    int64_t num_q_heads, int64_t num_k_heads, int64_t head_size,
    at::IntArrayRef mrope_section, int64_t q_stride, int64_t k_stride,
    bool is_neox_style, int64_t cache_mode)
{
    at::Tensor query_out = torch::empty_like(query_in);
    at::Tensor key_out = torch::empty_like(key_in);
    EXEC_NPU_CMD(aclnnCustomOp, positions, query_in, key_in, cos_sin_cache,
                 num_q_heads, num_k_heads, head_size,
                 mrope_section, q_stride, k_stride, is_neox_style, cache_mode,
                 query_out, key_out);
    return std::make_tuple(query_out, key_out);
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("rope_with_sin_cos_cache", &rope_with_sin_cos_cache, "");
}
