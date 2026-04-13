#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"
#include <vector>

/**
 * register forward implementation for NPU device
 */

std::tuple<at::Tensor, at::Tensor, at::Tensor> moe_token_permute_with_ep(
    at::Tensor &x, 
    at::Tensor &indices, 
    at::Tensor &probs, 
    at::IntArrayRef range_vals, 
    int64_t num_token_out, 
    bool pad_mode)
{
    TORCH_CHECK(x.dim() == 2, "x should be 2D tensor");
    TORCH_CHECK(indices.dim() == 2, "indices should be 2D tensor");
    TORCH_CHECK(probs.dim() == 2, "probs should be 2D tensor");
    TORCH_CHECK(x.size(0) == indices.size(0) && x.size(0) == probs.size(0), "Batch size mismatch");
    TORCH_CHECK(indices.size(1) == probs.size(1), "Number of experts mismatch");
    TORCH_CHECK(range_vals.size() == 2, "range should have exactly 2 elements");

    // Calculate output shapes
    int64_t num_tokens = x.size(0);
    int64_t topk = indices.size(1);
    int64_t hidden_size = x.size(1);
    int64_t range_size = range_vals[1] - range_vals[0];
    
    // Create output tensors
    at::Tensor expanded_x_out = torch::empty({num_token_out, hidden_size}, x.options());
    at::Tensor sorted_indices_out = torch::empty({num_tokens * topk}, x.options().dtype(torch::kInt32));
    at::Tensor expanded_probs_out = torch::empty({num_token_out}, probs.options());

    // Call ACLNN interface using EXEC_NPU_CMD macro
    EXEC_NPU_CMD(aclnnMoeTokenPermuteWithEp, x, indices, probs, range_vals, num_token_out, pad_mode, 
                 expanded_x_out, sorted_indices_out, expanded_probs_out);

    return std::make_tuple(expanded_x_out, sorted_indices_out, expanded_probs_out);
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("moe_token_permute_with_ep", &moe_token_permute_with_ep, "");
}