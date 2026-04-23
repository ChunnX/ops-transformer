#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

std::tuple<at::Tensor, at::Tensor> custom_pybind_api(
    const at::Tensor& permuted_tokens_output_grad,
    const at::Tensor& permuted_probs_output_grad_optional,
    const at::Tensor& sorted_indices,
    const at::Tensor& routing_map_optional,
    int64_t experts_num,
    int64_t tokens_num,
    bool padded_mode)
{
    TORCH_CHECK(permuted_tokens_output_grad.dim() == 2, "permuted_tokens_output_grad should be 2D tensor");
    TORCH_CHECK(sorted_indices.dim() == 1, "sorted_indices should be 1D tensor");
    TORCH_CHECK(sorted_indices.scalar_type() == at::kInt, "sorted_indices should be int32 type");
    TORCH_CHECK(routing_map_optional.dim() == 2, "routing_map_optional should be 2D tensor");
    TORCH_CHECK(experts_num > 0, "experts_num should be greater than 0");
    TORCH_CHECK(tokens_num > 0, "tokens_num should be greater than 0");

    const int64_t hidden_size = permuted_tokens_output_grad.size(1);
    const int64_t num_out = permuted_tokens_output_grad.size(0);
    TORCH_CHECK(
        num_out % tokens_num == 0,
        "permuted_tokens_output_grad.size(0) must be divisible by tokens_num (infershape top_k).");
    const int64_t top_k = num_out / tokens_num;

    at::Tensor tokens_grad_out = torch::empty({tokens_num, hidden_size}, permuted_tokens_output_grad.options());

    TORCH_CHECK(
        permuted_probs_output_grad_optional.defined() && permuted_probs_output_grad_optional.numel() > 0,
        "permuted_probs_output_grad_optional must be non-empty 1D (length num_out).");
    TORCH_CHECK(permuted_probs_output_grad_optional.dim() == 1, "probs grad must be 1D.");
    TORCH_CHECK(permuted_probs_output_grad_optional.size(0) == num_out, "probs grad length must match num_out.");

    at::Tensor probs_grad_out_optional =
        torch::empty({tokens_num, top_k}, permuted_probs_output_grad_optional.options());

    EXEC_NPU_CMD(aclnnCustomOp,
        permuted_tokens_output_grad,
        permuted_probs_output_grad_optional,
        sorted_indices,
        routing_map_optional,
        experts_num,
        tokens_num,
        padded_mode,
        tokens_grad_out,
        probs_grad_out_optional);

    return std::make_tuple(tokens_grad_out, probs_grad_out_optional);
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("moe_token_permute_with_routing_map_grad", &custom_pybind_api, "");
}
