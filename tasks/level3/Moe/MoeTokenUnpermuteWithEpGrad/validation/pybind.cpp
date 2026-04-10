#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

std::tuple<at::Tensor, at::Tensor> moe_token_unpermute_with_ep_grad(
    at::Tensor &unpermuted_tokens_grad,
    at::Tensor &sorted_indices,
    c10::optional<at::Tensor> permuted_tokens,
    c10::optional<at::Tensor> probs,
    bool padded_mode,
    at::IntArrayRef restore_shape,
    at::IntArrayRef range,
    int64_t topk_num)
{
    int64_t tokens_num = unpermuted_tokens_grad.size(0);
    int64_t hidden_size = unpermuted_tokens_grad.size(1);
    int64_t total_indices = sorted_indices.size(0);
    at::Tensor permuted_tokens_grad = torch::zeros({total_indices, hidden_size}, unpermuted_tokens_grad.options());

    at::Tensor probs_grad;
    if (probs.has_value() && probs->defined() && probs->numel() > 0) {
        probs_grad = torch::zeros_like(*probs);
    } else {
        probs_grad = torch::zeros({tokens_num, topk_num}, unpermuted_tokens_grad.options());
    }

    EXEC_NPU_CMD(aclnnMoeTokenUnpermuteWithEpGrad, unpermuted_tokens_grad, sorted_indices, permuted_tokens, probs,
                 padded_mode, restore_shape, range, topk_num, permuted_tokens_grad, probs_grad);
    return std::make_tuple(permuted_tokens_grad, probs_grad);
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("moe_token_unpermute_with_ep_grad", &moe_token_unpermute_with_ep_grad, "");
}
