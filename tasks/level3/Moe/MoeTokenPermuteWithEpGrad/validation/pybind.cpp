#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"
#include <vector>

/**
 * register forward implementation for NPU device
 */

std::tuple<at::Tensor, at::Tensor> moe_token_permute_with_ep_grad(
    at::Tensor &unpermuted_tokens_grad, 
    at::Tensor &sorted_indices, 
    c10::optional<at::Tensor> permuted_probs_output_grad_opt, 
    int64_t topk_num, 
    at::IntArrayRef range_vals, 
    bool padded_mode)
{
    TORCH_CHECK(unpermuted_tokens_grad.dim() == 2, "unpermuted_tokens_grad should be 2D tensor");
    TORCH_CHECK(sorted_indices.dim() == 1, "sorted_indices should be 1D tensor");
    TORCH_CHECK(topk_num > 0 && topk_num <= 512, "topk_num should be in (0, 512]");
    TORCH_CHECK(padded_mode == false, "padded_mode is not supported yet");
    
    if (permuted_probs_output_grad_opt.has_value()) {
        TORCH_CHECK(permuted_probs_output_grad_opt.value().dim() == 1, "permuted_probs_output_grad should be 1D tensor if provided");
    }
    
    // Calculate output shapes
    int64_t num_tokens = sorted_indices.size(0);
    int64_t hidden_size = unpermuted_tokens_grad.size(1);
    
    // Create output tensors
    at::Tensor token_grad_out = torch::empty({num_tokens / topk_num, hidden_size}, unpermuted_tokens_grad.options());
    at::Tensor probs_grad_out;
    
    if (permuted_probs_output_grad_opt.has_value()) {
        probs_grad_out = torch::empty({num_tokens / topk_num, topk_num}, permuted_probs_output_grad_opt.value().options());
    } else {
        probs_grad_out = at::Tensor();
    }
    
    // Call ACLNN interface using EXEC_NPU_CMD macro
    if (permuted_probs_output_grad_opt.has_value()) {
        EXEC_NPU_CMD(aclnnCustomOp, 
                     unpermuted_tokens_grad, 
                     sorted_indices, 
                     permuted_probs_output_grad_opt.value(), 
                     topk_num, 
                     range_vals, 
                     padded_mode, 
                     token_grad_out, 
                     probs_grad_out);
    } else {
        // 创建空的Tensor作为可选参数
        at::Tensor empty_tensor;
        at::Tensor empty_probs_grad;
        EXEC_NPU_CMD(aclnnCustomOp, 
                     unpermuted_tokens_grad, 
                     sorted_indices, 
                     empty_tensor, 
                     topk_num, 
                     range_vals, 
                     padded_mode, 
                     token_grad_out, 
                     empty_probs_grad);
    }
    
    return std::make_tuple(token_grad_out, probs_grad_out);
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("moe_token_permute_with_ep_grad", &moe_token_permute_with_ep_grad, "");
}