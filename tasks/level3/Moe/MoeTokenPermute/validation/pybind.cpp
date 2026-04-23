#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

std::vector<at::Tensor> custom_pybind_api(
    const at::Tensor& tokens,
    const at::Tensor& indices,
    int64_t num_out_tokens,
    bool padded_mode) {
    at::TensorOptions options = tokens.options();
    at::Tensor permuted_tokens_out;
    at::Tensor sorted_indices_out;

    if (padded_mode) {
        int64_t n = indices.numel();
        std::vector<int64_t> perm_shape;
        perm_shape.push_back(n);
        for (int64_t i = 1; i < tokens.dim(); ++i) {
            perm_shape.push_back(tokens.size(i));
        }
        permuted_tokens_out = at::empty(perm_shape, options);
        sorted_indices_out = at::empty(indices.sizes(), torch::dtype(torch::kInt32).device(indices.device()));
    } else {
        int64_t num_tokens = tokens.size(0);
        int64_t topk = 1;
        if (indices.dim() == 2) {
            topk = indices.size(1);
        }
        std::vector<int64_t> permuted_tokens_shape(tokens.dim());
        int64_t out_tokens = num_out_tokens;
        int64_t sorted_len = num_tokens * topk;
        if (num_out_tokens <= 0) {
            out_tokens = num_out_tokens + sorted_len;
        }
        out_tokens = std::min(out_tokens, sorted_len);
        out_tokens = std::max(out_tokens, (int64_t)0);
        permuted_tokens_shape[0] = out_tokens;
        for (int64_t i = 1; i < tokens.dim(); i++) {
            permuted_tokens_shape[i] = tokens.size(i);
        }
        permuted_tokens_out = at::empty(permuted_tokens_shape, options);
        sorted_indices_out = at::empty({sorted_len}, torch::dtype(torch::kInt32).device(indices.device()));
    }

    EXEC_NPU_CMD(
        aclnnCustomOp,
        tokens,
        indices,
        num_out_tokens,
        padded_mode,
        permuted_tokens_out,
        sorted_indices_out);

    return {permuted_tokens_out, sorted_indices_out};
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("moe_token_permute", &custom_pybind_api, "");
}
