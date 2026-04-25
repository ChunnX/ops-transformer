#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

std::tuple<at::Tensor, at::Tensor, at::Tensor> custom_pybind_api(
    const at::Tensor& tokens,
    const at::Tensor& routing_map,
    const c10::optional<at::Tensor>& probs_optional,
    int64_t num_out_tokens,
    bool drop_and_pad) {
    TORCH_CHECK(tokens.dim() == 2, "tokens should be 2D tensor");
    TORCH_CHECK(routing_map.dim() == 2, "routing_map should be 2D tensor");
    TORCH_CHECK(tokens.size(0) == routing_map.size(0), "tokens dim0 must match routing_map dim0");
    if (probs_optional.has_value() && probs_optional->defined() && probs_optional->numel() > 0) {
        TORCH_CHECK(probs_optional->dim() == 2, "probs_optional should be 2D tensor");
        TORCH_CHECK(probs_optional->size(0) == routing_map.size(0), "probs dim0 must match routing_map");
        TORCH_CHECK(probs_optional->size(1) == routing_map.size(1), "probs dim1 must match routing_map");
    }

    int64_t hidden_size = tokens.size(1);
    at::Tensor permute_tokens_out = at::empty({num_out_tokens, hidden_size}, tokens.options());
    at::Tensor sorted_indices_out = at::empty({num_out_tokens}, tokens.options().dtype(torch::kInt32));

    const bool has_probs =
        probs_optional.has_value() && probs_optional->defined() && probs_optional->numel() > 0;
    at::Tensor permute_probs_out =
        has_probs ? at::empty({num_out_tokens}, probs_optional->options()) : at::empty({0}, tokens.options());

    EXEC_NPU_CMD(
        aclnnCustomOp,
        tokens,
        routing_map,
        probs_optional,
        num_out_tokens,
        drop_and_pad,
        permute_tokens_out,
        permute_probs_out,
        sorted_indices_out);

    return std::make_tuple(permute_tokens_out, permute_probs_out, sorted_indices_out);
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("moe_token_permute_with_routing_map", &custom_pybind_api, "");
}
