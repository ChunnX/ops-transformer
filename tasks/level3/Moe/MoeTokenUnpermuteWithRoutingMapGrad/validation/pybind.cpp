#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

std::tuple<at::Tensor, at::Tensor> custom_pybind_api(
    at::Tensor &unpermuted_tokens_grad,
    at::Tensor &out_index,
    at::Tensor &permute_token_id,
    c10::optional<at::Tensor> routing_map,
    c10::optional<at::Tensor> permuted_tokens,
    c10::optional<at::Tensor> probs,
    bool drop_and_pad,
    at::IntArrayRef restore_shape,
    at::Tensor &permuted_tokens_grad,
    c10::optional<at::Tensor> probs_grad)
{
    EXEC_NPU_CMD(aclnnCustomOp, unpermuted_tokens_grad, out_index, permute_token_id,
                 routing_map, permuted_tokens, probs,
                 drop_and_pad, restore_shape,
                 permuted_tokens_grad, probs_grad);
    return std::make_tuple(permuted_tokens_grad,
                           probs_grad.has_value() ? probs_grad.value() : at::Tensor());
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
