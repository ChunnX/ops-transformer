#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor custom_pybind_api(
    at::Tensor &permuted_tokens,
    at::Tensor &sorted_indices,
    c10::optional<at::Tensor> routing_map,
    c10::optional<at::Tensor> probs,
    bool drop_and_pad,
    at::IntArrayRef restore_shape,
    at::Tensor &unpermuted_tokens,
    at::Tensor &out_index,
    at::Tensor &permute_token_id,
    at::Tensor &permute_probs)
{
    EXEC_NPU_CMD(aclnnCustomOp, permuted_tokens, sorted_indices, routing_map, probs,
                 drop_and_pad, restore_shape,
                 unpermuted_tokens, out_index, permute_token_id, permute_probs);
    return unpermuted_tokens;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
