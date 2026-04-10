#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor custom_pybind_api(
    at::Tensor &permuted_tokens,
    at::Tensor &sorted_indices,
    c10::optional<at::Tensor> probs,
    int64_t num_topk,
    at::IntArrayRef range,
    at::Tensor &out)
{
    bool padded_mode = false;
    c10::optional<at::IntArrayRef> restore_shape = c10::nullopt;
    EXEC_NPU_CMD(aclnnCustomOp, permuted_tokens, sorted_indices, probs, num_topk, range, padded_mode, restore_shape, out);
    return out;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
