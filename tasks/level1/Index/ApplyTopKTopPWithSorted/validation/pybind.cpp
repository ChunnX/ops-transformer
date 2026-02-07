#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
* register forward implementation for NPU device
*/

at::Tensor apply_top_k_top_p_with_sorted(
    at::Tensor &sorted_value,
    at::Tensor &sorted_indices,
    c10::optional<at::Tensor> p,
    c10::optional<at::Tensor> k)
{
    // alloc output memory
    at::Tensor result = torch::empty_like(sorted_value);

    // call aclnn interface to perform the computation
    EXEC_NPU_CMD(aclnnApplyTopKTopPWithSorted, sorted_value, sorted_indices, p, k, result);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for ApplyTopKTopPWithSorted operator";
    m.def("apply_top_k_top_p_with_sorted", &apply_top_k_top_p_with_sorted, "ApplyTopKTopPWithSorted operation");
}