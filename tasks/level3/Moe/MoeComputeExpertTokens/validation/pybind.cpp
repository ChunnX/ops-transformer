#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */

at::Tensor moe_compute_expert_tokens(at::Tensor &sorted_experts, int64_t num_experts)
{
    TORCH_CHECK(sorted_experts.dim() == 1, "sorted_experts should be 1D tensor");
    TORCH_CHECK(sorted_experts.scalar_type() == at::kInt, "sorted_experts should be int32 type");
    TORCH_CHECK(num_experts > 0, "num_experts should be greater than 0");

    at::Tensor result = torch::empty({num_experts}, sorted_experts.options().dtype(at::kInt));
    EXEC_NPU_CMD(aclnnMoeComputeExpertTokens, sorted_experts, num_experts, result);
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("moe_compute_expert_tokens", &moe_compute_expert_tokens, "");
}
