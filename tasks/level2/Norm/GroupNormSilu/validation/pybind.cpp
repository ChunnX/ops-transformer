/**
 * @file pybind.cpp
 * aclnnCustomOp 将在 batch_compile 时替换为 aclnnGroupNormSilu
 * API: aclnnGroupNormSiluGetWorkspaceSize(x, gammaOptional, betaOptional, numGroups, eps, activateSilu, yOut, meanOut, rstdOut, ...)
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"

std::vector<at::Tensor> group_norm_silu(
    at::Tensor x,
    at::Tensor gamma,
    at::Tensor beta,
    int64_t num_groups,
    double eps,
    bool activate_silu)
{
    TORCH_CHECK(x.defined() && gamma.defined() && beta.defined(),
                "group_norm_silu: x, gamma, beta must be defined");
    TORCH_CHECK(x.is_contiguous(), "group_norm_silu: x must be contiguous");
    x = x.contiguous();
    gamma = gamma.contiguous();
    beta = beta.contiguous();

    at::Tensor yOut = torch::empty_like(x, x.options().memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor meanOut = torch::empty({x.size(0), num_groups}, x.options().memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor rstdOut = torch::empty({x.size(0), num_groups}, x.options().memory_format(c10::MemoryFormat::Contiguous));

    EXEC_NPU_CMD(aclnnGroupNormSilu,
                 x, gamma, beta, num_groups, eps, activate_silu,
                 yOut, meanOut, rstdOut);

    return {yOut, meanOut, rstdOut};
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("group_norm_silu", &group_norm_silu, "");
}
