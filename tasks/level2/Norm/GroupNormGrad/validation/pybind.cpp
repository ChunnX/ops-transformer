/**
 * @file pybind.cpp
 * aclnnCustomOp 将在 batch_compile 时替换为 aclnnGroupNormGrad
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"

void _group_norm_grad_internal(
    at::Tensor dy,
    at::Tensor mean,
    at::Tensor rstd,
    at::Tensor x,
    at::Tensor gamma,
    int64_t num_groups,
    const char* data_format,
    bool dx_is_require,
    bool dgamma_is_require,
    bool dbeta_is_require,
    at::Tensor dxOut,
    at::Tensor dgammaOut,
    at::Tensor dbetaOut)
{
    EXEC_NPU_CMD(aclnnGroupNormGrad,
                 dy, mean, rstd, x, gamma,
                 num_groups, data_format,
                 dx_is_require, dgamma_is_require, dbeta_is_require,
                 dxOut, dgammaOut, dbetaOut);
}

std::vector<at::Tensor> group_norm_grad(
    at::Tensor dy,
    at::Tensor mean,
    at::Tensor rstd,
    at::Tensor x,
    at::Tensor gamma,
    int64_t num_groups,
    std::string data_format,
    bool dx_is_require,
    bool dgamma_is_require,
    bool dbeta_is_require)
{
    at::Tensor dxOut = dx_is_require ? torch::empty_like(x) : torch::empty_like(x);
    at::Tensor dgammaOut = dgamma_is_require ? torch::empty_like(gamma) : torch::zeros_like(gamma);
    at::Tensor dbetaOut = dbeta_is_require ? torch::empty_like(gamma) : torch::zeros_like(gamma);

    _group_norm_grad_internal(
        dy, mean, rstd, x, gamma,
        num_groups, data_format.c_str(),
        dx_is_require, dgamma_is_require, dbeta_is_require,
        dxOut, dgammaOut, dbetaOut);

    std::vector<at::Tensor> result;
    if (dx_is_require)
        result.push_back(dxOut);
    if (dgamma_is_require)
        result.push_back(dgammaOut);
    if (dbeta_is_require)
        result.push_back(dbetaOut);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("group_norm_grad", &group_norm_grad, "");
}
