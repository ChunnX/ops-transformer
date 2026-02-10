/**
 * @file extension_rms_norm_quant.cpp
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"

std::vector<at::Tensor> rms_norm_quant(
    at::Tensor x,
    at::Tensor gamma,
    at::Tensor beta,
    at::Tensor scale,
    at::Tensor offset,
    double epsilon)
{
    x = x.contiguous();
    gamma = gamma.contiguous();
    beta = beta.contiguous();
    scale = scale.contiguous();
    offset = offset.contiguous();

    auto options = torch::TensorOptions().dtype(torch::kInt8).device(x.device());
    at::Tensor yOut = torch::empty(x.sizes(), options);

    // 参数顺序与 aclnnRmsNormQuantGetWorkspaceSize 一致；epsilon 以 double 按值传入 executor
    EXEC_NPU_CMD(aclnnRmsNormQuant, x, gamma, beta, scale, offset, epsilon, yOut);

    std::vector<at::Tensor> result = {yOut};
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("rms_norm_quant", &rms_norm_quant, "");
}
