#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor scaled_masked_softmax_grad_v2(
    at::Tensor &yGrad,
    at::Tensor &y,
    at::Tensor &mask,
    double scale,
    bool fixed_triu_mask)
{
    at::Tensor result = torch::empty_like(yGrad);
    EXEC_NPU_CMD(aclnnScaledMaskedSoftmaxGradV2, yGrad, y, mask, scale, fixed_triu_mask, result);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("scaled_masked_softmax_grad_v2", &scaled_masked_softmax_grad_v2, "");
}
