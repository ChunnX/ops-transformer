#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor scaled_masked_softmax_v2(
    at::Tensor &x,
    at::Tensor &mask,
    double scale,
    bool fixed_triu_mask)
{
    at::Tensor result = torch::empty_like(x);
    EXEC_NPU_CMD(aclnnScaledMaskedSoftmaxV2, x, mask, scale, fixed_triu_mask, result);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("scaled_masked_softmax_v2", &scaled_masked_softmax_v2, "");
}
