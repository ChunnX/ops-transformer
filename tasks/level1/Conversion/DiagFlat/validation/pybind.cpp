#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"
#include <cstdlib>

at::Tensor custom_pybind_api(at::Tensor &x, int64_t diagonal)
{
    int64_t n = x.numel();
    int64_t k = std::llabs(diagonal);
    int64_t out_sz = n + k;
    at::Tensor result = torch::empty({out_sz, out_sz}, x.options());
    EXEC_NPU_CMD(aclnnCustomOp, x, diagonal, result);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
