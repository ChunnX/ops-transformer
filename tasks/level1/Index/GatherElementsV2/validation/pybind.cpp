#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor custom_pybind_api(at::Tensor &x, at::Tensor &index, int64_t dim)
{
    at::Tensor result = torch::empty_like(index, x.options());
    EXEC_NPU_CMD(aclnnCustomOp, x, index, dim, result);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
