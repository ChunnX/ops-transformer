#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */
std::vector<at::Tensor> custom_pybind_fun(at::Tensor x1, at::Tensor x2, int64_t dim) {
    at::Tensor y = at::empty_like(x1);

    // 调用 NPU 算子
    EXEC_NPU_CMD(aclnnCrossV2, x1, x2, dim, y);

    return {y};
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_fun, "");
}