#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */
std::vector<at::Tensor> custom_pybind_fun(at::Tensor x, at::Tensor dy, c10::optional<double> eps) {
    at::Tensor dx = at::empty_like(x);

    // 转换 eps 为 double
    double eps_val = -1.0;
    if (eps != c10::nullopt) {
        eps_val = eps.value();
    }

    // 调用 NPU 算子
    EXEC_NPU_CMD(aclnnLogitGrad, x, dy, eps_val, dx);
    return {dx};
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_fun, "");
}
  