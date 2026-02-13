#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */
std::vector<at::Tensor> custom_pybind_api(at::Tensor input, c10::optional<double> eps) {
    at::Tensor output = at::empty_like(input);
    
    // 转换 eps 为 double
    double eps_val = -1;
    if (eps != c10::nullopt) {
        eps_val = eps.value();
    }

    // 调用 NPU 算子
    EXEC_NPU_CMD(aclnnLogit, input, eps_val, output);

    return {output};
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}