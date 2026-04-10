#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */
at::Tensor custom_pybind_api(at::Tensor &var, at::Tensor &indices, at::Tensor &updates,
                              int64_t axis, std::string reduction)
{
    at::Tensor result = var.clone();
    char* reduction_ptr = const_cast<char*>(reduction.c_str());
    EXEC_NPU_CMD(aclnnCustomOp, result, indices, updates, axis, reduction_ptr);
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
