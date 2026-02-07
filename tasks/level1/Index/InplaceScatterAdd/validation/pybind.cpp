#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
* register forward implementation for NPU device
*/

at::Tensor inplace_scatter_add(at::Tensor &var, at::Tensor &indices, at::Tensor &updates)
{
    // alloc output memory (same as var)
    at::Tensor result = var.clone();

    // call aclnn interface to perform the computation
    EXEC_NPU_CMD(aclnnInplaceScatterAdd, result, indices, updates, result);
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("inplace_scatter_add", &inplace_scatter_add, "InplaceScatterAdd operator");
}