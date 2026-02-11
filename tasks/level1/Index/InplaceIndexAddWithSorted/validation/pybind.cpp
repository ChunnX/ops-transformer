#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
* register forward implementation for NPU device
*/

at::Tensor inplace_index_add_with_sorted(
    at::Tensor &var, 
    at::Tensor &value, 
    at::Tensor &sorted_indices, 
    at::Tensor &pos,
    int64_t axis,
    c10::optional<at::Tensor> alpha_tensor)
{
    EXEC_NPU_CMD(aclnnInplaceIndexAddWithSorted, var, value, sorted_indices, pos, alpha_tensor, axis);
    return var;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("inplace_index_add_with_sorted", &inplace_index_add_with_sorted, "InplaceIndexAddWithSorted operator");
}