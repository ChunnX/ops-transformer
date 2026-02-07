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
    // alloc output memory (inplace operation, so output is var)
    at::Tensor result = var;
    
    c10::Scalar alpha_scalar;
    if (alpha_tensor.has_value()) {
        alpha_scalar = c10::Scalar(alpha_tensor.value().item<float>());
    } else {
        alpha_scalar = c10::Scalar(1.0f);
    }

    // call aclnn interface to perform the computation
    EXEC_NPU_CMD(aclnnInplaceIndexAddWithSorted, var, value, sorted_indices, pos, axis, alpha_scalar, result);
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("inplace_index_add_with_sorted", &inplace_index_add_with_sorted, "InplaceIndexAddWithSorted operator");
}