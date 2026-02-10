#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */

at::Tensor custom_pybind_api(at::Tensor &x)
{
    // alloc output memory
    auto result_dtype = (x.dtype() == torch::kHalf) ? torch::kHalf ? torch::kFloat;
    at::Tensor result = torch::empty_like(x, x.options().dtype(result_dtype));

    // call aclnn interface to perform the computation
    EXEC_NPU_CMD(aclnnCustomOp, x, result);
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}