#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */

at::Tensor custom_pybind_api(at::Tensor &x, int32_t diagonal = 0)
{
    TORCH_CHECK(x.dim() == 2, "diag_v2 only support 2D tensor");
    at::Tensor result;
    result = torch::empty({std::min(x.size(0), x.size(1))}, x.options());
    EXEC_NPU_CMD(aclnnCustomOp, x, diagonal, result);
    return result;


}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}