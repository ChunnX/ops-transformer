#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */

at::Tensor custom_pybind_api(at::Tensor &x, int32_t diagonal = 0)
{
    TORCH_CHECK(x.dim() == 1 || x.dim() == 2, "diag_v2 only support 1D or 2D tensor");
    at::Tensor result;
    if (x.dim() == 2) {
        int64_t M = x.size(0);
        result = torch::empty({M}, x.options());
    }else {
        int64_t N = x.size(0);
        int64_t k_abs = std::abs(diagonal);
        result = torch::empty({N + k_abs, N + k_abs}, x.options());
    }
    EXEC_NPU_CMD(aclnnCustomOp, x, diagonal, result);
    return result;


}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}