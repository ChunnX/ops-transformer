#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor custom_pybind_api(at::Tensor &x1, at::Tensor &x2, at::Tensor &bias, at::Tensor &offset_w,
                              bool transpose_x1, bool transpose_x2, int64_t offset_x, int64_t opImplMode)
{
    auto x1_sizes = x1.sizes();
    auto x2_sizes = x2.sizes();
    int64_t m = transpose_x1 ? x1_sizes[x1_sizes.size() - 1] : x1_sizes[x1_sizes.size() - 2];
    int64_t n = transpose_x2 ? x2_sizes[x2_sizes.size() - 2] : x2_sizes[x2_sizes.size() - 1];

    auto dtype = x1.scalar_type();
    if (dtype == at::kBFloat16 && x2.scalar_type() == at::kBFloat16) {
        dtype = at::kBFloat16;
    } else if (dtype == at::kFloat && x2.scalar_type() == at::kFloat) {
        dtype = at::kFloat;
    }

    at::Tensor result = torch::empty({m, n}, x1.options().dtype(dtype));
    EXEC_NPU_CMD(aclnnCustomOp, x1, x2, bias, offset_w, transpose_x1, transpose_x2, offset_x, opImplMode, result);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
