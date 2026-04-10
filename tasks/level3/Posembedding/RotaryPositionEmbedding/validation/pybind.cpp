#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <cstdint>

#include "pytorch_npu_helper.hpp"

/**
 * 与 aclnnRotaryPositionEmbeddingGetWorkspaceSize 一致：x, cos, sin, mode, out（cos 在 sin 前）。
 */
at::Tensor custom_pybind_api(at::Tensor x, at::Tensor cos, at::Tensor sin, int64_t mode, at::Tensor out)
{
    EXEC_NPU_CMD(aclnnCustomOp, x, cos, sin, mode, out);
    return out;
}

PYBIND11_MODULE(kernel_gen_ops, m)
{
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
