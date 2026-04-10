#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"

/**
 * 与 aclnnInterleaveRopeGetWorkspaceSize 一致：x, cos, sin, out（cos 在 sin 之前）。
 * out 须预分配且与 x 同 shape/dtype。
 */
at::Tensor custom_pybind_api(at::Tensor x, at::Tensor cos, at::Tensor sin, at::Tensor out)
{
    EXEC_NPU_CMD(aclnnCustomOp, x, cos, sin, out);
    return out;
}

PYBIND11_MODULE(kernel_gen_ops, m)
{
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
