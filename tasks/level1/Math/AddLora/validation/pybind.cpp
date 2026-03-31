#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor custom_pybind_api(
    const at::Tensor &y,
    const at::Tensor &x,
    const at::Tensor &weight_b,
    const at::Tensor &indices,
    const at::Tensor &weight_a,
    int64_t layer_idx,
    double scale,
    int64_t y_offset,
    int64_t y_slice_size)
{
    at::Tensor out = y.clone();
    EXEC_NPU_CMD(
        aclnnCustomOp,
        y,
        x,
        weight_b,
        indices,
        weight_a,
        layer_idx,
        scale,
        y_offset,
        y_slice_size,
        out);
    return out;
}

PYBIND11_MODULE(kernel_gen_ops, m)
{
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
