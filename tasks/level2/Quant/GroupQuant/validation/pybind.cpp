#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <cstdint>
#include <vector>

#include "pytorch_npu_helper.hpp"

/**
 * 与 answer/0/op_host/op_api/aclnn_group_quant.h 一致：
 * x, scale, groupIndex, offsetOptional, dstType(int32), y。
 * autogen 中 dstType 为 int64_t；ge::DT_INT8=2，ge::DT_INT4=29。
 */
std::vector<at::Tensor> custom_pybind_api(
    const at::Tensor &x,
    const at::Tensor &scale,
    const at::Tensor &group_index,
    c10::optional<at::Tensor> offset_optional,
    int64_t dst_type)
{
    at::Tensor y = at::empty(
        x.sizes(),
        at::TensorOptions().dtype(at::kChar).device(x.device()));
    EXEC_NPU_CMD(
        aclnnCustomOp,
        x,
        scale,
        group_index,
        offset_optional,
        dst_type,
        y);
    return {y};
}

PYBIND11_MODULE(kernel_gen_ops, m)
{
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
