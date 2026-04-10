#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <cstdint>
#include <vector>

#include "pytorch_npu_helper.hpp"

/**
 * 与 aclnnApplyRotaryPosEmbGetWorkspaceSize 一致：queryRef, keyRef, cos, sin, layout。
 * query/key 为就地改写；须与 build_out/autogen/aclnn_apply_rotary_pos_emb.h 参数顺序、类型一致。
 */
std::vector<at::Tensor> custom_pybind_api(
    at::Tensor query,
    at::Tensor key,
    const at::Tensor& cos,
    const at::Tensor& sin,
    int64_t layout)
{
    EXEC_NPU_CMD(aclnnCustomOp, query, key, cos, sin, layout);
    return {query, key};
}

PYBIND11_MODULE(kernel_gen_ops, m)
{
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
