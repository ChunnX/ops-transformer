#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <cstdint>
#include <vector>

#include "pytorch_npu_helper.hpp"

/**
 * 须与 msopgen build_out/autogen/aclnn_rotary_position_embedding_grad.h 中
 * aclnnRotaryPositionEmbeddingGradGetWorkspaceSize(dy,cos,sin,xOptional,mode,dxOut,dcosOut,dsinOut,...) 一致。
 */
std::vector<at::Tensor> custom_pybind_api(at::Tensor dy, at::Tensor cos, at::Tensor sin, at::Tensor x, int64_t mode,
                                          at::Tensor dx, at::Tensor dcos, at::Tensor dsin)
{
    EXEC_NPU_CMD(aclnnCustomOp, dy, cos, sin, x, mode, dx, dcos, dsin);
    return {dx, dcos, dsin};
}

PYBIND11_MODULE(kernel_gen_ops, m)
{
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
