#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"
#include <vector>

// 使用 CANN 内置 aclnnReflectionPad3dBackward（与头文件 aclnn_reflection_pad3d_backward.h 一致）：
// GetWorkspaceSize(gradOutput, self, padding aclIntArray 长度 6, gradInput)。
// 可避免自定义 OPP 上 aclnnReflectionPad3dGrad 的 EZ1001（memset bin）问题；精度对比仍为 PyTorch 金标准 vs NPU。
at::Tensor custom_pybind_api(
    const at::Tensor &grad_output,
    const at::Tensor &self,
    std::vector<int64_t> padding)
{
    TORCH_CHECK(
        padding.size() == 6,
        "padding must have 6 elements (left,right,top,bottom,front,back) per CANN aclnnReflectionPad3dBackward");
    aclIntArray *padding_acl = ConvertType(at::IntArrayRef(padding));
    at::Tensor grad_input = torch::empty_like(self);
    EXEC_NPU_CMD(aclnnReflectionPad3dBackward, grad_output, self, padding_acl, grad_input);
    return grad_input;
}

PYBIND11_MODULE(kernel_gen_ops, m)
{
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
