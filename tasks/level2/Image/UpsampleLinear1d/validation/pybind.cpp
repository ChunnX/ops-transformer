#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
* 为 NPU 设备注册前向实现
*/

at::Tensor upsample_linear1d(at::Tensor &input, at::IntArrayRef size, bool align_corners, float scale)
{
    // 分配输出内存
    auto output_size = input.sizes().vec();
    output_size[2] = size[0];
    at::Tensor result = torch::empty(output_size, input.options());

    // 创建 size tensor
    at::Tensor size_tensor = torch::tensor({static_cast<int32_t>(size[0])}, torch::kInt32);

    // 调用 aclnn 接口执行计算
    // 参数顺序：输入张量 → 属性参数 → 输出张量（必须在最后）
    EXEC_NPU_CMD(aclnnUpsampleLinear1d, input, size_tensor, align_corners, scale, result);
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("upsample_linear1d", &upsample_linear1d, "");
}
