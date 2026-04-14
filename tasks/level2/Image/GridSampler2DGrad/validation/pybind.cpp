#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
* 为 NPU 设备注册前向实现
* 注意：AscendC kernel 使用 NHWC 数据布局，需要将 NCHW 输入转换为 NHWC
*/

std::tuple<at::Tensor, at::Tensor> grid_sampler2_d_grad(
    at::Tensor &grad, at::Tensor &x, at::Tensor &grid,
    const std::string &interpolation_mode, const std::string &padding_mode, bool align_corners)
{
    // 将 grad 和 x 从 NCHW 转换为 NHWC
    auto grad_nhwc = grad.permute({0, 2, 3, 1}).contiguous();
    auto x_nhwc = x.permute({0, 2, 3, 1}).contiguous();
    // grid 已经是 [N, H_out, W_out, 2] 格式，无需转换

    // 分配 NHWC 格式的输出内存
    auto dx_nhwc = torch::zeros_like(x_nhwc);
    auto dgrid = torch::zeros_like(grid);

    // 调用 aclnn 接口执行计算
    char *mode_cstr = const_cast<char*>(interpolation_mode.c_str());
    char *padding_mode_cstr = const_cast<char*>(padding_mode.c_str());
    EXEC_NPU_CMD(aclnnGridSampler2DGrad, grad_nhwc, x_nhwc, grid, mode_cstr, padding_mode_cstr, align_corners, dx_nhwc, dgrid);

    // 将 dx 从 NHWC 转换回 NCHW
    auto dx = dx_nhwc.permute({0, 3, 1, 2}).contiguous();
    return std::make_tuple(dx, dgrid);
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("grid_sampler2_d_grad", &grid_sampler2_d_grad, "");
}
