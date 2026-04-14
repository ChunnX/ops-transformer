#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
* 为 NPU 设备注册前向实现
* 注意：AscendC kernel 使用 NDHWC 数据布局，需要将 NCDHW 输入转换为 NDHWC
*/

std::tuple<at::Tensor, at::Tensor> grid_sampler3_d_grad(
    at::Tensor &grad, at::Tensor &x, at::Tensor &grid,
    const std::string &interpolation_mode, const std::string &padding_mode, bool align_corners)
{
    // 将 grad 和 x 从 NCDHW 转换为 NDHWC
    auto grad_ndhwc = grad.permute({0, 2, 3, 4, 1}).contiguous();
    auto x_ndhwc = x.permute({0, 2, 3, 4, 1}).contiguous();
    // grid 已经是 [N, D_out, H_out, W_out, 3] 格式，无需转换

    // 分配 NDHWC 格式的输出内存
    auto dx_ndhwc = torch::zeros_like(x_ndhwc);
    auto dgrid = torch::zeros_like(grid);

    // 调用 aclnn 接口执行计算
    char *mode_cstr = const_cast<char*>(interpolation_mode.c_str());
    char *padding_mode_cstr = const_cast<char*>(padding_mode.c_str());
    EXEC_NPU_CMD(aclnnGridSampler3DGrad, grad_ndhwc, x_ndhwc, grid, mode_cstr, padding_mode_cstr, align_corners, dx_ndhwc, dgrid);

    // 将 dx 从 NDHWC 转换回 NCDHW
    auto dx = dx_ndhwc.permute({0, 4, 1, 2, 3}).contiguous();
    return std::make_tuple(dx, dgrid);
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("grid_sampler3_d_grad", &grid_sampler3_d_grad, "");
}
