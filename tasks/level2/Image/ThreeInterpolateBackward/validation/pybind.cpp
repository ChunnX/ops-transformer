#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * 为 NPU 设备注册前向实现
 */

at::Tensor three_interpolate_backward(at::Tensor &grad_x, at::Tensor &idx, at::Tensor &weight, int64_t m)
{
    auto B = grad_x.size(0);
    auto C1 = grad_x.size(1);
    auto N = grad_x.size(2);
    int64_t C0 = 16;

    // Output: (B, C1, m, 1, C0)
    std::vector<int64_t> output_shape = {B, C1, m, 1, C0};
    at::Tensor result = torch::zeros(output_shape, grad_x.options());

    EXEC_NPU_CMD(aclnnThreeInterpolateBackward, grad_x, idx, weight, m, result);
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("three_interpolate_backward", &three_interpolate_backward, "");
}
