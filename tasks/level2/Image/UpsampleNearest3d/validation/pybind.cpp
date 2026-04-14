#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */

at::Tensor upsample_nearest3d(at::Tensor &x,
                               at::IntArrayRef output_size,
                               float scale_d,
                               float scale_h,
                               float scale_w)
{
    auto x_sizes = x.sizes();
    at::Tensor y = torch::empty({x_sizes[0], x_sizes[1], output_size[0], output_size[1], output_size[2]}, x.options());
    EXEC_NPU_CMD(aclnnCustomOp, x, output_size, scale_d, scale_h, scale_w, y);
    return y;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("upsample_nearest3d", &upsample_nearest3d, "");
}
