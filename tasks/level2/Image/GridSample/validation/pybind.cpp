#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
* register forward implementation for NPU device
*/

at::Tensor grid_sample_npu(at::Tensor &x, at::Tensor &grid,
                           const std::string &mode, const std::string &padding_mode,
                           bool align_corners)
{
    // Output shape: (N, C, outH, outW) for 2D or (N, C, outD, outH, outW) for 3D
    auto x_sizes = x.sizes().vec();
    auto grid_sizes = grid.sizes().vec();
    std::vector<int64_t> out_sizes;
    out_sizes.push_back(x_sizes[0]); // N
    out_sizes.push_back(x_sizes[1]); // C
    for (size_t i = 1; i < grid_sizes.size() - 1; i++) {
        out_sizes.push_back(grid_sizes[i]); // H, W or D, H, W
    }

    at::Tensor result = torch::empty(out_sizes, x.options());

    char *mode_cstr = const_cast<char*>(mode.c_str());
    char *padding_mode_cstr = const_cast<char*>(padding_mode.c_str());
    bool channel_last = false;
    int64_t scheduler_mode = 0;
    EXEC_NPU_CMD(aclnnGridSample, x, grid, mode_cstr, padding_mode_cstr, align_corners, channel_last, scheduler_mode, result);
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("grid_sample", &grid_sample_npu, "");
}
