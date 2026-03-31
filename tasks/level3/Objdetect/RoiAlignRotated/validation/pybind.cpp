#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor custom_pybind_api(
    at::Tensor &x,
    at::Tensor &rois,
    int64_t pooled_height,
    int64_t pooled_width,
    double spatial_scale,
    int64_t sampling_ratio,
    bool aligned,
    bool clockwise)
{
    TORCH_CHECK(x.dim() == 4, "x must be 4D (B, H, W, C)");
    TORCH_CHECK(rois.dim() == 2 && rois.size(0) == 6, "rois must be shape (6, num_rois)");
    int64_t num_rois = rois.size(1);
    at::Tensor y = at::empty({num_rois, pooled_height, pooled_width, x.size(3)}, x.options());
    EXEC_NPU_CMD(
        aclnnCustomOp,
        x,
        rois,
        pooled_height,
        pooled_width,
        spatial_scale,
        sampling_ratio,
        aligned,
        clockwise,
        y);
    return y;
}

PYBIND11_MODULE(kernel_gen_ops, m)
{
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "RoiAlignRotated / aclnnRoiAlignRotated");
}
