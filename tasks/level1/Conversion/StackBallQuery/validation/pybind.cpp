#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor custom_pybind_api(
    at::Tensor &xyz,
    at::Tensor &center_xyz,
    at::Tensor &xyz_batch_cnt,
    at::Tensor &center_xyz_batch_cnt,
    double max_radius,
    int64_t sample_num)
{
    int64_t m = center_xyz.size(0);
    at::Tensor idx = at::empty({m * sample_num}, xyz.options().dtype(at::kInt));
    EXEC_NPU_CMD(
        aclnnCustomOp,
        xyz,
        center_xyz,
        xyz_batch_cnt,
        center_xyz_batch_cnt,
        max_radius,
        sample_num,
        idx);
    return idx;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
