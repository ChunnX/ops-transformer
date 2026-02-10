#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */
std::vector<at::Tensor> custom_pybind_fun(const at::Tensor &xyz1, const at::Tensor &xyz2, const at::Tensor &idx1, const at::Tensor &idx2, const at::Tensor &gradDist1, const at::Tensor &gradDist2)
{
    at::Tensor grad_xyz1 = torch::empty_like(xyz1);
    at::Tensor grad_xyz2 = torch::empty_like(xyz2);

    EXEC_NPU_CMD(aclnnCustomOp, xyz1, xyz2, idx1, idx2, gradDist1, gradDist2, grad_xyz1, grad_xyz2);
    return {grad_xyz1, grad_xyz2};
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_fun, "");
}