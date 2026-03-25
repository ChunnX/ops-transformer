#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * autogen aclnn_fill_diagonal_v2：fillValue 为 aclTensor*。
 * 须与 x **同 dtype**（EZ1001）。AICore 从 GM 读 fill，故 fill 须在与 x **相同 device**（一般为 NPU）上；
 * 若将 fill 留在 CPU，易出现 vector core / UB D-cache 类异常（507035）。
 */
at::Tensor custom_pybind_api(at::Tensor &x, at::Tensor &fill_value, bool wrap)
{
    TORCH_CHECK(fill_value.numel() == 1, "fillValue must contain exactly one element");
    at::Tensor fill = fill_value.detach().contiguous().reshape({1});
    fill = fill.to(at::TensorOptions().dtype(x.scalar_type()).device(x.device())).contiguous();
    EXEC_NPU_CMD(aclnnCustomOp, x, fill, wrap);
    return x;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
