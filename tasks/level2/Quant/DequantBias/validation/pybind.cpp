#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <cstdint>
#include <vector>

#include "pytorch_npu_helper.hpp"

/**
 * 与 aclnnDequantBias 文档一致：outputDtype 为 int64（ge::DT_FLOAT16=1，BF16=27）。
 * batch_compile 将 aclnnCustomOp 替换为 aclnnDequantBias。
 */
std::vector<at::Tensor> custom_pybind_api(
    const at::Tensor &x,
    const at::Tensor &weight_scale,
    c10::optional<at::Tensor> activate_scale,
    c10::optional<at::Tensor> bias,
    int64_t output_dtype)
{
    at::ScalarType y_dtype = (output_dtype == 27) ? at::kBFloat16 : at::kHalf;
    at::Tensor y = at::empty(
        {x.size(0), x.size(1)},
        at::TensorOptions().dtype(y_dtype).device(x.device()));
    int64_t output_dtype_arg = output_dtype;
    EXEC_NPU_CMD(
        aclnnCustomOp,
        x,
        weight_scale,
        activate_scale,
        bias,
        output_dtype_arg,
        y);
    return {y};
}

PYBIND11_MODULE(kernel_gen_ops, m)
{
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
