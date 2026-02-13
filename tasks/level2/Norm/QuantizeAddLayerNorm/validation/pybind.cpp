/**
 * @file pybind.cpp
 * QuantizeAddLayerNorm: def 顺序为 inputs -> attrs(dtype, axis, epsilon, additional_output) -> outputs
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"

/**
 * aclnnCustomOp 将在 batch_compile 时替换为 aclnnQuantizeAddLayerNorm
 * GetWorkspaceSize 参数顺序须与 op 一致：x1,x2,gamma,beta,bias,scales,zp, dtype,axis,epsilon,additional_output, y,x
 */
std::vector<at::Tensor> custom_pybind_api(
    at::Tensor x1, at::Tensor x2, at::Tensor gamma, at::Tensor beta, at::Tensor bias,
    at::Tensor scales,
    c10::optional<at::Tensor> zero_points,
    int64_t dtype, int64_t axis, float epsilon, bool additional_output)
{
    auto y_options = torch::TensorOptions().dtype(torch::kInt8).device(x1.device());
    at::Tensor y = torch::empty_like(x1, y_options);
    at::Tensor x_out = torch::empty_like(x1);

    at::Tensor zp = zero_points.has_value() && zero_points.value().defined()
                        ? zero_points.value().contiguous()
                        : at::Tensor();

    EXEC_NPU_CMD(aclnnCustomOp, x1, x2, gamma, beta, bias, scales, zp,
                 dtype, axis, epsilon, additional_output,
                 y, x_out);

    return {y, x_out};
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("quantize_add_layer_norm", &custom_pybind_api, "");
}
