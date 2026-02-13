/**
 * @file pybind.cpp
 * DuaQuantizeAddLayerNorm: def 顺序为 inputs -> attrs(dtype, axis, epsilon, additional_output) -> outputs
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"

/**
 * aclnnCustomOp 将在 batch_compile 时替换为 aclnnDuaQuantizeAddLayerNorm
 * GetWorkspaceSize 参数顺序须与 opbuild 生成一致：x1,x2,gamma,beta,bias,scales1,scales2,zp1,zp2, dtype,axis,epsilon,additional_output, y1,y2,x
 */
std::vector<at::Tensor> custom_pybind_api(
    at::Tensor x1, at::Tensor x2, at::Tensor gamma, at::Tensor beta, at::Tensor bias,
    at::Tensor scales1, at::Tensor scales2,
    c10::optional<at::Tensor> zero_points1, c10::optional<at::Tensor> zero_points2,
    int64_t dtype, int64_t axis, float epsilon, bool additional_output)
{
    auto y1_options = torch::TensorOptions().dtype(torch::kInt8).device(x1.device());
    at::Tensor y1 = torch::empty_like(x1, y1_options);
    at::Tensor y2 = torch::empty_like(x1, y1_options);
    at::Tensor x_out = torch::empty_like(x1);

    at::Tensor zp1 = zero_points1.has_value() && zero_points1.value().defined()
                         ? zero_points1.value().contiguous()
                         : at::Tensor();
    at::Tensor zp2 = zero_points2.has_value() && zero_points2.value().defined()
                         ? zero_points2.value().contiguous()
                         : at::Tensor();

    EXEC_NPU_CMD(aclnnCustomOp, x1, x2, gamma, beta, bias, scales1, scales2,
                 zp1, zp2, dtype, axis, epsilon, additional_output,
                 y1, y2, x_out);

    return {y1, y2, x_out};
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("dua_quantize_add_layer_norm", &custom_pybind_api, "");
}
