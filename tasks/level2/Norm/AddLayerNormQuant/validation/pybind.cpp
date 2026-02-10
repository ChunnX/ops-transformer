/**
 * @file extension_add_layer_norm_quant.cpp
 * aclnnAddLayerNormQuant: Add + LayerNorm + Quant, outputs y1, y2, x, out_scales1, out_scales2.
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"

std::vector<at::Tensor> add_layer_norm_quant(
    at::Tensor x1,
    at::Tensor x2,
    at::Tensor gamma,
    at::Tensor beta,
    c10::optional<at::Tensor> bias_opt,
    c10::optional<at::Tensor> scales1_opt,
    c10::optional<at::Tensor> scales2_opt,
    c10::optional<at::Tensor> zero_points1_opt,
    c10::optional<at::Tensor> zero_points2_opt,
    std::string quant_mode,
    double epsilon,
    bool additional_output,
    bool div_mode)
{
    torch::IntArrayRef shapes = x1.sizes();
    auto options_int8 = torch::TensorOptions().dtype(torch::kInt8).device(x1.device());
    at::Tensor y1Out = torch::empty_like(x1, options_int8);
    at::Tensor y2Out = torch::empty_like(x1, options_int8);
    at::Tensor xOut = torch::empty_like(x1);

    std::vector<int64_t> scale_shape(shapes.begin(), shapes.end() - 1);
    if (scale_shape.empty()) {
        scale_shape.push_back(1);
    }
    auto options_float = torch::TensorOptions().dtype(torch::kFloat).device(x1.device());
    at::Tensor outScales1Out = torch::empty(scale_shape, options_float);
    at::Tensor outScales2Out = torch::empty(scale_shape, options_float);

    at::Tensor x1_cont = x1.contiguous();
    at::Tensor x2_cont = x2.contiguous();
    at::Tensor gamma_cont = gamma.contiguous();
    at::Tensor beta_cont = beta.contiguous();
    at::Tensor bias = bias_opt.has_value() ? bias_opt.value().contiguous() : at::Tensor();
    at::Tensor scales1 = scales1_opt.has_value() ? scales1_opt.value().contiguous() : at::Tensor();
    at::Tensor scales2 = scales2_opt.has_value() ? scales2_opt.value().contiguous() : at::Tensor();
    at::Tensor zero_points1 = zero_points1_opt.has_value() ? zero_points1_opt.value().contiguous() : at::Tensor();
    at::Tensor zero_points2 = zero_points2_opt.has_value() ? zero_points2_opt.value().contiguous() : at::Tensor();

    const char* quant_mode_cstr = quant_mode.c_str();
    double epsilon_val = epsilon;
    bool additional_output_val = additional_output;
    bool div_mode_val = div_mode;

    EXEC_NPU_CMD(aclnnAddLayerNormQuant,
                 x1_cont, x2_cont, gamma_cont, beta_cont,
                 bias, scales1, scales2, zero_points1, zero_points2,
                 quant_mode_cstr, epsilon_val, additional_output_val, div_mode_val,
                 y1Out, y2Out, xOut, outScales1Out, outScales2Out);

    std::vector<at::Tensor> result = {y1Out, y2Out, xOut, outScales1Out, outScales2Out};
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("add_layer_norm_quant", &add_layer_norm_quant, "");
}
