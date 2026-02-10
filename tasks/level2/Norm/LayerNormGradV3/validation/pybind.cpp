/**
 * @file pybind.cpp
 * aclnnLayerNormBackward: backward of LayerNorm, outputs dx, dgamma, dbeta.
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"

std::vector<at::Tensor> layer_norm_grad_v3(
    at::Tensor dy,
    at::Tensor x,
    at::IntArrayRef normalized_shape,
    at::Tensor mean,
    at::Tensor rstd,
    c10::optional<at::Tensor> weight_optional,
    c10::optional<at::Tensor> bias_optional)
{
    TORCH_CHECK(dy.defined() && x.defined() && mean.defined() && rstd.defined(),
                "layer_norm_grad_v3: dy, x, mean, rstd must be defined");
    dy = dy.contiguous();
    x = x.contiguous();
    mean = mean.contiguous();
    rstd = rstd.contiguous();

    at::Tensor dxOut = torch::empty_like(x, x.options().memory_format(c10::MemoryFormat::Contiguous));

    // dgamma/dbeta have shape normalized_shape
    std::vector<int64_t> norm_shape_vec(normalized_shape.begin(), normalized_shape.end());
    at::TensorOptions opts = x.options();
    if (weight_optional.has_value() && weight_optional.value().defined()) {
        opts = weight_optional.value().options();
    }
    at::Tensor dgammaOut = torch::empty(norm_shape_vec, opts.memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor dbetaOut = torch::empty(norm_shape_vec, opts.memory_format(c10::MemoryFormat::Contiguous));

    aclIntArray* normalizedShape_acl = ConvertType(normalized_shape);
    at::Tensor weight_npu = weight_optional.has_value() && weight_optional.value().defined()
                                ? weight_optional.value().contiguous()
                                : at::Tensor();
    at::Tensor bias_npu = bias_optional.has_value() && bias_optional.value().defined()
                              ? bias_optional.value().contiguous()
                              : at::Tensor();
    std::array<bool, 3> output_mask = {true, true, true};
    aclBoolArray* outputMask_acl = ConvertType(output_mask);

    EXEC_NPU_CMD(aclnnLayerNormBackward,
                 dy, x, normalizedShape_acl, mean, rstd,
                 weight_npu, bias_npu, outputMask_acl,
                 dxOut, dgammaOut, dbetaOut);

    std::vector<at::Tensor> result = {dxOut, dgammaOut, dbetaOut};
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("layer_norm_grad_v3", &layer_norm_grad_v3, "");
}
