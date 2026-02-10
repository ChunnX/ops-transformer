/**
 * @file pybind_masked_softmax_with_rel_pos_bias.cpp
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"

/**
 * MaskedSoftmaxWithRelPosBias: out = softmax(scaleValue * x + attenMaskOptional + relativePosBias)
 * aclnnMaskedSoftmaxWithRelPosBiasGetWorkspaceSize(x, attenMaskOptional, relativePosBias, scaleValue, innerPrecisionMode, out, &workspaceSize, &executor)
 * aclnnMaskedSoftmaxWithRelPosBias(workspace, workspaceSize, executor, stream)
 */
at::Tensor masked_softmax_with_rel_pos_bias(
    at::Tensor x,
    c10::optional<at::Tensor> atten_mask_optional,
    at::Tensor relative_pos_bias,
    double scale_value) {
  const int64_t inner_precision_mode = 0;
  at::Tensor out = torch::empty_like(x);
  c10::optional<at::Tensor> atten_opt = (atten_mask_optional.has_value() && atten_mask_optional->defined())
      ? c10::optional<at::Tensor>(atten_mask_optional->contiguous())
      : c10::nullopt;
  EXEC_NPU_CMD(
      aclnnMaskedSoftmaxWithRelPosBias,
      x,
      atten_opt,
      relative_pos_bias,
      scale_value,
      inner_precision_mode,
      out);
  return out;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
  m.doc() = "Python bindings for kernel_gen_ops";
  m.def("masked_softmax_with_rel_pos_bias", &masked_softmax_with_rel_pos_bias, "");
}
