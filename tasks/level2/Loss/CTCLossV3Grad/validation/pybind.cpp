#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */
at::Tensor ctc_loss_v3_grad(
    const at::Tensor& grad,
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    at::IntArrayRef input_lengths_list,
    at::IntArrayRef target_lengths_list,
    const at::Tensor& neg_log_likelihood,
    const at::Tensor& log_alpha,
    int64_t blank,
    bool zero_infinity)
{
    at::Tensor out = torch::empty(log_probs.sizes(), grad.options());

    // calculate the output result of the NPU
    EXEC_NPU_CMD(aclnnCtcLossBackward, grad, log_probs, targets, input_lengths_list, target_lengths_list,
                 neg_log_likelihood, log_alpha, blank, zero_infinity, out);

    return out;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("ctc_loss_v3_grad", &ctc_loss_v3_grad, "");
}