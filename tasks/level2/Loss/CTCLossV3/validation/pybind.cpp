#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */

std::tuple<at::Tensor, at::Tensor> ctc_loss_v3(
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    at::IntArrayRef input_lengths_list,
    at::IntArrayRef target_lengths_list,
    int64_t blank,
    bool zero_infinity)
{

    int64_t max_length = 0;
    for (auto& i : target_lengths_list) {
        if (i > max_length) {
            max_length = i;
        }
    }

    int64_t time_size = log_probs.size(0);
    int64_t batch_size = log_probs.size(1);
    std::vector<int64_t> neg_log_likelihood_size = {batch_size};
    int64_t alpha_tail_size = 2 * max_length + 1;
    // Apply for a 32 byte aligned space to avoid address shifting in the OP.
    int64_t alpha_tail_size_align = (alpha_tail_size + 7) / 8 * 8;
    std::vector<int64_t> log_alpha_size = {batch_size, time_size, alpha_tail_size_align};

    // construct the output tensor of the NPU
    at::Tensor neg_log_likelihood = torch::empty(neg_log_likelihood_size, log_probs.options());
    at::Tensor log_alpha = torch::empty(log_alpha_size, log_probs.options());

    // calculate the output result of the NPU
    EXEC_NPU_CMD(aclnnCtcLoss, log_probs, targets, input_lengths_list, target_lengths_list,
                 blank, zero_infinity, neg_log_likelihood, log_alpha);

    return std::make_tuple(neg_log_likelihood, log_alpha);
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("ctc_loss_v3", &ctc_loss_v3, "");
}