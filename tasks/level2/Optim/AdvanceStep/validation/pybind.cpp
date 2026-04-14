#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

std::vector<at::Tensor> advance_step(at::Tensor &input_tokens, at::Tensor &sampled_token_ids,
                                      at::Tensor &input_positions, at::Tensor &seq_lens,
                                      at::Tensor &slot_mapping, at::Tensor &block_tables,
                                      int64_t num_seqs, int64_t num_queries, int64_t block_size)
{
    EXEC_NPU_CMD(aclnnAdvanceStep, input_tokens, sampled_token_ids, input_positions,
                  seq_lens, slot_mapping, block_tables, num_seqs, num_queries, block_size);

    return {input_tokens, input_positions, seq_lens, slot_mapping};
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("advance_step", &advance_step, "AdvanceStep operation");
}
