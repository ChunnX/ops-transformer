#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <vector>
#include "pytorch_npu_helper.hpp"

std::tuple<at::Tensor, at::Tensor> nsa_compress_grad(
    const at::Tensor &output_grad,
    const at::Tensor &input,
    const at::Tensor &weight,
    const at::Tensor &act_seq_len_cumsum,
    int64_t compress_block_size,
    int64_t compress_stride,
    int64_t act_seq_len_type) {
    TORCH_CHECK(output_grad.dim() == 3, "output_grad must be [M,N,D]");
    TORCH_CHECK(input.dim() == 3, "input must be [T,N,D]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [compressBlockSize, N]");
    TORCH_CHECK(act_seq_len_cumsum.dim() == 1, "act_seq_len_cumsum must be 1D int64");

    at::Tensor act_cpu = act_seq_len_cumsum.cpu().contiguous();
    std::vector<int64_t> seq_vec(
        act_cpu.data_ptr<int64_t>(),
        act_cpu.data_ptr<int64_t>() + act_cpu.numel());
    aclIntArray *act_acl = ConvertType(at::IntArrayRef(seq_vec));

    std::vector<char> layout_buf = {'T', 'N', 'D', '\0'};
    char *layout = layout_buf.data();

    at::Tensor input_grad = at::empty_like(input);
    at::Tensor weight_grad = at::empty_like(weight);

    EXEC_NPU_CMD(
        aclnnCustomOp,
        output_grad,
        input,
        weight,
        act_acl,
        compress_block_size,
        compress_stride,
        act_seq_len_type,
        layout,
        input_grad,
        weight_grad);

    return std::make_tuple(input_grad, weight_grad);
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("nsa_compress_grad", &nsa_compress_grad, "NsaCompressGrad backward");
}
