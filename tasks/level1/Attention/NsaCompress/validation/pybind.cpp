#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <vector>
#include "pytorch_npu_helper.hpp"

static int64_t compress_out_rows_cpu(const at::Tensor &act_cumsum_1d, int64_t L, int64_t S) {
    auto cpu = act_cumsum_1d.cpu().contiguous();
    int64_t n = cpu.numel();
    int64_t total = 0;
    int64_t pre = 0;
    auto acc = cpu.data_ptr<int64_t>();
    for (int64_t i = 0; i < n; ++i) {
        int64_t end = acc[i];
        int64_t cur = end - pre;
        if (cur >= L) {
            total += (cur - L + S) / S;
        }
        pre = end;
    }
    return total;
}

at::Tensor nsa_compress(
    const at::Tensor &input,
    const at::Tensor &weight,
    const at::Tensor &act_seq_len_cumsum,
    int64_t compress_block_size,
    int64_t compress_stride,
    int64_t act_seq_len_type) {
    TORCH_CHECK(input.dim() == 3, "input must be [T,N,D]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [compressBlockSize, N]");
    TORCH_CHECK(act_seq_len_cumsum.dim() == 1, "act_seq_len_cumsum must be 1D int64 cumsum");

    // Precision 路径上张量在 NPU：必须先拷到 CPU 再读 host 指针，否则 data_ptr 指向设备内存会 SIGSEGV。
    at::Tensor act_cpu = act_seq_len_cumsum.cpu().contiguous();
    std::vector<int64_t> seq_vec(
        act_cpu.data_ptr<int64_t>(),
        act_cpu.data_ptr<int64_t>() + act_cpu.numel());
    aclIntArray *act_acl = ConvertType(at::IntArrayRef(seq_vec));

    std::vector<char> layout_buf = {'T', 'N', 'D', '\0'};
    char *layout = layout_buf.data();

    int64_t m = compress_out_rows_cpu(act_cpu, compress_block_size, compress_stride);
    int64_t n = input.size(1);
    int64_t d = input.size(2);
    at::Tensor output = at::empty({m, n, d}, input.options());

    EXEC_NPU_CMD(
        aclnnCustomOp,
        input,
        weight,
        act_acl,
        layout,
        compress_block_size,
        compress_stride,
        act_seq_len_type,
        output);

    return output;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("nsa_compress", &nsa_compress, "NsaCompress forward");
}
