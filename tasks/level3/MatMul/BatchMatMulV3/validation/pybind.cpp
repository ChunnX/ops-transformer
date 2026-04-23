#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor batch_mat_mul_v3(at::Tensor &x1, at::Tensor &x2, at::Tensor &bias, at::Tensor &offset_w,
                            bool adj_x1, bool adj_x2, int64_t offset_x, bool enable_hf32)
{
    auto x1_sizes = x1.sizes();
    auto x2_sizes = x2.sizes();
    int64_t m = adj_x1 ? x1_sizes[x1_sizes.size() - 1] : x1_sizes[x1_sizes.size() - 2];
    int64_t n = adj_x2 ? x2_sizes[x2_sizes.size() - 2] : x2_sizes[x2_sizes.size() - 1];

    std::vector<int64_t> out_sizes;
    auto batch_sizes = x1_sizes.vec();
    batch_sizes.pop_back();
    batch_sizes.pop_back();
    out_sizes = batch_sizes;
    out_sizes.push_back(m);
    out_sizes.push_back(n);

    auto dtype = x1.scalar_type();
    if (dtype == at::kBFloat16 && x2.scalar_type() == at::kBFloat16) {
        dtype = at::kBFloat16;
    } else if (dtype == at::kFloat && x2.scalar_type() == at::kFloat) {
        dtype = at::kFloat;
    }

    at::Tensor result = torch::empty(out_sizes, x1.options().dtype(dtype));
    EXEC_NPU_CMD(aclnnBatchMatMulV3, x1, x2, bias, offset_w, adj_x1, adj_x2, offset_x, enable_hf32, result);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("batch_mat_mul_v3", &batch_mat_mul_v3, "");
}
