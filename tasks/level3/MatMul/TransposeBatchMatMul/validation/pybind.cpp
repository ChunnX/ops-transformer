#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor transpose_batch_mat_mul(
    at::Tensor &x1, at::Tensor &x2,
    c10::optional<at::Tensor> &bias,
    c10::optional<at::Tensor> &scale,
    at::IntArrayRef perm_x1,
    at::IntArrayRef perm_x2,
    at::IntArrayRef perm_y,
    bool enable_hf32,
    int64_t batch_split_factor)
{
    auto dtype = x1.scalar_type();
    if (dtype == at::kFloat && x2.scalar_type() == at::kFloat) {
        dtype = at::kFloat;
    } else if (dtype == at::kBFloat16 && x2.scalar_type() == at::kBFloat16) {
        dtype = at::kBFloat16;
    }

    // Infer output shape from perm_y
    // With perm_y = {1, 0, 2}, output is (m, batch, n) or (batch_split_factor, m, batch_n / batch_split_factor)
    auto x1_sizes = x1.sizes();
    auto x2_sizes = x2.sizes();

    // Transpose x1 according to perm_x1
    int64_t m, k, batch, n;
    // perm_x1 determines how to read x1 dims
    // For perm_x1={0,1,2}: x1 is (batch, m, k)
    // For perm_x1={1,0,2}: x1 is (m, batch, k) -> transposed to (batch, m, k)
    int64_t x1_dim0 = x1_sizes[perm_x1[0]];
    int64_t x1_dim1 = x1_sizes[perm_x1[1]];
    int64_t x1_dim2 = x1_sizes[perm_x1[2]];
    batch = x1_dim0;
    m = x1_dim1;
    k = x1_dim2;

    // perm_x2 determines how to read x2 dims
    int64_t x2_dim0 = x2_sizes[perm_x2[0]];
    int64_t x2_dim1 = x2_sizes[perm_x2[1]];
    int64_t x2_dim2 = x2_sizes[perm_x2[2]];
    n = x2_dim2;

    // Matmul result shape is (batch, m, n)
    std::vector<int64_t> matmul_shape = {batch, m, n};
    std::vector<int64_t> out_sizes;
    if (batch_split_factor > 1) {
        out_sizes = {batch_split_factor, m, batch * n / batch_split_factor};
    } else if (scale.has_value() && scale.value().defined() && scale.value().numel() > 0) {
        out_sizes = {m, 1, batch * n};
    } else {
        // Apply perm_y to matmul result shape (batch, m, n)
        out_sizes = {matmul_shape[perm_y[0]], matmul_shape[perm_y[1]], matmul_shape[perm_y[2]]};
    }

    at::Tensor result = torch::empty(out_sizes, x1.options().dtype(dtype));

    at::Tensor bias_tensor = bias.has_value() && bias.value().defined() ? bias.value() : at::Tensor();
    at::Tensor scale_tensor = scale.has_value() && scale.value().defined() ? scale.value() : at::Tensor();

    EXEC_NPU_CMD(aclnnTransposeBatchMatMul, x1, x2, bias_tensor, scale_tensor,
                 perm_x1, perm_x2, perm_y, enable_hf32, batch_split_factor, result);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("transpose_batch_mat_mul", &transpose_batch_mat_mul, "");
}
