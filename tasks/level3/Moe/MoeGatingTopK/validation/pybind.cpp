#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */

std::tuple<at::Tensor, at::Tensor, at::Tensor> moe_gating_top_k(at::Tensor &x, at::Tensor finished_optional, int64_t k)
{
    TORCH_CHECK(x.dim() == 2, "x should be 2D tensor");
    TORCH_CHECK(k > 0, "k should be greater than 0");

    at::Tensor out = torch::empty({x.size(0), k}, x.options());
    at::Tensor indices = torch::empty({x.size(0), k}, x.options().dtype(at::kInt));
    at::Tensor source_row_out = torch::empty({x.size(0), k}, x.options().dtype(at::kInt));

    if (finished_optional.defined()) {
        TORCH_CHECK(finished_optional.dim() == 1, "finished_optional should be 1D tensor");
        TORCH_CHECK(finished_optional.size(0) == x.size(0), "finished_optional size should match x's first dimension");
        EXEC_NPU_CMD(aclnnMoeGatingTopKSoftmax, x, finished_optional, k, out, indices, source_row_out);
    } else {
        // Create a dummy tensor for finished_optional when it's None
        at::Tensor dummy_finished_optional = torch::zeros({x.size(0)}, x.options().dtype(at::kBool));
        EXEC_NPU_CMD(aclnnMoeGatingTopKSoftmax, x, dummy_finished_optional, k, out, indices, source_row_out);
    }

    return std::make_tuple(out, indices, source_row_out);
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("moe_gating_top_k", &moe_gating_top_k, "");
}