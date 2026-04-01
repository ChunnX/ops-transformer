#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor custom_pybind_api(
    at::Tensor &features,
    at::Tensor &features_batch_cnt,
    at::Tensor &indices,
    at::Tensor &indices_batch_cnt)
{
    TORCH_CHECK(features.dim() == 2, "features must be 2D (N, C)");
    TORCH_CHECK(features_batch_cnt.dim() == 1, "features_batch_cnt must be 1D");
    TORCH_CHECK(indices.dim() == 2, "indices must be 2D (M, nsample)");
    TORCH_CHECK(indices_batch_cnt.dim() == 1, "indices_batch_cnt must be 1D");

    int64_t m = indices.size(0);
    int64_t nsample = indices.size(1);
    int64_t c = features.size(1);
    at::Tensor out = at::empty({m, c, nsample}, features.options());
    EXEC_NPU_CMD(
        aclnnCustomOp,
        features,
        features_batch_cnt,
        indices,
        indices_batch_cnt,
        out);
    return out;
}

PYBIND11_MODULE(kernel_gen_ops, m)
{
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "StackGroupPoints / aclnnStackGroupPoints");
}
