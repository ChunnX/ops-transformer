#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <string>
#include "pytorch_npu_helper.hpp"

at::Tensor custom_pybind_api(at::Tensor &bboxes, at::Tensor &gtboxes, const std::string &mode, float eps, bool aligned)
{
    at::Tensor overlap;
    if (aligned) {
        TORCH_CHECK(bboxes.dim() == 2 && bboxes.size(0) == 4, "aligned bboxes expect shape [4, N]");
        TORCH_CHECK(gtboxes.sizes() == bboxes.sizes(), "aligned gtboxes must match bboxes shape");
        int64_t n = bboxes.size(1);
        overlap = at::empty({n, 1}, bboxes.options());
    } else {
        TORCH_CHECK(bboxes.dim() == 2 && bboxes.size(1) == 4, "non-aligned bboxes expect shape [N, 4]");
        TORCH_CHECK(gtboxes.dim() == 2 && gtboxes.size(1) == 4, "non-aligned gtboxes expect shape [M, 4]");
        overlap = at::empty({gtboxes.size(0), bboxes.size(0)}, bboxes.options());
    }
    // GetWorkspaceSize 签名为 double eps，须传 double 以免 EXEC_NPU_CMD 侧 ABI 错位导致 eps 无效。
    const char *mode_c = mode.c_str();
    double eps_d = static_cast<double>(eps);
    EXEC_NPU_CMD(aclnnCustomOp, bboxes, gtboxes, mode_c, eps_d, aligned, overlap);
    return overlap;
}

PYBIND11_MODULE(kernel_gen_ops, m)
{
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "IouV2 / aclnnIou");
}
