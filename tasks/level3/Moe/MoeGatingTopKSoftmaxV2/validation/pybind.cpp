#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */

std::tuple<at::Tensor, at::Tensor, at::Tensor> moe_gating_top_k_softmax_v2(at::Tensor &x, at::Tensor finished_optional, int64_t k, int64_t renorm)
{
    TORCH_CHECK(x.dim() == 2, "x should be 2D tensor");
    TORCH_CHECK(k > 0, "k should be greater than 0");

    // 输出张量的形状应该是 [x.size(0), k]
    at::Tensor yOut = torch::empty({x.size(0), k}, x.options());
    // expertIdxOut应该是INT32类型，使用at::kInt（等同于torch::kInt32）
    at::Tensor expertIdxOut = torch::empty({x.size(0), k}, x.options().dtype(at::kInt));
    // softmaxResultOutOptional should always be float according to the error message
    at::Tensor softmaxResultOutOptional = torch::empty_like(x, x.options().dtype(at::kFloat));
    
    // 调用NPU算子 - 根据文档，正确的参数顺序为：x, finished_optional, k, renorm, outputSoftmaxResultFlag, yOut, expertIdxOut, softmaxResultOutOptional
    bool outputSoftmaxResultFlag = true;
    if (finished_optional.defined()) {
        TORCH_CHECK(finished_optional.dim() == 1, "finished_optional should be 1D tensor");
        TORCH_CHECK(finished_optional.size(0) == x.size(0), "finished_optional size should match x's first dimension");
        EXEC_NPU_CMD(aclnnMoeGatingTopKSoftmaxV2, x, finished_optional, k, renorm, outputSoftmaxResultFlag, yOut, expertIdxOut, softmaxResultOutOptional);
    } else {
        at::Tensor dummy_finished_optional = torch::zeros({x.size(0)}, x.options().dtype(at::kBool));
        EXEC_NPU_CMD(aclnnMoeGatingTopKSoftmaxV2, x, dummy_finished_optional, k, renorm, outputSoftmaxResultFlag, yOut, expertIdxOut, softmaxResultOutOptional);
    }
    
    return std::make_tuple(yOut, expertIdxOut, softmaxResultOutOptional);
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("moe_gating_top_k_softmax_v2", &moe_gating_top_k_softmax_v2, "");
}