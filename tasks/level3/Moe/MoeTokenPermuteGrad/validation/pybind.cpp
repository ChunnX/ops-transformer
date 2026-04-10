#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor moe_token_permute_grad(
    const at::Tensor& permuted_output_grad,
    const at::Tensor& sorted_indices,
    int64_t num_topk,
    bool padded_mode
) {
    // 创建输出张量
    at::TensorOptions options = permuted_output_grad.options();
    at::Tensor output = at::empty({permuted_output_grad.size(0) / num_topk, permuted_output_grad.size(1)}, options);
    
    // 调用 ACLNN 接口
    EXEC_NPU_CMD(
        aclnnCustomOp,
        permuted_output_grad,
        sorted_indices,
        num_topk,
        padded_mode,
        output
    );
    
    return output;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("moe_token_permute_grad", &moe_token_permute_grad, "MoeTokenPermuteGrad operator");
}