#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */

at::Tensor embedding_dense_grad_v2_pybind_api(
    at::Tensor &grad,
    at::Tensor &sort_indices,
    at::Tensor &pos_idx,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq)
{
    // 获取 embedding_dim
    int64_t embedding_dim = grad.size(-1);
    
    // 分配输出张量
    at::Tensor result = torch::zeros({num_weights, embedding_dim}, grad.options());
    
    // 调用 aclnn 接口
    EXEC_NPU_CMD(aclnnEmbeddingDenseGradV2, grad, sort_indices, pos_idx,  
                 num_weights, padding_idx, scale_grad_by_freq, result);
    
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("embedding_dense_grad_v2", &embedding_dense_grad_v2_pybind_api, 
          "EmbeddingDenseGradV2 custom operator");
}