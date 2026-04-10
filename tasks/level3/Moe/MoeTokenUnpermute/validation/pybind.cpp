#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor moe_token_unpermute(
    const at::Tensor& permuted_tokens,
    const at::Tensor& sorted_indices,
    c10::optional<at::Tensor> probs,
    bool padded_mode
) {
    // 获取形状信息
    int64_t tokens_num_times_topk = permuted_tokens.size(0);
    int64_t hidden_size = permuted_tokens.size(1);

    // 推断 tokens_num 和 top_k
    int64_t tokens_num = 0;
    int64_t top_k = 0;
    if (probs.has_value()) {
        tokens_num = probs.value().size(0);
        top_k = probs.value().size(1);
    } else {
        tokens_num = tokens_num_times_topk;
        top_k = 1;
    }

    // 创建输出张量
    at::TensorOptions options = permuted_tokens.options();
    at::Tensor output = at::empty({tokens_num, hidden_size}, options);
    
    // 处理可选参数
    at::Tensor probs_tensor = at::Tensor();
    if (probs.has_value()) {
        probs_tensor = probs.value();
    }
    
    // restoreShapeOptional 目前仅支持 nullptr，使用空数组
    aclIntArray* restore_shape_acl = nullptr;
    // padded_mode当前仅支持false
    if (padded_mode) {
        throw std::runtime_error("padded_mode only support false");
    }

    // 调用 ACLNN 接口
    EXEC_NPU_CMD(
        aclnnCustomOp,
        permuted_tokens,
        sorted_indices,
        probs_tensor,
        padded_mode,
        restore_shape_acl,
        output
    );
    
    return output;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("moe_token_unpermute", &moe_token_unpermute, "MoeTokenUnpermute operator");
}