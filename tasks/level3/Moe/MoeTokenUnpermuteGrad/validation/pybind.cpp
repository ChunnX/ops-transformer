#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
* register forward implementation for NPU device
* 使用说明：
* 1. 将此文件中的 aclnnCustomOp 替换为实际算子名称，如 aclnnForeachExp
* 2. 将 custom_pybind_api 替换为对应的下划线命名形式，如 foreach_exp
*
* 替换示例：
* - aclnnCustomOp -> aclnnXXX (其中XXX为算子名，如替换为aclnnForeachExp)
* - custom_pybind_api -> YYY (其中YYY为算子名的下划线形式，如foreach_exp)
*
* 注意：替换时需保持函数签名和逻辑不变，仅修改上述指定的名称，这一替换过程将在batch_compile.py文件中自动被执行
*/

std::vector<at::Tensor> custom_pybind_api(
    const at::Tensor& permuted_tokens,
    const at::Tensor& unpermuted_tokens_grad,
    const at::Tensor& sorted_indices,
    c10::optional<at::Tensor> probs,
    bool padded_mode)
{
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
    // permutedTokensGradOut: (tokens_num * top_k, hidden_size)
    c10::TensorOptions options = permuted_tokens.options().dtype(permuted_tokens.scalar_type());
    at::Tensor permuted_tokens_grad = at::empty({tokens_num_times_topk, hidden_size}, options);

    // probsGradOut: (tokens_num, top_k)
    at::TensorOptions probs_options = permuted_tokens.options();
    if (probs.has_value()) {
        probs_options = probs_options.dtype(probs.value().scalar_type());
    }
    at::Tensor probs_grad = at::empty({tokens_num, top_k}, probs_options);


    // restoreShapeOptional 目前仅支持 nullptr，使用空数组
    aclIntArray* restore_shape_acl = nullptr;

    // 调用 NPU 算子 paddedMode 目前仅支持 false
    if (probs.has_value()) {
        EXEC_NPU_CMD(aclnnCustomOp, permuted_tokens, unpermuted_tokens_grad, sorted_indices,
                     probs.value(), padded_mode, restore_shape_acl, permuted_tokens_grad, probs_grad);
    } else {
        // probs 为空时，传入空张量
        at::Tensor empty_probs = at::Tensor();
        EXEC_NPU_CMD(aclnnCustomOp, permuted_tokens, unpermuted_tokens_grad, sorted_indices,
                     empty_probs, padded_mode, restore_shape_acl, permuted_tokens_grad, probs_grad);
    }

    return {permuted_tokens_grad, probs_grad};
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    // Function name in Python will be `moe_token_unpermute_grad`
    m.def("moe_token_unpermute_grad", &custom_pybind_api, "");
}
