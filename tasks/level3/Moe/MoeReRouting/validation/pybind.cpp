#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

std::vector<at::Tensor> moe_re_routing(
    const at::Tensor& tokens,
    const at::Tensor& expert_token_num_per_rank,
    const c10::optional<at::Tensor>& per_token_scales,
    int64_t expert_token_num_type,
    int64_t idx_type)
{
    if (expert_token_num_type != 1) {
        throw std::runtime_error("expert_token_num_type only support count mode");
    }
    if (idx_type != 0) {
        throw std::runtime_error("idx_type only support gather mode");
    }

    at::TensorOptions options = tokens.options();
    at::Tensor permute_tokens = at::empty(tokens.sizes(), options);
    at::Tensor permute_token_idx = at::empty({tokens.size(0)}, tokens.options().dtype(torch::kInt32));

    int64_t E = expert_token_num_per_rank.size(1);
    at::Tensor expert_token_num = at::empty({E}, expert_token_num_per_rank.options());

    // 算子始终写出 permute_per_token_scales（infershape 为 [A]）；无输入 scale 时仍需合法输出缓冲。
    at::Tensor permute_per_token_scales =
        at::empty({tokens.size(0)}, tokens.options().dtype(torch::kFloat32));

    EXEC_NPU_CMD(aclnnCustomOp,
        tokens,
        expert_token_num_per_rank,
        per_token_scales,
        expert_token_num_type,
        idx_type,
        permute_tokens,
        permute_per_token_scales,
        permute_token_idx,
        expert_token_num);

    std::vector<at::Tensor> outputs;
    outputs.push_back(permute_tokens);
    if (per_token_scales.has_value() && per_token_scales->defined() && per_token_scales->numel() > 0) {
        outputs.push_back(permute_per_token_scales);
    }
    outputs.push_back(permute_token_idx);
    outputs.push_back(expert_token_num);
    return outputs;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.def("moe_re_routing", &moe_re_routing, "MoeReRouting operator");
}
