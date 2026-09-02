#include <tuple>
#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> BlockAttentionResidualsGrad(
    const at::Tensor &partialBlock,
    const at::Tensor &blockRes,
    const at::Tensor &projWeight,
    const at::Tensor &normWeight,
    const at::Tensor &gradHiddenStates,
    const at::Tensor &invNorm,
    const at::Tensor &probs,
    int64_t validBlockNum)
{
    const int64_t tokenCount = partialBlock.size(0);
    const int64_t hiddenSize = partialBlock.size(1);
    const int64_t blockCount = blockRes.size(1);

    at::Tensor gradPartialBlock;
    at::Tensor gradBlockRes;
    at::Tensor gradProjWeight;
    at::Tensor gradNormWeight;
    {
        auto local_device = c10::Device(partialBlock.device());
        const c10::OptionalDeviceGuard device_guard(local_device);
        gradPartialBlock = at::empty({tokenCount, hiddenSize}, partialBlock.options());
        gradBlockRes = at::empty({tokenCount, blockCount, hiddenSize}, partialBlock.options());
        gradProjWeight = at::empty({1, hiddenSize}, partialBlock.options());
        gradNormWeight = at::empty({hiddenSize}, partialBlock.options());
    }

    at::Tensor invNormCast =
        invNorm.scalar_type() == at::kFloat ? invNorm : invNorm.to(at::kFloat);
    at::Tensor probsCast =
        probs.scalar_type() == at::kFloat ? probs : probs.to(at::kFloat);

    ACLNN_CMD(
        aclnnBlockAttentionResidualsGrad,
        partialBlock, blockRes, projWeight, normWeight, gradHiddenStates,
        invNormCast, probsCast, validBlockNum,
        gradPartialBlock, gradBlockRes, gradProjWeight, gradNormWeight);

    return std::make_tuple(
        gradPartialBlock, gradBlockRes, gradProjWeight, gradNormWeight);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "block_attention_residuals_grad",
        &BlockAttentionResidualsGrad,
        "block_attention_residuals_grad",
        py::arg("partial_block"), py::arg("block_res"),
        py::arg("proj_weight"), py::arg("norm_weight"),
        py::arg("grad_hidden_states"), py::arg("inv_norm"),
        py::arg("probs"), py::arg("valid_block_num"));
}

} // namespace op_api
