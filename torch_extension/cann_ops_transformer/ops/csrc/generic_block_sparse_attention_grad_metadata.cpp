#include <torch/extension.h>
#include "aclnn_common.h"

#include <tuple>
#include <vector>
#include "aclnn_generic_block_sparse_attention_grad_metadata.h"

namespace op_api {
using OptionalTensor = c10::optional<at::Tensor>;

at::Tensor GenericBlockSparseAttentionGradMetadata(
    const at::Tensor &sparseBlockIdx, const at::Tensor &sparseBlockCount,
    const OptionalTensor &cuSeqQ, const OptionalTensor &cuSeqKv,
    const OptionalTensor &sequsedQ, const OptionalTensor &sequsedKv,
    int64_t maxQSeqlen, int64_t maxKvSeqlen, int64_t numQHeads, int64_t numKvHeads,
    int64_t headDim, const c10::optional<std::vector<int64_t>> &blockShape,
    int64_t isPackedGqa, c10::string_view layoutQ, c10::string_view layoutKv,
    int64_t maskType, int64_t softmaxPrecision, int64_t winLeft, int64_t winRight)
{
    TORCH_CHECK(sparseBlockIdx.defined() && sparseBlockCount.defined(), "sparse block tensors must be defined");
    at::Device device = sparseBlockIdx.device();
    auto output = at::empty({32 + 64 * 8}, sparseBlockIdx.options().dtype(at::kInt));
    const std::vector<int64_t> defaultShape{1, 128};
    const at::IntArrayRef shape = blockShape.has_value() ? at::IntArrayRef(*blockShape) :
                                                           at::IntArrayRef(defaultShape);
    TORCH_CHECK(shape.size() == 2, "block_shape must have two elements");
    auto to_optional = [&](const OptionalTensor &value) -> OptionalTensor {
        return value.has_value() ? value : c10::nullopt;
    };
    auto q = to_optional(cuSeqQ);
    auto kv = to_optional(cuSeqKv);
    auto usedQ = to_optional(sequsedQ);
    auto usedKv = to_optional(sequsedKv);
    std::string qLayout(layoutQ);
    std::string kvLayout(layoutKv);
    char *qLayoutPtr = const_cast<char *>(qLayout.c_str());
    char *kvLayoutPtr = const_cast<char *>(kvLayout.c_str());
    ACLNN_CMD(aclnnGenericBlockSparseAttentionGradMetadata,
              sparseBlockIdx, sparseBlockCount,
              q, kv, usedQ, usedKv, maxQSeqlen, maxKvSeqlen, numQHeads, numKvHeads,
              headDim, shape, isPackedGqa, qLayoutPtr, kvLayoutPtr, maskType,
              softmaxPrecision, winLeft, winRight, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("generic_block_sparse_attention_grad_metadata", &GenericBlockSparseAttentionGradMetadata,
          "generic_block_sparse_attention_grad_metadata");
}
} // namespace op_api
