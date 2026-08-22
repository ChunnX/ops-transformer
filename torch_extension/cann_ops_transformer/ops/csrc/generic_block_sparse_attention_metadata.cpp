// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").

#include <torch/extension.h>

#include "aclnn_common.h"

namespace op_api {
namespace {
constexpr int64_t METADATA_SIZE = 1024;
}

at::Tensor npu_generic_block_sparse_attention_metadata(
    const at::Tensor &sparseBlockIdx, const at::Tensor &sparseBlockCount,
    const c10::optional<at::Tensor> &cuSeqLengths, const c10::optional<at::Tensor> &cuSeqLengthsKv,
    const c10::optional<at::Tensor> &seqUsedQ, const c10::optional<at::Tensor> &seqUsedKv,
    int64_t maxQSeqLen, int64_t maxKvSeqLen, int64_t numQHeads, int64_t numKvHeads, int64_t headDim,
    at::IntArrayRef blockShape, int64_t isPackedGQA, c10::string_view qInputLayout,
    c10::string_view kvInputLayout, int64_t maskType, int64_t quantType, int64_t softmaxPrecision,
    int64_t windowSizeLeft, int64_t windowSizeRight)
{
    at::Tensor metadata = at::empty({METADATA_SIZE}, sparseBlockIdx.options().dtype(at::kInt));
    std::string qInputLayoutStr(qInputLayout);
    std::string kvInputLayoutStr(kvInputLayout);

    ACLNN_CMD(aclnnGenericBlockSparseAttentionMetadata, sparseBlockIdx, sparseBlockCount, cuSeqLengths,
        cuSeqLengthsKv, seqUsedQ, seqUsedKv, maxQSeqLen, maxKvSeqLen, numQHeads, numKvHeads, headDim,
        blockShape, isPackedGQA, qInputLayoutStr, kvInputLayoutStr, maskType, quantType, softmaxPrecision,
        windowSizeLeft, windowSizeRight, metadata);
    return metadata;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_generic_block_sparse_attention_metadata", &npu_generic_block_sparse_attention_metadata,
        "npu_generic_block_sparse_attention_metadata");
}
} // namespace op_api
