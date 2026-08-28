/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>

#include "aclnn_common.h"

namespace op_api {
namespace {
constexpr int64_t METADATA_SIZE = 1024;
}

at::Tensor NpuGenericBlockSparseAttentionMetadata(
    const at::Tensor &sparseBlockIdx, const at::Tensor &sparseBlockCount, const c10::optional<at::Tensor> &cuSeqLengths,
    const c10::optional<at::Tensor> &cuSeqLengthsKv, const c10::optional<at::Tensor> &seqUsedQ,
    const c10::optional<at::Tensor> &seqUsedKv, int64_t maxQSeqLen, int64_t maxKvSeqLen, int64_t numQHeads,
    int64_t numKvHeads, int64_t headDim, at::IntArrayRef blockShape, int64_t isPackedGQA, c10::string_view qInputLayout,
    c10::string_view kvInputLayout, int64_t maskType, int64_t quantType, int64_t softmaxPrecision,
    int64_t windowSizeLeft, int64_t windowSizeRight)
{
    at::Tensor metadata = at::empty({METADATA_SIZE}, sparseBlockIdx.options().dtype(at::kInt));
    std::string qInputLayoutStr(qInputLayout);
    std::string kvInputLayoutStr(kvInputLayout);
    char *qInputLayoutPtr = const_cast<char *>(qInputLayoutStr.c_str());
    char *kvInputLayoutPtr = const_cast<char *>(kvInputLayoutStr.c_str());

    ACLNN_CMD(aclnnGenericBlockSparseAttentionMetadata, sparseBlockIdx, sparseBlockCount, cuSeqLengths, cuSeqLengthsKv,
              seqUsedQ, seqUsedKv, maxQSeqLen, maxKvSeqLen, numQHeads, numKvHeads, headDim, blockShape, isPackedGQA,
              qInputLayoutPtr, kvInputLayoutPtr, maskType, quantType, softmaxPrecision, windowSizeLeft, windowSizeRight,
              metadata);
    return metadata;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_generic_block_sparse_attention_metadata", &NpuGenericBlockSparseAttentionMetadata,
          "npu_generic_block_sparse_attention_metadata");
}
} // namespace op_api
