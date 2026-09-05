/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file generic_block_sparse_attention_grad.cpp
 * \brief PyTorch ACLNN binding for GenericBlockSparseAttentionGrad.
 */

#include <torch/extension.h>

#include "aclnn_common.h"

namespace op_api {
namespace {
// Keep in sync with generic_block_sparse_attention_grad_metadata.h
constexpr int64_t GSAG_TASK_LIST_OFFSET = 80; // 8 + 2 * 36
constexpr int64_t GSAG_TASK_ENTRY_SIZE = 4;

inline int64_t CalcGsagMetadataSize(int64_t batchSize, int64_t numQHeads, int64_t numJ)
{
    return GSAG_TASK_LIST_OFFSET + batchSize * numQHeads * numJ * GSAG_TASK_ENTRY_SIZE;
}
} // namespace

at::Tensor GenericBlockSparseAttentionGradMetadata(
    const at::Tensor &rsvdBlockIdx, const at::Tensor &rsvdBlockCount, const c10::optional<at::Tensor> &cuSeqLengths,
    const c10::optional<at::Tensor> &cuSeqLengthsKv, const c10::optional<at::Tensor> &sequsedQ,
    const c10::optional<at::Tensor> &sequsedKv, int64_t maxQSeqlen, int64_t maxKvSeqlen, int64_t numQHeads,
    int64_t numKvHeads, int64_t headDim, at::IntArrayRef blockShape, int64_t isPackedGqa,
    const std::string &qInputLayout, const std::string &kvInputLayout, int64_t maskType, int64_t softmaxPrecision,
    int64_t windowSizeLeft, int64_t windowSizeRight)
{
    TORCH_CHECK(rsvdBlockIdx.dim() == 4, "rsvd_block_idx must be 4D [B, N2, J, maxS1]");
    TORCH_CHECK(numQHeads > 0, "num_heads_q must be > 0");
    const int64_t batchSize = rsvdBlockIdx.size(0);
    const int64_t numJ = rsvdBlockIdx.size(2);
    const int64_t metadataSize = CalcGsagMetadataSize(batchSize, numQHeads, numJ);

    at::Tensor metadata =
        torch::empty({metadataSize}, torch::TensorOptions().dtype(torch::kInt64).device(rsvdBlockIdx.device()));

    const at::Tensor &cuSeqLengthsValue = cuSeqLengths.value_or(at::Tensor());
    const at::Tensor &cuSeqLengthsKvValue = cuSeqLengthsKv.value_or(at::Tensor());
    const at::Tensor &sequsedQValue = sequsedQ.value_or(at::Tensor());
    const at::Tensor &sequsedKvValue = sequsedKv.value_or(at::Tensor());

    char *qInputLayoutPtr = const_cast<char *>(qInputLayout.c_str());
    char *kvInputLayoutPtr = const_cast<char *>(kvInputLayout.c_str());

    ACLNN_CMD(aclnnGenericBlockSparseAttentionGradMetadata, rsvdBlockIdx, rsvdBlockCount, cuSeqLengthsValue,
              cuSeqLengthsKvValue, sequsedQValue, sequsedKvValue, maxQSeqlen, maxKvSeqlen, numQHeads, numKvHeads,
              headDim, blockShape, isPackedGqa, qInputLayoutPtr, kvInputLayoutPtr, maskType, softmaxPrecision,
              windowSizeLeft, windowSizeRight, metadata);

    return metadata;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> GenericBlockSparseAttentionGrad(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &dout,
    const at::Tensor &out, const at::Tensor &lse, const at::Tensor &rsvdBlockIdx, const at::Tensor &rsvdBlockCount,
    const at::Tensor &metadata, const c10::optional<at::Tensor> &attenMask,
    const c10::optional<at::Tensor> &cuSeqLengths, const c10::optional<at::Tensor> &cuSeqLengthsKv,
    const c10::optional<at::Tensor> &sequsedQ, const c10::optional<at::Tensor> &sequsedKv, at::IntArrayRef blockShape,
    int64_t isPackedGqa, const std::string &qInputLayout, const std::string &kvInputLayout, double scaleValue,
    int64_t maskType, int64_t softmaxPrecision, int64_t windowSizeLeft, int64_t windowSizeRight)
{
    const at::Tensor &attenMaskValue = attenMask.value_or(at::Tensor());
    const at::Tensor &cuSeqLengthsValue = cuSeqLengths.value_or(at::Tensor());
    const at::Tensor &cuSeqLengthsKvValue = cuSeqLengthsKv.value_or(at::Tensor());
    const at::Tensor &sequsedQValue = sequsedQ.value_or(at::Tensor());
    const at::Tensor &sequsedKvValue = sequsedKv.value_or(at::Tensor());

    at::Tensor dQuery = at::empty_like(query);
    at::Tensor dKey = at::empty_like(key);
    at::Tensor dValue = at::empty_like(value);

    char *qInputLayoutPtr = const_cast<char *>(qInputLayout.c_str());
    char *kvInputLayoutPtr = const_cast<char *>(kvInputLayout.c_str());

    ACLNN_CMD(aclnnGenericBlockSparseAttentionGrad, query, key, value, dout, out, lse, rsvdBlockIdx, rsvdBlockCount,
              metadata, attenMaskValue, cuSeqLengthsValue, cuSeqLengthsKvValue, sequsedQValue, sequsedKvValue,
              blockShape, isPackedGqa, qInputLayoutPtr, kvInputLayoutPtr, scaleValue, maskType, softmaxPrecision,
              windowSizeLeft, windowSizeRight, dQuery, dKey, dValue);

    return std::make_tuple(dQuery, dKey, dValue);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("generic_block_sparse_attention_grad_metadata", &GenericBlockSparseAttentionGradMetadata,
          "generic_block_sparse_attention_grad_metadata");
    m.def("generic_block_sparse_attention_grad", &GenericBlockSparseAttentionGrad,
          "generic_block_sparse_attention_grad");
}
} // namespace op_api
