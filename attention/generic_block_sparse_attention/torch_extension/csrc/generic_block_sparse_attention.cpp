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

std::tuple<at::Tensor, at::Tensor> GenericBlockSparseAttention(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &sparseBlockIdx,
    const at::Tensor &sparseBlockCount, std::vector<int64_t> blockShape, const c10::optional<at::Tensor> &metadata,
    const c10::optional<at::Tensor> &attenMask, const c10::optional<at::Tensor> &qDequantScale,
    const c10::optional<at::Tensor> &kDequantScale, const c10::optional<at::Tensor> &vDequantScale,
    const c10::optional<at::Tensor> &pQuantScale, const c10::optional<at::Tensor> &cuSeqLengthsQ,
    const c10::optional<at::Tensor> &cuSeqLengthsKv, const c10::optional<at::Tensor> &seqUsedQ,
    const c10::optional<at::Tensor> &seqUsedKv, const c10::optional<at::Tensor> &blockTable, int64_t isPackedGQA,
    c10::string_view qInputLayout, c10::string_view kvInputLayout, double scaleValue, int64_t maskType,
    int64_t quantType, double dstTypeMax, int64_t softmaxPrecision, int64_t windowSizeLeft, int64_t windowSizeRight,
    bool returnSoftmaxLse, c10::optional<at::ScalarType> attentionOutDtype)
{
    // pybind 按值收下 std::vector，再在本作用域构建 IntArrayRef，避免悬垂指针。
    at::IntArrayRef blockShapeRef(blockShape);

    // Resolve output dtype: an explicit value passes through unchanged.
    // A missing value (None) only falls back to the input dtype for
    // non-FP8; FP8 path requires it explicitly (same idea as BSA quant_mode).
    at::ScalarType outDtype;
    if (attentionOutDtype.has_value()) {
        outDtype = *attentionOutDtype;
    } else if (query.scalar_type() != at::kFloat8_e4m3fn) {
        outDtype = query.scalar_type();
    } else {
        TORCH_CHECK(false, "attention_out_dtype must be specified when query dtype is float8_e4m3fn");
    }

    at::Tensor attentionOut = at::empty(query.sizes(), query.options().dtype(outDtype));

    at::Tensor softmaxLse;
    auto optsF32 = query.options().dtype(at::kFloat);
    if (returnSoftmaxLse) {
        if (qInputLayout == "TND") {
            softmaxLse = at::empty({query.size(0), query.size(1), 1}, optsF32);
        } else if (qInputLayout == "BNSD") {
            softmaxLse = at::empty({query.size(0), query.size(1), query.size(2), 1}, optsF32);
        } else {
            // BSND
            softmaxLse = at::empty({query.size(0), query.size(2), query.size(1), 1}, optsF32);
        }
    } else {
        softmaxLse = at::empty({0}, optsF32);
    }

    std::string qInputLayoutStr(qInputLayout);
    std::string kvInputLayoutStr(kvInputLayout);
    char *qInputLayoutPtr = const_cast<char *>(qInputLayoutStr.c_str());
    char *kvInputLayoutPtr = const_cast<char *>(kvInputLayoutStr.c_str());

    const int64_t returnSoftmaxLseFlag = returnSoftmaxLse ? 1 : 0;
    ACLNN_CMD(aclnnGenericBlockSparseAttention, query, key, value, sparseBlockIdx, sparseBlockCount, metadata,
              attenMask, qDequantScale, kDequantScale, vDequantScale, pQuantScale, cuSeqLengthsQ, cuSeqLengthsKv,
              seqUsedQ, seqUsedKv, blockTable, blockShapeRef, isPackedGQA, qInputLayoutPtr, kvInputLayoutPtr,
              scaleValue, maskType, quantType, dstTypeMax, softmaxPrecision, windowSizeLeft, windowSizeRight,
              returnSoftmaxLseFlag, attentionOut, softmaxLse);
    return std::make_tuple(attentionOut, softmaxLse);
}

} // namespace op_api

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("generic_block_sparse_attention", &op_api::GenericBlockSparseAttention, "generic_block_sparse_attention");
}
