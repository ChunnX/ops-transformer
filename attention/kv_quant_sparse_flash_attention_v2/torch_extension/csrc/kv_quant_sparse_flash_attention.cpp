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
 * \file kv_quant_sparse_flash_attention.cpp
 * \brief C++ wrapper for aclnnKvQuantSparseFlashAttentionV2
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

const int SIZE = 8;
const int DIM_0 = 0;
const int DIM_1 = 1;
const int DIM_2 = 2;
const int DIM_3 = 3;

inline TensorWrapper KvQsfaMakeWrapper(const at::Tensor &tensor, const c10::optional<int64_t> &kvDtype)
{
    aclDataType aclDtype = ACL_DT_UNDEFINED;
    if (kvDtype.has_value()) {
        aclDtype = GetAclDataType(kvDtype.value());
    } else {
        aclDtype = kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(tensor.scalar_type())];
    }
    return {tensor, aclDtype};
}

at::Tensor KvQuantSparseFlashAttention(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &sparseIndices,
    const c10::optional<at::Tensor> &keyDequantScale,
    const c10::optional<at::Tensor> &valueDequantScale,
    const c10::optional<at::Tensor> &blockTable,
    const c10::optional<at::Tensor> &actualSeqLengthsQuery,
    const c10::optional<at::Tensor> &actualSeqLengthsKv,
    double scaleValue, int64_t keyQuantMode, int64_t valueQuantMode,
    int64_t sparseBlockSize, c10::string_view layoutQuery, c10::string_view layoutKv,
    int64_t sparseMode, int64_t preTokens, int64_t nextTokens,
    int64_t attentionMode, int64_t quantScaleRepoMode,
    int64_t tileSize, int64_t ropeHeadDim,
    c10::optional<int64_t> keyDtype, c10::optional<int64_t> valueDtype,
    const c10::optional<at::Tensor> &sinks)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.")
    TORCH_CHECK(key.numel() > 0, "Tensor key is empty.")
    TORCH_CHECK(sparseIndices.numel() > 0, "Tensor sparse_indices is empty.")

    std::string layoutQStr(layoutQuery);
    std::string layoutKvStr(layoutKv);
    char *layoutQPtr = const_cast<char *>(layoutQStr.c_str());
    char *layoutKvPtr = const_cast<char *>(layoutKvStr.c_str());

    auto outSizes = query.sizes().vec();
    outSizes.back() -= ropeHeadDim;
    at::Tensor attentionOut = at::empty(outSizes, query.options().dtype(query.dtype()));

    bool isHifloat8Kv = keyDtype.has_value() && GetAclDataType(keyDtype.value()) == ACL_HIFLOAT8;
    bool hasSinks = sinks.has_value() && sinks->defined();
    if (!hasSinks) {
        if (isHifloat8Kv) {
            TensorWrapper keyWrapper = KvQsfaMakeWrapper(key, keyDtype);
            TensorWrapper valueWrapper = KvQsfaMakeWrapper(value, valueDtype);
            ACLNN_CMD(aclnnKvQuantSparseFlashAttention, query, keyWrapper, valueWrapper,
                      sparseIndices, keyDequantScale, valueDequantScale, blockTable,
                      actualSeqLengthsQuery, actualSeqLengthsKv, scaleValue, keyQuantMode,
                      valueQuantMode, sparseBlockSize, layoutQPtr, layoutKvPtr, sparseMode,
                      preTokens, nextTokens, attentionMode, quantScaleRepoMode, tileSize,
                      ropeHeadDim, attentionOut);
        } else {
            ACLNN_CMD(aclnnKvQuantSparseFlashAttention, query, key, value,
                      sparseIndices, keyDequantScale, valueDequantScale, blockTable,
                      actualSeqLengthsQuery, actualSeqLengthsKv, scaleValue, keyQuantMode,
                      valueQuantMode, sparseBlockSize, layoutQPtr, layoutKvPtr, sparseMode,
                      preTokens, nextTokens, attentionMode, quantScaleRepoMode, tileSize,
                      ropeHeadDim, attentionOut);
        }
    } else {
        if (isHifloat8Kv) {
            TensorWrapper keyWrapper = KvQsfaMakeWrapper(key, keyDtype);
            TensorWrapper valueWrapper = KvQsfaMakeWrapper(value, valueDtype);
            ACLNN_CMD(aclnnKvQuantSparseFlashAttentionV2, query, keyWrapper, valueWrapper,
                      sparseIndices, keyDequantScale, valueDequantScale, blockTable,
                      actualSeqLengthsQuery, actualSeqLengthsKv, sinks, scaleValue,
                      keyQuantMode, valueQuantMode, sparseBlockSize, layoutQPtr, layoutKvPtr,
                      sparseMode, preTokens, nextTokens, attentionMode, quantScaleRepoMode,
                      tileSize, ropeHeadDim, attentionOut);
        } else {
            ACLNN_CMD(aclnnKvQuantSparseFlashAttentionV2, query, key, value,
                      sparseIndices, keyDequantScale, valueDequantScale, blockTable,
                      actualSeqLengthsQuery, actualSeqLengthsKv, sinks, scaleValue,
                      keyQuantMode, valueQuantMode, sparseBlockSize, layoutQPtr, layoutKvPtr,
                      sparseMode, preTokens, nextTokens, attentionMode, quantScaleRepoMode,
                      tileSize, ropeHeadDim, attentionOut);
        }
    }

    return attentionOut;
}

} // namespace op_api

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_kv_quant_sparse_flash_attention", &op_api::KvQuantSparseFlashAttention,
          "kv_quant_sparse_flash_attention");
}
