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
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

const int SIZE = 8;
const int DIM_0 = 0;
const int DIM_1 = 1;
const int DIM_2 = 2;
const int DIM_3 = 3;
const int DIM_4 = 4;

inline TensorWrapper MakeWrapper(const at::Tensor &tensor, aclDataType tensorAclType)
{
    return {tensor, tensorAclType};
}

void CheckQueryShape(const at::Tensor &q, const std::string &layoutQStr)
{
    TORCH_CHECK(layoutQStr == "BSND" || layoutQStr == "TND", "The layout of query only support BSND and TND, but got ",
                layoutQStr);
    TORCH_CHECK(q.numel() > 0, "Tensor query is empty.");
    for (int64_t i = 0; i < q.dim(); i++) {
        TORCH_CHECK(q.size(i) > 0, "All values within query's shape should be greater than 0, but shape[", i, "] is ",
                    q.size(i));
    }
    if (layoutQStr == "BSND") {
        TORCH_CHECK(q.dim() == DIM_4, "When the layout of query is BSND, the query dimension must be 4, but got ",
                    q.dim());
    } else {
        TORCH_CHECK(q.dim() == DIM_3, "When the layout of query is TND, the query dimension must be 3, but got ",
                    q.dim());
    }
}

int64_t GetKvHeadNum(const at::Tensor &key, const std::string &layoutQStr, const std::string &layoutKvStr)
{
    if (layoutQStr == "BSND") {
        return key.size(DIM_2);
    }
    if (layoutKvStr == "PA_BSND") {
        return key.size(DIM_2);
    }
    return key.size(DIM_1);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> MakeKvQuantSparseFlashAttentionOutputs(
    const at::Tensor &q, const at::Tensor &key, const std::string &layoutQStr, const std::string &layoutKvStr,
    int64_t ropeHeadDim, bool returnSoftmaxLse)
{
    CheckQueryShape(q, layoutQStr);

    at::SmallVector<int64_t, SIZE> attenOutSize;
    if (layoutQStr == "BSND") {
        attenOutSize = {q.size(DIM_0), q.size(DIM_1), q.size(DIM_2), q.size(DIM_3) - ropeHeadDim};
    } else {
        attenOutSize = {q.size(DIM_0), q.size(DIM_1), q.size(DIM_2) - ropeHeadDim};
    }
    at::Tensor attenOut = at::empty(attenOutSize, q.options().dtype(q.dtype()));

    if (!returnSoftmaxLse) {
        at::Tensor softmaxMax = at::empty({0}, q.options().dtype(torch::kFloat32));
        at::Tensor softmaxSum = at::empty({0}, q.options().dtype(torch::kFloat32));
        return {attenOut, softmaxMax, softmaxSum};
    }

    int64_t kvHeadNum = GetKvHeadNum(key, layoutQStr, layoutKvStr);
    TORCH_CHECK(kvHeadNum > 0, "head num of key must be greater than 0, but got ", kvHeadNum);

    at::SmallVector<int64_t, SIZE> lseSize;
    if (layoutQStr == "BSND") {
        lseSize = {q.size(DIM_0), kvHeadNum, q.size(DIM_1), q.size(DIM_2) / kvHeadNum};
    } else {
        lseSize = {kvHeadNum, q.size(DIM_0), q.size(DIM_1) / kvHeadNum};
    }
    at::Tensor softmaxMax = at::empty(lseSize, q.options().dtype(torch::kFloat32));
    at::Tensor softmaxSum = at::empty(lseSize, q.options().dtype(torch::kFloat32));
    return {attenOut, softmaxMax, softmaxSum};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> KvQuantSparseFlashAttention(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &sparseIndices,
    double scaleValue, int64_t keyQuantMode, int64_t valueQuantMode, const c10::optional<at::Tensor> &keyDequantScale,
    const c10::optional<at::Tensor> &valueDequantScale, const c10::optional<at::Tensor> &blockTable,
    const c10::optional<at::Tensor> &actualSeqLengthsQuery, const c10::optional<at::Tensor> &actualSeqLengthsKv,
    const c10::optional<at::Tensor> &sinks, int64_t sparseBlockSize, c10::string_view layoutQuery,
    c10::string_view layoutKv, int64_t sparseMode, int64_t preTokens, int64_t nextTokens, int64_t attentionMode,
    int64_t quantScaleRepoMode, int64_t tileSize, int64_t ropeHeadDim, bool returnSoftmaxLse,
    c10::optional<int64_t> keyDtype, c10::optional<int64_t> valueDtype)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.")
    TORCH_CHECK(key.numel() > 0, "Tensor key is empty.")
    TORCH_CHECK(value.numel() > 0, "Tensor value is empty.")
    TORCH_CHECK(sparseIndices.numel() > 0, "Tensor sparse_indices is empty.")

    std::string layoutQStr = std::string(layoutQuery);
    std::string layoutKvStr = std::string(layoutKv);
    // char *layoutQPtr = const_cast<char *>(layoutQStr.c_str());
    // char *layoutKvPtr = const_cast<char *>(layoutKvStr.c_str());
    const char *layoutQPtr = layoutQStr.c_str();
    const char *layoutKvPtr = layoutKvStr.c_str();

    auto outputs =
        MakeKvQuantSparseFlashAttentionOutputs(query, key, layoutQStr, layoutKvStr, ropeHeadDim, returnSoftmaxLse);
    at::Tensor attenOut = std::get<0>(outputs);
    at::Tensor softmaxMax = std::get<1>(outputs);
    at::Tensor softmaxSum = std::get<2>(outputs);

    bool isHifloat8Kv = keyDtype.has_value() && GetAclDataType(keyDtype.value()) == ACL_HIFLOAT8;
    if (isHifloat8Kv) {
        TensorWrapper keyWrapper = MakeWrapper(key, GetAclDataType(keyDtype.value()));
        TensorWrapper valueWrapper = MakeWrapper(value, GetAclDataType(valueDtype.value()));
        ACLNN_CMD(aclnnInnerKvQuantSparseFlashAttentionV2, query, keyWrapper, valueWrapper, sparseIndices,
                  keyDequantScale, valueDequantScale, blockTable, actualSeqLengthsQuery, actualSeqLengthsKv, sinks,
                  scaleValue, keyQuantMode, valueQuantMode, sparseBlockSize, layoutQPtr, layoutKvPtr, sparseMode,
                  preTokens, nextTokens, attentionMode, quantScaleRepoMode, tileSize, ropeHeadDim, returnSoftmaxLse,
                  attenOut, softmaxMax, softmaxSum);
    } else {
        ACLNN_CMD(aclnnInnerKvQuantSparseFlashAttentionV2, query, key, value, sparseIndices, keyDequantScale,
                  valueDequantScale, blockTable, actualSeqLengthsQuery, actualSeqLengthsKv, sinks, scaleValue,
                  keyQuantMode, valueQuantMode, sparseBlockSize, layoutQPtr, layoutKvPtr, sparseMode, preTokens,
                  nextTokens, attentionMode, quantScaleRepoMode, tileSize, ropeHeadDim, returnSoftmaxLse, attenOut,
                  softmaxMax, softmaxSum);
    }
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(attenOut, softmaxMax, softmaxSum);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("kv_quant_sparse_flash_attention", &KvQuantSparseFlashAttention, "kv_quant_sparse_flash_attention");
}
} // namespace op_api
