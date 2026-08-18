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
 * \file kv_quant_flash_attention.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;

inline TensorWrapper KvQuantFlashAttentionMakeWrapper(const at::Tensor &tensor, aclDataType tensorAcltype)
{
    return {tensor, tensorAcltype};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
ConstructKvQuantFlashAttentionOutputs(const at::Tensor &q, const at::Tensor &key,
                                      const std::string &layoutQStr,
                                      const std::string &layoutKvStr, bool returnSoftmaxLse,
                                      int64_t ropeHeadDim)
{
    TORCH_CHECK(layoutQStr == "BSND" || layoutQStr == "TND",
                "The layout of query only support BSND and TND, but got ", layoutQStr);
    for (auto i = 0; i < q.sizes().size(); i++) {
        TORCH_CHECK(q.size(i) > 0, "All values within query's shape should be greater "
                                   "than 0, but shape[",
                    i, "] is ", q.size(i));
    }
    at::SmallVector<int64_t, 4> attenOutSize;
    at::SmallVector<int64_t, 4> softmaxMaxSize;
    at::SmallVector<int64_t, 4> softmaxSumSize;
    if (layoutQStr == "BSND") {
        TORCH_CHECK(q.dim() == DIM_3 + 1,
                    "When the layout of query is BSND, the query dimension must be 4, but got ", q.dim());
        attenOutSize = {q.size(DIM_0), q.size(DIM_1), q.size(DIM_2), q.size(DIM_3) - ropeHeadDim};
    } else {
        TORCH_CHECK(q.dim() == DIM_3,
                    "When the layout of query is TND, the query dimension must be 3, but got ", q.dim());
        attenOutSize = {q.size(DIM_0), q.size(DIM_1), q.size(DIM_2) - ropeHeadDim};
    }
    at::Tensor attenOut = at::empty(attenOutSize, q.options());

    if (returnSoftmaxLse) {
        TORCH_CHECK(key.size(DIM_1) > 0, "key.size(DIM_1) must be greater than 0, but got ", key.size(DIM_1));
        TORCH_CHECK(key.size(DIM_2) > 0, "key.size(DIM_2) must be greater than 0, but got ", key.size(DIM_2));
        if (layoutQStr == "BSND") {
            int64_t dim0 = static_cast<int64_t>(q.size(DIM_0));
            int64_t dim1 = static_cast<int64_t>(key.size(DIM_2));
            int64_t dim2 = static_cast<int64_t>(q.size(DIM_1));
            int64_t dim3 = static_cast<int64_t>(q.size(DIM_2)) / static_cast<int64_t>(key.size(DIM_2));
            softmaxMaxSize = {dim0, dim1, dim2, dim3};
            softmaxSumSize = {dim0, dim1, dim2, dim3};
        } else {
            if (layoutKvStr == "PA_BSND") {
                int64_t dim0 = static_cast<int64_t>(key.size(DIM_2));
                int64_t dim1 = static_cast<int64_t>(q.size(DIM_0));
                int64_t dim2 = static_cast<int64_t>(q.size(DIM_1)) / static_cast<int64_t>(key.size(DIM_2));
                softmaxMaxSize = {dim0, dim1, dim2};
                softmaxSumSize = {dim0, dim1, dim2};
            } else {
                int64_t dim0 = static_cast<int64_t>(key.size(DIM_1));
                int64_t dim1 = static_cast<int64_t>(q.size(DIM_0));
                int64_t dim2 = static_cast<int64_t>(q.size(DIM_1)) / static_cast<int64_t>(key.size(DIM_1));
                softmaxMaxSize = {dim0, dim1, dim2};
                softmaxSumSize = {dim0, dim1, dim2};
            }
        }
    } else {
        softmaxMaxSize = {};
        softmaxSumSize = {};
    }
    at::Tensor softmaxMax = at::empty(softmaxMaxSize, q.options().dtype(torch::kFloat32));
    at::Tensor softmaxSum = at::empty(softmaxSumSize, q.options().dtype(torch::kFloat32));
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(attenOut, softmaxMax, softmaxSum);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
KvQuantFlashAttention(
    const at::Tensor &query, const at::Tensor &key, const c10::optional<at::Tensor> &valueOpt,
    const at::Tensor &sparseIndices, const c10::optional<at::Tensor> &keyDequantScaleOpt,
    const c10::optional<at::Tensor> &valueDequantScaleOpt, const c10::optional<at::Tensor> &blockTableOpt,
    const c10::optional<at::Tensor> &actualSeqLengthsQueryOpt,
    const c10::optional<at::Tensor> &actualSeqLengthsKvOpt, const c10::optional<at::Tensor> &sinksOpt,
    double scaleValue, int64_t keyQuantMode, int64_t valueQuantMode, int64_t sparseBlockSize,
    c10::string_view layoutQuery, c10::string_view layoutKv, int64_t sparseMode, int64_t preTokens,
    int64_t nextTokens, int64_t attentionMode, int64_t quantScaleRepoMode, int64_t tileSize,
    int64_t ropeHeadDim, bool returnSoftmaxLse, c10::optional<int64_t> keyDtypeOpt,
    c10::optional<int64_t> valueDtypeOpt)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.")

    std::string layoutQStr = std::string(layoutQuery);
    std::string layoutKvStr = std::string(layoutKv);
    char *layoutQPtr = const_cast<char *>(layoutQStr.c_str());
    char *layoutKvPtr = const_cast<char *>(layoutKvStr.c_str());

    const at::Tensor &keyTensor = key;
    at::Tensor valueTensor;
    if (valueOpt.has_value()) {
        valueTensor = valueOpt.value();
    } else {
        valueTensor = keyTensor;
    }

    auto outputs = op_api::ConstructKvQuantFlashAttentionOutputs(
        query, keyTensor, layoutQStr, layoutKvStr, returnSoftmaxLse, ropeHeadDim);
    at::Tensor attenOut = std::get<0>(outputs);
    at::Tensor softmaxMax = std::get<1>(outputs);
    at::Tensor softmaxSum = std::get<2>(outputs);

    if (keyDtypeOpt.has_value() && valueDtypeOpt.has_value()) {
        aclDataType keyAclType = GetAclDataType(keyDtypeOpt.value());
        aclDataType valueAclType = GetAclDataType(valueDtypeOpt.value());
        TensorWrapper keyWrapper = KvQuantFlashAttentionMakeWrapper(keyTensor, keyAclType);
        TensorWrapper valueWrapper = KvQuantFlashAttentionMakeWrapper(valueTensor, valueAclType);
        ACLNN_CMD(aclnnInnerKvQuantSparseFlashAttentionV2, query, keyWrapper, valueWrapper, sparseIndices,
                  keyDequantScaleOpt, valueDequantScaleOpt, blockTableOpt, actualSeqLengthsQueryOpt,
                  actualSeqLengthsKvOpt, sinksOpt, scaleValue, keyQuantMode, valueQuantMode, sparseBlockSize,
                  layoutQPtr, layoutKvPtr, sparseMode, preTokens, nextTokens, attentionMode, quantScaleRepoMode,
                  tileSize, ropeHeadDim, returnSoftmaxLse, attenOut, softmaxMax, softmaxSum);
    } else {
        ACLNN_CMD(aclnnInnerKvQuantSparseFlashAttentionV2, query, keyTensor, valueTensor, sparseIndices,
                  keyDequantScaleOpt, valueDequantScaleOpt, blockTableOpt, actualSeqLengthsQueryOpt,
                  actualSeqLengthsKvOpt, sinksOpt, scaleValue, keyQuantMode, valueQuantMode, sparseBlockSize,
                  layoutQPtr, layoutKvPtr, sparseMode, preTokens, nextTokens, attentionMode, quantScaleRepoMode,
                  tileSize, ropeHeadDim, returnSoftmaxLse, attenOut, softmaxMax, softmaxSum);
    }
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(attenOut, softmaxMax, softmaxSum);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("kv_quant_flash_attention", &KvQuantFlashAttention,
          "kv_quant_flash_attention");
}
} // namespace op_api
