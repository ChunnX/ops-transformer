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
 * \file sparse_flash_attention.cpp
 * \brief
 */

#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/GetCANNInfo.h>
#include "aclnn_common.h"

namespace op_api {
using namespace at_npu::native;

// npu tensor max size
const int SIZE = 8;
const int DIM_0 = 0;
const int DIM_1 = 1;
const int DIM_2 = 2;
const int DIM_3 = 3;
const int DIM_4 = 4;
const std::string SFA_VALUE_NONE_MIN_CANN_VERSION = "9.2.0";

bool ParseCannVersion(const std::string &version, int64_t (&parsed)[4])
{
    parsed[0] = parsed[1] = parsed[2] = parsed[3] = 0;
    int part = 0;
    size_t begin = 0;
    while (begin < version.size()) {
        // only "a.b.c" / "a.b.RCd" / "a.b.Tg" / "a.b.RCd.alphaf" style tails are meaningful
        const char c = version[begin];
        if (c >= '0' && c <= '9') {
            size_t end = begin;
            while (end < version.size() && version[end] >= '0' && version[end] <= '9') {
                ++end;
            }
            if (part < 4) {
                parsed[part] = std::stoll(version.substr(begin, end - begin));
            }
            ++part;
            begin = end;
        } else {
            ++begin;
        }
    }
    return part > 0;
}

bool IsSupportValueNone()
{
    static const bool is_support = []() -> bool {
        std::string version;
        try {
            version = GetCANNVersion("CANN");
        } catch (const std::exception &) {
            return false;
        }
        int64_t cur[4] = {0};
        int64_t min[4] = {0};
        if (!ParseCannVersion(version, cur) || !ParseCannVersion(SFA_VALUE_NONE_MIN_CANN_VERSION, min)) {
            return false;
        }
        for (int i = 0; i < 4; ++i) {
            if (cur[i] != min[i]) {
                return cur[i] > min[i];
            }
        }
        return true;
    }();
    return is_support;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> ConstructSparseFlashAttentionOutputTensor(
    const at::Tensor &query, const at::Tensor &key, const std::string &queryLayoutStr, const std::string &keyLayoutStr,
    bool returnSoftmaxLse)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.")
    TORCH_CHECK(key.numel() > 0, "Tensor key is empty.")
    TORCH_CHECK(queryLayoutStr == "BSND" || queryLayoutStr == "TND",
                "The layout of query only support BSND and TND, but got ", queryLayoutStr);
    if (queryLayoutStr == "TND") {
        TORCH_CHECK(query.dim() == DIM_3,
                    "When the layout of query is TND, the query dimension must be 3, but got ", query.dim());
    } else {
        TORCH_CHECK(query.dim() == DIM_4,
                    "When the layout of query is BSND, the query dimension must be 4, but got ", query.dim());
    }
    TORCH_CHECK(keyLayoutStr == "BSND" || keyLayoutStr == "TND" || keyLayoutStr == "PA_BSND",
                "The layout of key only support BSND, TND and PA_BSND, but got ", keyLayoutStr);
    if (keyLayoutStr == "TND") {
        TORCH_CHECK(key.dim() == DIM_3,
                    "When the layout of key is TND, the key dimension must be 3, but got ", key.dim());
    } else {
        TORCH_CHECK(key.dim() == DIM_4,
                    "When the layout of key is BSND or PA_BSND, the key dimension must be 4, but got ", key.dim());
    }
    int64_t keyHeadNum = (keyLayoutStr == "TND") ? key.size(DIM_1) : key.size(DIM_2);

    // construct the output tensor
    at::Tensor attentionOut = at::empty(query.sizes(), query.options());
    at::Tensor softmaxMax;
    at::Tensor softmaxSum;
    if (returnSoftmaxLse) {
        at::SmallVector<int64_t, SIZE> lseSize;
        if (queryLayoutStr == "BSND") {
            lseSize = {query.size(DIM_0), keyHeadNum, query.size(DIM_1), query.size(DIM_2) / keyHeadNum};
        } else {
            lseSize = {keyHeadNum, query.size(DIM_0), query.size(DIM_1) / keyHeadNum};
        }
        softmaxMax = at::empty(lseSize, query.options().dtype(at::kFloat));
        softmaxSum = at::empty(lseSize, query.options().dtype(at::kFloat));
    } else {
        softmaxMax = at::empty({0}, query.options().dtype(at::kFloat));
        softmaxSum = at::empty({0}, query.options().dtype(at::kFloat));
    }
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(attentionOut, softmaxMax, softmaxSum);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> SparseFlashAttention(
    const at::Tensor &query, const at::Tensor &key, const c10::optional<at::Tensor> &value, const at::Tensor &sparseIndices,
    double scaleValue, const c10::optional<at::Tensor> &blockTable,
    const c10::optional<at::Tensor> &actualSeqLengthsQuery, const c10::optional<at::Tensor> &actualSeqLengthsKv,
    const c10::optional<at::Tensor> &queryRope, const c10::optional<at::Tensor> &keyRope,
    const c10::optional<at::Tensor> &sinks, int64_t sparseBlockSize, c10::string_view layoutQuery,
    c10::string_view layoutKv, int64_t sparseMode, int64_t preTokens, int64_t nextTokens, int64_t attentionMode,
    bool returnSoftmaxLse)
{
    if (value.has_value()) {
        TORCH_CHECK(value->numel() > 0, "Tensor value is empty.")
    } else {
        TORCH_CHECK(IsSupportValueNone(),
                    "npu_sparse_flash_attention: value=None is only supported with CANN >= 9.2.0.")
    }
    TORCH_CHECK(sparseIndices.numel() > 0, "Tensor sparseIndices is empty.")
    std::string queryLayoutStr = std::string(layoutQuery);
    std::string keyLayoutStr = std::string(layoutKv);

    // construct the output tensor
    std::tuple<at::Tensor, at::Tensor, at::Tensor> sparseFlashAttentionOutput =
        ConstructSparseFlashAttentionOutputTensor(query, key, queryLayoutStr, keyLayoutStr, returnSoftmaxLse);
    at::Tensor attentionOut = std::get<0>(sparseFlashAttentionOutput);
    at::Tensor softmaxMax = std::get<1>(sparseFlashAttentionOutput);
    at::Tensor softmaxSum = std::get<2>(sparseFlashAttentionOutput);
    // convert str
    char *queryLayoutPtr = const_cast<char *>(queryLayoutStr.c_str());
    char *keyLayoutPtr = const_cast<char *>(keyLayoutStr.c_str());

    if (sinks.has_value() && sinks->defined()) {
        ACLNN_CMD(aclnnSparseFlashAttentionV2, query, key, value, sparseIndices, blockTable, actualSeqLengthsQuery,
                  actualSeqLengthsKv, queryRope, keyRope, sinks, scaleValue, sparseBlockSize, queryLayoutPtr,
                  keyLayoutPtr, sparseMode, preTokens, nextTokens, attentionMode, returnSoftmaxLse, attentionOut,
                  softmaxMax, softmaxSum);
    } else {
        ACLNN_CMD(aclnnSparseFlashAttention, query, key, value, sparseIndices, blockTable, actualSeqLengthsQuery,
                  actualSeqLengthsKv, queryRope, keyRope, scaleValue, sparseBlockSize, queryLayoutPtr, keyLayoutPtr,
                  sparseMode, preTokens, nextTokens, attentionMode, returnSoftmaxLse, attentionOut, softmaxMax,
                  softmaxSum);
    }
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(attentionOut, softmaxMax, softmaxSum);
}
// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("sparse_flash_attention", &SparseFlashAttention, "sparse_flash_attention");
}
} // namespace op_api
