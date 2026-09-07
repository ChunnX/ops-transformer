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
 * \file npu_turbo_quant_sparse_flash_attention.cpp
 * \brief TurboQuantSparseFlashAttention operator implementation for PyTorch NPU extension
 */

#include <torch/library.h>
#include <limits>
#include "ops_common.h"

namespace custom {
using namespace at_npu::native;
const int64_t DIM_0 = 0;
const int64_t DIM_1 = 1;
const int64_t DIM_2 = 2;
const int64_t KEY_DIM = 4;
const int64_t QUERY_DIM = 3;
const int64_t KV_SLOT_BYTES = 386;
const int64_t MAX_BLOCK_SIZE = 1024;
const int64_t BLOCK_ALIGNMENT = 16;
const int64_t MAX_QUERY_HEADS = 128;
const int64_t QUERY_HEAD_DIM = 576;

// query is TND: [T, N, D] where D = nope(512) + rope(64).
// attention_out drops the rope tail: [T, N, D - rope_head_dim].
static at::Tensor ConstructAttentionOut(const at::Tensor &query, int64_t ropeHeadDim)
{
    auto queryShape = query.sizes();
    return at::empty({queryShape[DIM_0], queryShape[DIM_1], queryShape[DIM_2] - ropeHeadDim}, query.options());
}

// softmax_max/softmax_sum mirror kv_quant's TND shape so a DCP decode merge can consume them.
// Empty {0} when the caller does not ask for them.
static at::SmallVector<int64_t, 8> ConstructSoftmaxSize(const at::Tensor &query, const at::Tensor &key,
                                                        const std::string &layoutKv, bool returnSoftmaxLse)
{
    at::SmallVector<int64_t, 8> softmaxSize;
    if (returnSoftmaxLse) {
        const int64_t kvHeadDim = key.size(layoutKv == "PA_BNBD" ? DIM_1 : DIM_2);
        softmaxSize = {kvHeadDim, query.size(DIM_0), query.size(DIM_1) / kvHeadDim};
    } else {
        softmaxSize = {0};
    }
    return softmaxSize;
}

// Runtime and Meta implementations share these checks to keep validation and output inference aligned.
static void CheckInputsAndLayout(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                                 const std::string &layoutQuery, const std::string &layoutKv, int64_t ropeHeadDim)
{
    TORCH_CHECK(query.defined(), "Check query != nullptr failed");
    TORCH_CHECK(key.defined(), "Check key != nullptr failed");
    TORCH_CHECK(layoutQuery == "TND", "layout_query[", layoutQuery, "] is not supported, only TND");
    TORCH_CHECK(layoutKv == "PA_BSND" || layoutKv == "PA_BNBD", "layout_kv[", layoutKv,
                "] is not supported, only PA_BSND and PA_BNBD");
    TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16,
                "query dtype is not supported, only float16 and bfloat16");
    TORCH_CHECK(query.dim() == QUERY_DIM, "query dim num[", query.dim(), "] should be 3 (TND)");
    const int64_t heads = query.size(DIM_1);
    TORCH_CHECK(heads > 0 && heads <= MAX_QUERY_HEADS && (heads & (heads - 1)) == 0,
                "query head count must be a power of two in [1, 128]");
    TORCH_CHECK(query.size(DIM_2) == QUERY_HEAD_DIM, "query head dimension must be 576");
    TORCH_CHECK(key.dim() == KEY_DIM, "key dim num[", key.dim(), "] should be 4 (paged KV)");
    const int64_t kvHeadAxis = layoutKv == "PA_BNBD" ? DIM_1 : DIM_2;
    const int64_t blockAxis = layoutKv == "PA_BNBD" ? DIM_2 : DIM_1;
    TORCH_CHECK(key.size(kvHeadAxis) == 1, "key must contain exactly one KV head");
    TORCH_CHECK(key.size(DIM_0) > 0 && key.size(blockAxis) > 0 && key.size(blockAxis) <= MAX_BLOCK_SIZE &&
                    key.size(blockAxis) % BLOCK_ALIGNMENT == 0,
                "KV block count must be positive and block size must be 16-aligned in [16, 1024]");
    TORCH_CHECK(key.size(KEY_DIM - 1) == KV_SLOT_BYTES, "KV slot width must be 386");
    TORCH_CHECK(value.defined() && key.sizes() == value.sizes(), "key and value shapes must match");
    TORCH_CHECK(key.scalar_type() == at::kChar && value.scalar_type() == at::kChar,
                "key and value must have int8 dtype");
    for (const auto *cache : {&key, &value}) {
        int64_t pageBytes = 1;
        for (int64_t axis = KEY_DIM - 1; axis > 0; --axis) {
            TORCH_CHECK(cache->size(axis) == 1 || cache->stride(axis) == pageBytes,
                        "KV cache only supports non-contiguous tensor on axis 0; each page must be contiguous");
            pageBytes *= cache->size(axis);
        }
        const int64_t pages = cache->size(0);
        const int64_t stride0 = cache->stride(0);
        TORCH_CHECK(stride0 >= 0 && (pages <= 1 || stride0 >= pageBytes), "KV cache pages must not overlap");
        TORCH_CHECK(pages <= 1 || stride0 <= (std::numeric_limits<int64_t>::max() - pageBytes) / (pages - 1),
                    "KV cache address span exceeds INT64_MAX");
    }
    TORCH_CHECK(ropeHeadDim >= 0 && ropeHeadDim < query.size(DIM_2), "rope_head_dim[", ropeHeadDim,
                "] should be in [0, query last dim)");
}

// ACLNN may short-circuit empty outputs before Host tiling; validate their metadata here too.
static void CheckSequenceInputs(const at::Tensor &query, const at::Tensor &sparseIndices, const at::Tensor &blockTable,
                                const at::Tensor &queryLengths, const at::Tensor &kvLengths,
                                const c10::optional<at::Tensor> &keyScale, const c10::optional<at::Tensor> &valueScale)
{
    TORCH_CHECK(sparseIndices.dim() == 3 && sparseIndices.size(0) == query.size(0) && sparseIndices.size(1) == 1,
                "sparse_indices must have shape [T, 1, K]");
    TORCH_CHECK(queryLengths.dim() == 1 && queryLengths.size(0) > 0 && kvLengths.dim() == 1 &&
                    kvLengths.size(0) == queryLengths.size(0),
                "sequence lengths must have matching nonempty batch dimensions");
    TORCH_CHECK(blockTable.dim() == 2 && blockTable.size(0) == queryLengths.size(0) && blockTable.size(1) > 0,
                "block_table must match the sequence batch and contain at least one block column");
    TORCH_CHECK(sparseIndices.scalar_type() == at::kInt && blockTable.scalar_type() == at::kInt &&
                    queryLengths.scalar_type() == at::kInt && kvLengths.scalar_type() == at::kInt,
                "sparse indices, block table and sequence lengths must have int32 dtype");
    TORCH_CHECK(!keyScale.has_value() || keyScale->scalar_type() == at::kFloat,
                "key_dequant_scale must have float32 dtype");
    TORCH_CHECK(!valueScale.has_value() || valueScale->scalar_type() == at::kFloat,
                "value_dequant_scale must have float32 dtype");
}

static void CheckAttributes(const at::Tensor &key, const std::string &layoutKv, int64_t keyQuantMode,
                            int64_t valueQuantMode, int64_t sparseBlockSize, int64_t sparseMode, int64_t preTokens,
                            int64_t nextTokens, int64_t attentionMode, int64_t quantScaleRepoMode, int64_t tileSize,
                            int64_t ropeHeadDim)
{
    TORCH_CHECK(keyQuantMode == 3 && valueQuantMode == 3, "key/value quant modes must both be 3");
    TORCH_CHECK(sparseBlockSize > 0 && sparseBlockSize <= 16 && (sparseBlockSize & (sparseBlockSize - 1)) == 0,
                "sparse_block_size must be a power of two in [1, 16]");
    TORCH_CHECK(key.size(layoutKv == "PA_BNBD" ? DIM_2 : DIM_1) % sparseBlockSize == 0,
                "block size must be divisible by sparse_block_size");
    TORCH_CHECK(sparseMode == 0 || sparseMode == 3, "sparse_mode must be 0 or 3");
    TORCH_CHECK(preTokens == INT64_MAX && nextTokens == INT64_MAX, "pre_tokens and next_tokens must be INT64_MAX");
    TORCH_CHECK(attentionMode == 2 && quantScaleRepoMode == 1 && tileSize == 128 && ropeHeadDim == 64,
                "attention_mode, quant_scale_repo_mode, tile_size and rope_head_dim must be 2, 1, 128 and 64");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_turbo_quant_sparse_flash_attention_npu(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &sparseIndices,
    const c10::optional<at::Tensor> &keyDequantScale, const c10::optional<at::Tensor> &valueDequantScale,
    const at::Tensor &blockTable, const at::Tensor &actualSeqLengthsQuery, const at::Tensor &actualSeqLengthsKv,
    double scaleValue, int64_t keyQuantMode, int64_t valueQuantMode, int64_t sparseBlockSize,
    c10::string_view layoutQuery, c10::string_view layoutKv, int64_t sparseMode, int64_t preTokens, int64_t nextTokens,
    int64_t attentionMode, int64_t quantScaleRepoMode, int64_t tileSize, int64_t ropeHeadDim, bool returnSoftmaxLse)
{
    std::string layoutQueryStr(layoutQuery);
    std::string layoutKvStr(layoutKv);
    CheckInputsAndLayout(query, key, value, layoutQueryStr, layoutKvStr, ropeHeadDim);
    CheckSequenceInputs(query, sparseIndices, blockTable, actualSeqLengthsQuery, actualSeqLengthsKv, keyDequantScale,
                        valueDequantScale);
    CheckAttributes(key, layoutKvStr, keyQuantMode, valueQuantMode, sparseBlockSize, sparseMode, preTokens, nextTokens,
                    attentionMode, quantScaleRepoMode, tileSize, ropeHeadDim);
    char *layoutQueryPtr = const_cast<char *>(layoutQueryStr.c_str());
    char *layoutKvPtr = const_cast<char *>(layoutKvStr.c_str());

    at::Tensor attentionOut = ConstructAttentionOut(query, ropeHeadDim);
    auto softmaxSize = ConstructSoftmaxSize(query, key, layoutKvStr, returnSoftmaxLse);
    at::Tensor softmaxMax = at::empty(softmaxSize, query.options().dtype(at::kFloat));
    at::Tensor softmaxSum = at::empty(softmaxSize, query.options().dtype(at::kFloat));

    EXEC_NPU_CMD_V1(aclnnTurboQuantSparseFlashAttention, query, key, value, sparseIndices, keyDequantScale,
                    valueDequantScale, blockTable, actualSeqLengthsQuery, actualSeqLengthsKv, scaleValue, keyQuantMode,
                    valueQuantMode, sparseBlockSize, layoutQueryPtr, layoutKvPtr, sparseMode, preTokens, nextTokens,
                    attentionMode, quantScaleRepoMode, tileSize, ropeHeadDim, returnSoftmaxLse, attentionOut,
                    softmaxMax, softmaxSum);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(attentionOut, softmaxMax, softmaxSum);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_turbo_quant_sparse_flash_attention_meta(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &sparseIndices,
    const c10::optional<at::Tensor> &keyDequantScale, const c10::optional<at::Tensor> &valueDequantScale,
    const at::Tensor &blockTable, const at::Tensor &actualSeqLengthsQuery, const at::Tensor &actualSeqLengthsKv, double,
    int64_t, int64_t, int64_t, c10::string_view layoutQuery, c10::string_view layoutKv, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t ropeHeadDim, bool returnSoftmaxLse)
{
    std::string layoutQueryStr(layoutQuery);
    std::string layoutKvStr(layoutKv);
    CheckInputsAndLayout(query, key, value, layoutQueryStr, layoutKvStr, ropeHeadDim);
    CheckSequenceInputs(query, sparseIndices, blockTable, actualSeqLengthsQuery, actualSeqLengthsKv, keyDequantScale,
                        valueDequantScale);
    at::Tensor attentionOut = ConstructAttentionOut(query, ropeHeadDim);
    auto softmaxSize = ConstructSoftmaxSize(query, key, layoutKvStr, returnSoftmaxLse);
    at::Tensor softmaxMax = at::empty(softmaxSize, query.options().dtype(at::kFloat));
    at::Tensor softmaxSum = at::empty(softmaxSize, query.options().dtype(at::kFloat));
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(attentionOut, softmaxMax, softmaxSum);
}

} // namespace custom

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m)
{
    m.impl("npu_turbo_quant_sparse_flash_attention", &custom::npu_turbo_quant_sparse_flash_attention_npu);
}

TORCH_LIBRARY_IMPL(custom, Meta, m)
{
    m.impl("npu_turbo_quant_sparse_flash_attention", &custom::npu_turbo_quant_sparse_flash_attention_meta);
}
