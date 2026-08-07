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
 * \file k_pool.cpp
 * \brief KPool operator implementation for PyTorch NPU extension
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
const int64_t DIM_ONE = 1;
const int64_t DIM_TWO = 2;
const int64_t DIM_THREE = 3;
const int64_t MAX_DIM_SIZE = 8;
const int64_t VALUE_0 = 0;
const int64_t VALUE_1 = 1;
const int64_t VALUE_2 = 2;

std::vector<bool> IsContiguousAxes(const at::Tensor &tensor)
{
    auto sizes = tensor.sizes();
    auto strides = tensor.strides();
    int64_t ndim = sizes.size();
    if (ndim == 0) {
        return {};
    }
    std::vector<bool> result(ndim, false);

    std::vector<int64_t> contiguousStride(ndim, 1);
    for (int64_t i = ndim - 2; i >= 0; i--) {
        contiguousStride[i] = contiguousStride[i + 1] * sizes[i + 1];
    }

    for (int64_t i = 0; i < ndim; i++) {
        result[i] = (strides[i] == contiguousStride[i]);
    }
    return result;
}

at::Tensor ConstructKPoolOutputTensor(const at::Tensor &hidden_states, const at::Tensor &wk,
                                      const at::Tensor &stateCache, const at::Tensor &cacheBlockTable, int64_t cmpRatio)
{
    auto hiddenStatesDim = hidden_states.dim();
    at::SmallVector<int64_t, MAX_DIM_SIZE> pooledKeySize;
    at::Tensor pooledKey;

    TORCH_CHECK(wk.defined(), "Check hidden_states != nullptr failed");
    auto wkDim = wk.dim();
    TORCH_CHECK(wkDim == DIM_TWO, "wk dim num[", wkDim, "] should be 2");

    TORCH_CHECK(hiddenStatesDim == DIM_TWO || hiddenStatesDim == DIM_THREE,
                "hidden_states dim num[", hiddenStatesDim, "] should be 2 or 3");
    TORCH_CHECK(stateCache.dim() == DIM_THREE, "state_cache dim num[", stateCache.dim(), "] should be 3");
    TORCH_CHECK(cacheBlockTable.dim() == DIM_TWO,
                "cache_block_table dim num[", cacheBlockTable.dim(), "] should be 2");
    int64_t pcap = (cacheBlockTable.size(1) * stateCache.size(1) + cmpRatio - 1) / cmpRatio;
    pooledKeySize = {cacheBlockTable.size(0), pcap, wk.size(0)};

    pooledKey = at::zeros(pooledKeySize, hidden_states.options().dtype(hidden_states.dtype()));
    return pooledKey;
}

at::Tensor KPool(const at::Tensor &hidden_states, const at::Tensor &wk, const at::Tensor &gate_weight,
                 const at::Tensor &ape, at::Tensor &stateCache, const at::Tensor &cacheBlockTable,
                 const at::Tensor &startPos, const c10::optional<at::Tensor> &normWeight,
                 const c10::optional<at::Tensor> &normBias, const c10::optional<at::Tensor> &cos,
                 const c10::optional<at::Tensor> &sin, const c10::optional<at::Tensor> &seqLens,
                 int64_t cmpRatio, double normEps, int64_t rotaryMode)
{
    TORCH_CHECK(hidden_states.defined(), "Check hidden_states != nullptr failed");
    auto hiddenStatesDim = hidden_states.dim();
    TORCH_CHECK(hiddenStatesDim == DIM_TWO || hiddenStatesDim == DIM_THREE, "hidden_states dim num[", hiddenStatesDim, "] should be 2 or 3");

    TORCH_CHECK(cmpRatio > VALUE_0, "cmp_ratio should be greater than 0");

    TORCH_CHECK(normWeight.has_value() == normBias.has_value(),
                "norm_weight and norm_bias must be provided as a pair");
    if (normWeight.has_value()) {
        TORCH_CHECK(normWeight->scalar_type() == at::kFloat && normBias->scalar_type() == at::kFloat,
                    "norm_weight and norm_bias must be FP32");
        TORCH_CHECK(normWeight->dim() == DIM_ONE && normBias->dim() == DIM_ONE,
                    "norm_weight and norm_bias must be rank-1");
        TORCH_CHECK(normWeight->size(0) == wk.size(0) && normBias->size(0) == wk.size(0),
                    "norm_weight and norm_bias size must match wk.size(0)");
    }
    TORCH_CHECK(normEps > 0.0, "norm_eps should be greater than 0");
    TORCH_CHECK(cos.has_value() == sin.has_value(), "cos and sin must be provided as a pair");
    TORCH_CHECK(!cos.has_value(), "KPool RoPE is not implemented in this stage");
    TORCH_CHECK(hiddenStatesDim != DIM_TWO || seqLens.has_value(), "seq_lens is required for TH layout");
    TORCH_CHECK(hiddenStatesDim != DIM_THREE || !seqLens.has_value(), "seq_lens must be absent for BSH layout");
    at::Tensor pooledKey = ConstructKPoolOutputTensor(hidden_states, wk, stateCache, cacheBlockTable, cmpRatio);

    auto stateCacheDim = stateCache.dim();
    TORCH_CHECK(stateCacheDim == DIM_THREE, "state_cache dim num[", stateCacheDim, "] should be 3");

    auto contiguousAxesResult = IsContiguousAxes(stateCache);
    int64_t stateCacheStrideDim0 = stateCache.stride(0);

    c10::optional<at::Tensor> seqOffsets = c10::nullopt;
    if (seqLens.has_value()) {
        TORCH_CHECK(seqLens->dim() == DIM_ONE, "seq_lens dim num[", seqLens->dim(), "] should be 1");
        auto cumulative = at::cumsum(*seqLens, VALUE_0, seqLens->scalar_type());
        auto prefix = at::zeros({VALUE_1}, seqLens->options());
        seqOffsets = at::cat({prefix, cumulative}, VALUE_0);
    }
    ACLNN_CMD(aclnnKPool, hidden_states, wk, gate_weight, ape, stateCache, cacheBlockTable, startPos,
              normWeight, normBias, cos, sin, seqOffsets, cmpRatio, normEps, rotaryMode,
              stateCacheStrideDim0, pooledKey);

    return pooledKey;
}

at::Tensor KPoolMeta(const at::Tensor &hidden_states, const at::Tensor &wk, const at::Tensor &gate_weight,
                     const at::Tensor &ape, at::Tensor &stateCache, const at::Tensor &cacheBlockTable,
                     const at::Tensor &startPos, const c10::optional<at::Tensor> &normWeight,
                     const c10::optional<at::Tensor> &normBias, const c10::optional<at::Tensor> &cos,
                     const c10::optional<at::Tensor> &sin, const c10::optional<at::Tensor> &seqLens,
                     int64_t cmpRatio, double normEps, int64_t rotaryMode)
{
    TORCH_CHECK(hidden_states.defined(), "Check hidden_states != nullptr failed");
    auto hiddenStatesDim = hidden_states.dim();
    TORCH_CHECK(hiddenStatesDim == DIM_TWO || hiddenStatesDim == DIM_THREE, "hidden_states dim num[", hiddenStatesDim, "] should be 2 or 3");

    TORCH_CHECK(cmpRatio > VALUE_0, "cmp_ratio should be greater than 0");

    TORCH_CHECK(normWeight.has_value() == normBias.has_value(),
                "norm_weight and norm_bias must be provided as a pair");
    if (normWeight.has_value()) {
        TORCH_CHECK(normWeight->scalar_type() == at::kFloat && normBias->scalar_type() == at::kFloat,
                    "norm_weight and norm_bias must be FP32");
        TORCH_CHECK(normWeight->dim() == DIM_ONE && normBias->dim() == DIM_ONE,
                    "norm_weight and norm_bias must be rank-1");
        TORCH_CHECK(normWeight->size(0) == wk.size(0) && normBias->size(0) == wk.size(0),
                    "norm_weight and norm_bias size must match wk.size(0)");
    }
    TORCH_CHECK(normEps > 0.0, "norm_eps should be greater than 0");
    TORCH_CHECK(cos.has_value() == sin.has_value(), "cos and sin must be provided as a pair");
    TORCH_CHECK(!cos.has_value(), "KPool RoPE is not implemented in this stage");
    return ConstructKPoolOutputTensor(hidden_states, wk, stateCache, cacheBlockTable, cmpRatio);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("k_pool", &KPool, "k_pool");
}
} // namespace op_api
