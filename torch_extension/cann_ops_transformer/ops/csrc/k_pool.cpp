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
                                           const c10::optional<at::Tensor> &seqLens, int64_t cmpRatio, int64_t coff)
{
    auto hiddenStatesDim = hidden_states.dim();
    at::SmallVector<int64_t, MAX_DIM_SIZE> pooledKeySize;
    at::Tensor pooledKey;
    int64_t cmpS = 0;
    int64_t bSize = 0;

    TORCH_CHECK(wk.defined(), "Check hidden_states != nullptr failed");
    auto wkDim = wk.dim();
    TORCH_CHECK(wkDim == DIM_TWO, "wk dim num[", wkDim, "] should be 2");

    TORCH_CHECK(coff == VALUE_1 || coff == VALUE_2, "coff value[", coff, "] should be 1 or 2");

    if (hiddenStatesDim == DIM_THREE) {
        cmpS = (hidden_states.size(1) + cmpRatio - 1) / cmpRatio;
        pooledKeySize = {hidden_states.size(0), cmpS, wk.size(0) / coff};
    } else {
        TORCH_CHECK(seqLens.has_value(), "Check seq_lens != nullptr failed");
        bSize = seqLens->size(0) - 1;
        cmpS = std::min(hidden_states.size(0), hidden_states.size(0) / cmpRatio + bSize);
        pooledKeySize = {cmpS, wk.size(0) / coff};
    }

    pooledKey = at::empty(pooledKeySize, hidden_states.options().dtype(hidden_states.dtype()));
    return pooledKey;
}

at::Tensor KPool(const at::Tensor &hidden_states, const at::Tensor &wk, const at::Tensor &gate_weight, at::Tensor &stateCache,
                      const at::Tensor &ape, int64_t cmpRatio, const c10::optional<at::Tensor> &cacheBlockTable,
                      const c10::optional<at::Tensor> &seqLens, const c10::optional<at::Tensor> &seq_used,
                      const c10::optional<at::Tensor> &startPos, int64_t coff, int64_t cacheMode)
{
    TORCH_CHECK(hidden_states.defined(), "Check hidden_states != nullptr failed");
    auto hiddenStatesDim = hidden_states.dim();
    TORCH_CHECK(hiddenStatesDim == DIM_TWO || hiddenStatesDim == DIM_THREE, "hidden_states dim num[", hiddenStatesDim, "] should be 2 or 3");

    TORCH_CHECK(cmpRatio > VALUE_0, "cmp_ratio should be greater than 0");

    at::Tensor pooledKey = ConstructKPoolOutputTensor(hidden_states, wk, seqLens, cmpRatio, coff);

    auto stateCacheDim = stateCache.dim();
    TORCH_CHECK(stateCacheDim == DIM_THREE, "state_cache dim num[", stateCacheDim, "] should be 3");

    auto contiguousAxesResult = IsContiguousAxes(stateCache);
    int64_t stateCacheStrideDim0 = stateCache.stride(0);

    ACLNN_CMD(aclnnKPool, hidden_states, wk, gate_weight, stateCache, ape, cacheBlockTable, seqLens, seq_used, startPos, cmpRatio,
              coff, cacheMode, stateCacheStrideDim0, pooledKey);

    return pooledKey;
}

at::Tensor KPoolMeta(const at::Tensor &hidden_states, const at::Tensor &wk, const at::Tensor &gate_weight, at::Tensor &stateCache,
                          const at::Tensor &ape, int64_t cmpRatio, const c10::optional<at::Tensor> &cacheBlockTable,
                          const c10::optional<at::Tensor> &seqLens, const c10::optional<at::Tensor> &seq_used,
                          const c10::optional<at::Tensor> &startPos, int64_t coff, int64_t cacheMode)
{
    TORCH_CHECK(hidden_states.defined(), "Check hidden_states != nullptr failed");
    auto hiddenStatesDim = hidden_states.dim();
    TORCH_CHECK(hiddenStatesDim == DIM_TWO || hiddenStatesDim == DIM_THREE, "hidden_states dim num[", hiddenStatesDim, "] should be 2 or 3");

    TORCH_CHECK(cmpRatio > VALUE_0, "cmp_ratio should be greater than 0");

    return ConstructKPoolOutputTensor(hidden_states, wk, seqLens, cmpRatio, coff);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("k_pool", &KPool, "k_pool");
}
} // namespace op_api
