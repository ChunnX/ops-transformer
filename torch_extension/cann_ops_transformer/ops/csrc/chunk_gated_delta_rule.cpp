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
 * \file chunk_gated_delta_rule.cpp
 * \brief Torch bridge for aclnnChunkGatedDeltaRule.
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
const int64_t MAX_DIM_SIZE = 8;

std::tuple<at::Tensor, at::Tensor> ChunkGatedDeltaRule(const at::Tensor &query, const at::Tensor &key,
                                                       const at::Tensor &value, const at::Tensor &beta,
                                                       const at::Tensor &initialState,
                                                       const at::Tensor &actualSeqLengths,
                                                       const c10::optional<at::Tensor> &gOptional, double scaleValue)
{
    int64_t tSize = value.size(0);
    int64_t nvSize = value.size(1);
    int64_t dvSize = value.size(2);
    at::SmallVector<int64_t, MAX_DIM_SIZE> outSize = {tSize, nvSize, dvSize};

    at::SmallVector<int64_t, MAX_DIM_SIZE> finalStateSize;
    for (int64_t i = 0; i < initialState.sizes().size(); i++) {
        finalStateSize.push_back(initialState.size(i));
    }

    at::Tensor out;
    at::Tensor finalState;
    {
        auto localDevice = c10::Device(value.device());
        const c10::OptionalDeviceGuard deviceGuard(localDevice);
        out = at::empty(outSize, value.options().dtype(value.dtype()));
        finalState = at::empty(finalStateSize, initialState.options().dtype(initialState.dtype()));
    }

    // aclnnChunkGatedDeltaRule 的 scaleValue 形参为 float，此处显式转换以匹配接口类型
    float scale = static_cast<float>(scaleValue);
    ACLNN_CMD(aclnnChunkGatedDeltaRule, query, key, value, beta, initialState, actualSeqLengths, gOptional, scale, out,
              finalState);

    return std::tuple<at::Tensor, at::Tensor>(out, finalState);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("chunk_gated_delta_rule", &ChunkGatedDeltaRule, "chunk_gated_delta_rule");
}
} // namespace op_api
