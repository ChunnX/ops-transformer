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
 * \brief torch 绑定: 封装 aclnnChunkGatedDeltaRule, 返回 (out, final_state)。
 *        initial_state 与 final_state 同 dtype(bf16 或 fp32), 由 host tilingKey 分流。
 */
#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
std::tuple<at::Tensor, at::Tensor> ChunkGatedDeltaRule(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &beta,
    const at::Tensor &initialState, const at::Tensor &actualSeqLengths,
    const c10::optional<at::Tensor> &g, double scale)
{
    // dtype 校验(与 IR 一致): q/k/v/beta bf16; seq_lengths int32; state bf16 或 fp32; g 可选 fp32
    TORCH_CHECK(query.scalar_type() == at::kBFloat16, "query must be bfloat16.");
    TORCH_CHECK(key.scalar_type() == at::kBFloat16, "key must be bfloat16.");
    TORCH_CHECK(value.scalar_type() == at::kBFloat16, "value must be bfloat16.");
    TORCH_CHECK(beta.scalar_type() == at::kBFloat16, "beta must be bfloat16.");
    TORCH_CHECK(actualSeqLengths.scalar_type() == at::kInt, "actual_seq_lengths must be int32.");
    TORCH_CHECK(
        initialState.scalar_type() == at::kBFloat16 || initialState.scalar_type() == at::kFloat,
        "initial_state must be bfloat16 or float32.");
    if (g.has_value()) {
        TORCH_CHECK(g->scalar_type() == at::kFloat, "g must be float32.");
    }

    // 预分配输出: out bf16 (T, Nv, Dv); finalState 同 initialState dtype (B, Nv, Dv, Dk)
    at::Tensor out = at::empty(value.sizes(), value.options().dtype(at::kBFloat16));
    at::Tensor finalState = at::empty(initialState.sizes(), initialState.options());

    // ACLNN_CMD 入参顺序 = GetWorkspaceSize 去掉末尾 workspaceSize/executor
    float scaleValue = static_cast<float>(scale);
    ACLNN_CMD(aclnnChunkGatedDeltaRule, query, key, value, beta, initialState, actualSeqLengths, g,
              scaleValue, out, finalState);
    return std::tuple<at::Tensor, at::Tensor>(out, finalState);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("chunk_gated_delta_rule", &ChunkGatedDeltaRule, "chunk_gated_delta_rule");
}
} // namespace op_api
