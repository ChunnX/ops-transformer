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
 * \brief Torch bridge — chunk gated delta rule forward, delegates to aclnnChunkGatedDeltaRule.
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

std::tuple<at::Tensor, at::Tensor>
ChunkGatedDeltaRule(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                    const at::Tensor &beta, const at::Tensor &initial_state,
                    const at::Tensor &actual_seq_lengths, const c10::optional<at::Tensor> &g, float scale)
{
    TORCH_CHECK(query.scalar_type() == at::kBFloat16, "query dtype should be bfloat16, but got: ", query.scalar_type());
    TORCH_CHECK(key.scalar_type() == at::kBFloat16, "key dtype should be bfloat16, but got: ", key.scalar_type());
    TORCH_CHECK(value.scalar_type() == at::kBFloat16, "value dtype should be bfloat16, but got: ", value.scalar_type());
    TORCH_CHECK(beta.scalar_type() == at::kBFloat16, "beta dtype should be bfloat16, but got: ", beta.scalar_type());
    TORCH_CHECK(initial_state.scalar_type() == at::kBFloat16,
               "initial_state dtype should be bfloat16, but got: ", initial_state.scalar_type());
    TORCH_CHECK(actual_seq_lengths.scalar_type() == at::kInt,
               "actual_seq_lengths dtype should be int32, but got: ", actual_seq_lengths.scalar_type());
    if (g.has_value() && g.value().defined()) {
        TORCH_CHECK(g.value().scalar_type() == at::kFloat, "g dtype should be float32, but got: ", g.value().scalar_type());
    }

    at::Tensor out{nullptr};
    at::Tensor final_state{nullptr};
    {
        auto local_device = c10::Device(query.device());
        const c10::OptionalDeviceGuard device_guard(local_device);
        out = at::empty_like(value);
        final_state = at::empty_like(initial_state);
    }

    ACLNN_CMD(aclnnChunkGatedDeltaRule, query, key, value, beta, initial_state, actual_seq_lengths, g, scale, out,
              final_state);

    return std::make_tuple(out, final_state);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("chunk_gated_delta_rule", &ChunkGatedDeltaRule, "chunk_gated_delta_rule forward");
}

} // namespace op_api
