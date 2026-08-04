/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <acl/acl.h>
#include <torch/extension.h>

#include "aclnn_common.h"

namespace {

std::tuple<at::Tensor, at::Tensor> NpuChunkGatedDeltaRule(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &beta,
    const at::Tensor &initialState,
    const at::Tensor &actualSeqLengths,
    double scale,
    const c10::optional<at::Tensor> &g)
{
    auto out = at::empty(value.sizes(), value.options());
    auto finalState = at::empty(initialState.sizes(), initialState.options());
    float scaleValue = static_cast<float>(scale);

    ACLNN_CMD(aclnnChunkGatedDeltaRule,
              query,
              key,
              value,
              beta,
              initialState,
              actualSeqLengths,
              g,
              scaleValue,
              out,
              finalState);

    return std::make_tuple(out, finalState);
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_chunk_gated_delta_rule",
          &NpuChunkGatedDeltaRule,
          "pytest-only aclnnChunkGatedDeltaRule launcher");
}
