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
 * \file apply_rotary_pos_emb.cpp
 * \brief ACLNN Wrapper for aclnnAiInfraApplyRotaryPosEmb
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

std::tuple<at::Tensor, at::Tensor> ApplyRotaryPosEmb(
    const at::Tensor &query, const at::Tensor &key,
    const at::Tensor &cos, const at::Tensor &sin,
    int64_t layout, std::string rotary_mode)
{
    TORCH_CHECK(rotary_mode == "half",
                "rotary_mode only supports 'half', got '", rotary_mode, "'.");

    // Clone inputs so the originals are not mutated (functional / out-of-place semantics)
    at::Tensor query_out = query.clone();
    at::Tensor key_out = key.clone();

    ACLNN_CMD(aclnnAiInfraApplyRotaryPosEmb, query_out, key_out, cos, sin, layout, rotary_mode);

    return std::make_tuple(query_out, key_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("apply_rotary_pos_emb", &ApplyRotaryPosEmb, "apply_rotary_pos_emb");
}

} // namespace op_api
