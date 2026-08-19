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
 * \brief PyTorch extension wrapper for aclnnApplyRotaryPosEmbV2.
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
namespace {
constexpr int64_t ROTARY_MODE_HALF = 0;
} // namespace

std::tuple<at::Tensor, at::Tensor> ApplyRotaryPosEmb(const at::Tensor &query, const at::Tensor &key,
                                                     const at::Tensor &cos, const at::Tensor &sin, int64_t layout,
                                                     int64_t rotary_mode)
{
    TORCH_CHECK(rotary_mode == ROTARY_MODE_HALF, "apply_rotary_pos_emb: rotary_mode only supports 0 (half), got ",
                rotary_mode, ".");

    std::string rotary_mode_str = "half";

    at::Tensor query_out = query.clone();
    at::Tensor key_out = key.clone();

    ACLNN_CMD(aclnnApplyRotaryPosEmbV2, query_out, key_out, cos, sin, layout, rotary_mode_str);

    return std::make_tuple(query_out, key_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("apply_rotary_pos_emb", &ApplyRotaryPosEmb, "apply_rotary_pos_emb");
}
} // namespace op_api
