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
 * \file apply_rotary_pos_emb_backward.cpp
 * \brief ACLNN Wrapper for aclnnAiInfraApplyRotaryPosEmbGrad
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

constexpr int64_t LAYOUT_BSND_BSH = 1;
constexpr int64_t LAYOUT_SBND = 2;
constexpr int64_t LAYOUT_TND = 4;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> ApplyRotaryPosEmbBackward(
    const at::Tensor &grad_query_embed, const at::Tensor &grad_key_embed,
    const at::Tensor &cos, const at::Tensor &sin,
    const at::Tensor &query, const at::Tensor &key,
    std::string rotary_mode, int64_t layout)
{
    TORCH_CHECK(rotary_mode == "half",
                "rotary_mode only supports 'half', got '", rotary_mode, "'.");
    TORCH_CHECK(layout == LAYOUT_BSND_BSH || layout == LAYOUT_SBND || layout == LAYOUT_TND,
                "Layout only supports BSND_BSH:1 / SBND:2 / TND:4, got ", layout, ".");

    at::Tensor grad_query = at::empty_symint(grad_query_embed.sym_sizes(), grad_query_embed.options());
    at::Tensor grad_key = at::empty_symint(grad_key_embed.sym_sizes(), grad_key_embed.options());
    at::Tensor grad_cos = at::empty_symint(cos.sym_sizes(), cos.options());
    at::Tensor grad_sin = at::empty_symint(sin.sym_sizes(), sin.options());

    ACLNN_CMD(aclnnAiInfraApplyRotaryPosEmbGrad, grad_query_embed, grad_key_embed, cos, sin, query, key,
              rotary_mode, layout, grad_query, grad_key, grad_cos, grad_sin);

    return std::make_tuple(grad_query, grad_key, grad_cos, grad_sin);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("apply_rotary_pos_emb_backward", &ApplyRotaryPosEmbBackward, "apply_rotary_pos_emb_backward");
}

} // namespace op_api
