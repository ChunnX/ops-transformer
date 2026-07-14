/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file apply_rotary_pos_emb_grad_tiling.cpp
 * \brief
 */
#include "apply_rotary_pos_emb_grad_tiling.h"
#include "apply_rotary_pos_emb_grad_half_tiling.h"
#include "tiling_base/tiling_templates_registry.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"

namespace optiling {
static constexpr uint32_t ATTR_ROTARY_MODE_IDX = 0;
static constexpr uint32_t ATTR_LAYOUT_IDX = 1;

ge::graphStatus ApplyRotaryPosEmbGradTiling::GetShapeAttrsInfo()  // 获取两个属性的值
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs is null."), return ge::GRAPH_FAILED);

    auto rotaryModePtr = attrs->GetAttrPointer<char>(ATTR_ROTARY_MODE_IDX);
    if (rotaryModePtr != nullptr) {
        rotaryMode_ = rotaryModePtr;
    }
    OP_LOGI("[ApplyRotaryPosEmbGrad]", "[rotary_mode]: %s", rotaryMode_.data());

    const int64_t *layoutMode = attrs->GetAttrPointer<int64_t>(ATTR_LAYOUT_IDX);
    layout_ = layoutMode == nullptr ? 1 : *layoutMode;
    OP_LOGI("[ApplyRotaryPosEmbGrad]", "[layout]: %lld", layout_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradTiling::GetPlatformInfo()  // 获取硬件信息
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicoreParams_.blockDim = static_cast<uint64_t>(ascendcPlatform.GetCoreNumAiv());
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        aicoreParams_.ubSize = ubSizePlatForm;
        socVersion_ = ascendcPlatform.GetSocVersion();
    } else {
        auto compileInfoPtr = context_->GetCompileInfo<ApplyRotaryPosEmbGradCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        aicoreParams_.blockDim = compileInfoPtr->numBlocks;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
        socVersion_ = compileInfoPtr->socVersion;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4ApplyRotaryPosEmbGrad(gert::TilingContext *context)
{
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForApplyRotaryPosEmbGrad(gert::TilingParseContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context->GetNodeName(), "context is null."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(ApplyRotaryPosEmbGrad, ApplyRotaryPosEmbGradHalfTiling, 60000);

IMPL_OP_OPTILING(ApplyRotaryPosEmbGrad)
    .Tiling(Tiling4ApplyRotaryPosEmbGrad)
    .TilingParse<ApplyRotaryPosEmbGradCompileInfo>(TilingPrepareForApplyRotaryPosEmbGrad);
}  // namespace optiling
