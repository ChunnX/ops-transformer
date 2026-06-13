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
 * \file inplace_partial_rotary_mul_grad_tiling.cpp
 * \brief
 */
#include "inplace_partial_rotary_mul_grad_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

namespace optiling {
constexpr uint32_t ROTARY_MODE_ATTR_IDX = 0;
constexpr uint32_t PARTIAL_SLICE_ATTR_IDX = 1;

ge::graphStatus InplacePartialRotaryMulGradTilingClass::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicoreParams_.numBlocks = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        aicoreParams_.ubSize = ubSizePlatForm;
        socVersion_ = ascendcPlatform.GetSocVersion();
    } else {
        auto compileInfoPtr = context_->GetCompileInfo<InplacePartialRotaryMulGradCompileInfo>();
        if (compileInfoPtr == nullptr) {
            return ge::GRAPH_FAILED;
        }
        aicoreParams_.numBlocks = compileInfoPtr->numBlocks;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
        socVersion_ = compileInfoPtr->socVersion;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradTilingClass::GetShapeAttrsInfo()
{
    auto attrs = context_->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    rotaryMode_ = *(attrs->GetAttrPointer<char>(ROTARY_MODE_ATTR_IDX));
    auto partialSlicePtr = attrs->GetAttrPointer<int64_t>(PARTIAL_SLICE_ATTR_IDX);
    if (partialSlicePtr != nullptr) {
        partialStart_ = partialSlicePtr[0];
        partialEnd_ = partialSlicePtr[1];
    }

    auto dyShape = context_->GetInputShape(0);
    if (dyShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    totalLength_ = 1;
    for (size_t i = 0; i < dyShape->GetDimNum(); ++i) {
        totalLength_ *= dyShape->GetDim(i);
    }
    usedCoreNum_ = aicoreParams_.numBlocks;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradTilingClass::DoOpTiling()
{
    auto tilingData = context_->GetTilingData<InplacePartialRotaryMulGradTiling>();
    if (tilingData == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tilingData->totalLength = static_cast<uint64_t>(totalLength_);
    tilingData->usedCoreNum = static_cast<uint64_t>(usedCoreNum_);
    tilingData->rotaryMode = static_cast<uint64_t>(rotaryMode_);
    tilingData->partialStart = static_cast<uint64_t>(partialStart_);
    tilingData->partialEnd = static_cast<uint64_t>(partialEnd_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4InplacePartialRotaryMulGrad(gert::TilingContext* context)
{
    InplacePartialRotaryMulGradTilingClass tiling(context);
    return tiling.Run();
}

ge::graphStatus TilingPrepareForInplacePartialRotaryMulGrad(gert::TilingParseContext* context)
{
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto compileInfoPtr = context->GetCompiledInfo<InplacePartialRotaryMulGradCompileInfo>();
    if (compileInfoPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfoPtr->numBlocks = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfoPtr->ubSize = ubSizePlatForm;
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(InplacePartialRotaryMulGrad)
    .Tiling(Tiling4InplacePartialRotaryMulGrad)
    .TilingParse<InplacePartialRotaryMulGradCompileInfo>(TilingPrepareForInplacePartialRotaryMulGrad);
} // namespace optiling
