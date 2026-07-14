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
 * \file apply_rotary_pos_emb_grad_half_tiling.h
 * \brief
 */
#ifndef APPLY_ROTARY_POS_EMB_GRAD_HALF_TILING_H
#define APPLY_ROTARY_POS_EMB_GRAD_HALF_TILING_H

#include "apply_rotary_pos_emb_grad_tiling.h"

namespace optiling {

class AiInfraApplyRotaryPosEmbGradHalfTiling : public AiInfraApplyRotaryPosEmbGradTiling {
public:
    explicit AiInfraApplyRotaryPosEmbGradHalfTiling(gert::TilingContext* context)
        : AiInfraApplyRotaryPosEmbGradTiling(context)
    {}

    void Reset(gert::TilingContext* context) override
    {
        TilingBase::Reset(context);
    }

protected:
    bool IsCapable() override
    {
        return true;
    }
    ge::graphStatus DoOpTiling() override;

private:
    void ApplyRopeHalfGradAssignParam();
    void ApplyRopeHalfGradPrintParam();
    ge::graphStatus ApplyRopeHalfGradCheckInput();
    ge::graphStatus ApplyRopeHalfGradCheckAttr();
    ge::graphStatus ApplyRopeHalfGradCheckGradQueryEmbed();
    ge::graphStatus ApplyRopeHalfGradCheckGradKeyEmbed();
    ge::graphStatus ApplyRopeHalfGradCheckCos();
    ge::graphStatus ApplyRopeHalfGradCheckSin();
    ge::graphStatus ApplyRopeHalfGradCheckQuery();
    ge::graphStatus CheckShapeDimMatch(const gert::StorageShape *actualShape, const gert::StorageShape *expectShape,
                                       const std::string &label);
    ge::graphStatus CheckShapeDimMatch3D(const gert::StorageShape *actualShape, const int64_t expVals[3],
                                         const std::string &label);
    ge::graphStatus ApplyRopeHalfGradCheckKey();
    ge::graphStatus ApplyRopeHalfGradCheckInputDatatype();
    ge::graphStatus ApplyRopeHalfGradCheckOptionalInputDatatype();
    ge::graphStatus ApplyRopeHalfGradCheckOutput();
    ge::graphStatus ApplyRopeHalfGradCheckGradQuery();
    ge::graphStatus ApplyRopeHalfGradCheckGradKey();
    ge::graphStatus ApplyRopeHalfGradCheckGradCos();
    ge::graphStatus ApplyRopeHalfGradCheckGradSin();
    ge::graphStatus ApplyRopeHalfGradCheckOutputDatatype();
    ge::graphStatus ApplyRopeHalfGradCheckOptionalOutputDatatype();
    ge::graphStatus ApplyRopeHalfGradSetTiling(const int64_t availableUbSize, const uint64_t maxCoreNum);
    ge::graphStatus ApplyRopeHalfGradInitSplitInfo();
    ge::graphStatus ApplyRopeHalfGradInitParamInfo();
    uint32_t ApplyRopeHalfGradSelectTilingKey();

    // Miscellaneous
    int64_t inputDtypeSize_ = 0;
    int64_t gradQueryEmbedDimSize_ = 0;
    ge::DataType gradQueryEmbedDtype_ = ge::DataType::DT_FLOAT;

    // Tiling data
    AiInfraApplyRotaryPosEmbGradTilingData tiling;

    // Tiling data parameters
    int64_t hasOptionalInputs_ = 0;
    int64_t dQShapeSize_ = 0;
    int64_t dKShapeSize_ = 0;
    int64_t cosSinShapeSize_ = 0;
    int64_t dQKDimB_ = 0;
    int64_t dimS_ = 0;
    int64_t dQDimN_ = 0;
    int64_t dKDimN_ = 0;
    int64_t dimD_ = 0;
    int64_t cosSinDimB_ = 0;
    int64_t cosSinDimN_ = 0;
    int64_t halfDimDAlignNum_ = 0;
    int64_t elementsPerCore_ = 0;
    int64_t lastCoreElements_ = 0;
    int64_t fullTileLoopsPerCore_ = 0;
    int64_t tailElementsPerCore_ = 0;
    int64_t lastCoreFullTileLoops_ = 0;
    int64_t lastCoreTailElements_ = 0;
    int64_t alignUbSize_ = 0;
    int64_t calcUbSize_ = 0;
    int64_t coreUsed_ = 0;
    int64_t cosSinOuterLoops_ = 0;
    int64_t dQHeadInnerLoops_ = 0;
    int64_t dKHeadInnerLoops_ = 0;
    int64_t dQOuterLoopGmStride_ = 0;
    int64_t dKOuterLoopGmStride_ = 0;
    int64_t innerPaddedBlockLen_ = 0;
    int64_t dQInnerPaddedSrcStride_ = 0;
    int64_t dKInnerPaddedSrcStride_ = 0;
};

} // namespace optiling

#endif // ARPE_GRAD_HALF_TILING_H
