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
 * \file apply_rotary_pos_emb_grad_tiling.h
 * \brief
 */
#ifndef APPLY_ROTARY_POS_EMB_GRAD_TILING_H
#define APPLY_ROTARY_POS_EMB_GRAD_TILING_H

#include <tiling/tiling_api.h>
#include "register/tilingdata_base.h"
#include "../../../common/include/tiling_base/tiling_base.h"
#include "../../../common/include/err/ops_err.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ApplyRotaryPosEmbHalfGradParams)
TILING_DATA_FIELD_DEF(int64_t, hasOptionalInputs);
TILING_DATA_FIELD_DEF(int64_t, layout);
TILING_DATA_FIELD_DEF(int64_t, dQShapeSize);
TILING_DATA_FIELD_DEF(int64_t, dKShapeSize);
TILING_DATA_FIELD_DEF(int64_t, cosSinShapeSize);

TILING_DATA_FIELD_DEF(int64_t, dQKDimB);
TILING_DATA_FIELD_DEF(int64_t, dimS);
TILING_DATA_FIELD_DEF(int64_t, dQDimN);
TILING_DATA_FIELD_DEF(int64_t, dKDimN);
TILING_DATA_FIELD_DEF(int64_t, dimD);
TILING_DATA_FIELD_DEF(int64_t, cosSinDimB);
TILING_DATA_FIELD_DEF(int64_t, cosSinDimN);
TILING_DATA_FIELD_DEF(int64_t, halfDimDAlignNum);

TILING_DATA_FIELD_DEF(int64_t, elementsPerCore);        // coreData
TILING_DATA_FIELD_DEF(int64_t, lastCoreElements);       // corelast
TILING_DATA_FIELD_DEF(int64_t, fullTileLoopsPerCore);   // copyLoop 前核loop次数
TILING_DATA_FIELD_DEF(int64_t, tailElementsPerCore);    // copyTail 前核尾loop处理的元素个数
TILING_DATA_FIELD_DEF(int64_t, lastCoreFullTileLoops);  // 尾核loop次数
TILING_DATA_FIELD_DEF(int64_t, lastCoreTailElements);   // 尾核尾loop处理的元素个数

TILING_DATA_FIELD_DEF(int64_t, alignUbSize);  // 每个buffer的所需空间（包含pad）
TILING_DATA_FIELD_DEF(int64_t, calcUbSize);   // 每次搬入/计算实际元素空间（不包含pad）
TILING_DATA_FIELD_DEF(int64_t, coreUsed);

TILING_DATA_FIELD_DEF(int64_t, cosSinOuterLoops);
TILING_DATA_FIELD_DEF(int64_t, dQHeadInnerLoops);
TILING_DATA_FIELD_DEF(int64_t, dKHeadInnerLoops);
TILING_DATA_FIELD_DEF(int64_t, dQOuterLoopGmStride);
TILING_DATA_FIELD_DEF(int64_t, dKOuterLoopGmStride);
TILING_DATA_FIELD_DEF(int64_t, innerPaddedBlockLen);
TILING_DATA_FIELD_DEF(int64_t, dQInnerPaddedSrcStride);
TILING_DATA_FIELD_DEF(int64_t, dKInnerPaddedSrcStride);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ApplyRotaryPosEmbHalfGradParamsOp, ApplyRotaryPosEmbHalfGradParams)

BEGIN_TILING_DATA_DEF(ApplyRotaryPosEmbGradTilingData)
TILING_DATA_FIELD_DEF_STRUCT(ApplyRotaryPosEmbHalfGradParams, applyRopeHalfGradParams);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ApplyRotaryPosEmbGrad, ApplyRotaryPosEmbGradTilingData)

struct ApplyRotaryPosEmbGradCompileInfo {
    uint64_t numBlocks = 0;
    uint64_t ubSize = 0;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
};

enum class ApplyRotaryPosEmbGradLayout : int64_t {
    BSND = 1,
    SBND = 2,
    BNSD = 3,
    TND = 4
};

enum class ApplyRotaryPosEmbGradMode : int64_t {
    HALF = 0
};

enum class ApplyRotaryPosEmbGradTilingKey : uint32_t {
    TILING_KEY_LAYOUT_COMMON = 0,
    TILING_KEY_LAYOUT_BSND = 1
};

class ApplyRotaryPosEmbGradTiling : public Ops::Transformer::OpTiling::TilingBase {
public:
    explicit ApplyRotaryPosEmbGradTiling(gert::TilingContext *context) : TilingBase(context)
    {}

    void Reset(gert::TilingContext *context) override
    {
        TilingBase::Reset(context);
    }

protected:
    ge::graphStatus GetPlatformInfo() override;

    ge::graphStatus GetWorkspaceSize() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    bool IsCapable() override
    {
        return true;
    }

    ge::graphStatus DoOpTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PostTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetShapeAttrsInfo() override;

    uint64_t GetTilingKey() const override
    {
        return context_->GetTilingKey();
    }

protected:
    std::string_view rotaryMode_ = "half";
    int64_t layout_ = 1;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
};
}  // namespace optiling
#endif  // APPLY_ROTARY_POS_EMB_GRAD_TILING_H