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
 * \file inplace_partial_rotary_mul_grad_tiling.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_INPLACE_PARTIAL_ROTARY_MUL_GRAD_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_INPLACE_PARTIAL_ROTARY_MUL_GRAD_H

#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "../op_kernel/arch35/inplace_partial_rotary_mul_grad_tiling_data.h"
#include "platform/platform_info.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(InplacePartialRotaryMulGradTiling)
TILING_DATA_FIELD_DEF(uint64_t, totalLength);
TILING_DATA_FIELD_DEF(uint64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, rotaryMode);
TILING_DATA_FIELD_DEF(uint64_t, partialStart);
TILING_DATA_FIELD_DEF(uint64_t, partialEnd);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(InplacePartialRotaryMulGrad, InplacePartialRotaryMulGradTiling)

struct InplacePartialRotaryMulGradCompileInfo {
    int64_t numBlocks;
    uint64_t ubSize;
    platform_ascendc::SocVersion socVersion;
};

class InplacePartialRotaryMulGradTilingClass : public Ops::Transformer::OpTiling::TilingBaseClass
{
public:
    explicit InplacePartialRotaryMulGradTilingClass(gert::TilingContext* context) : TilingBaseClass(context)
    {}

    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
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

    ge::graphStatus DoOpTiling() override;

    ge::graphStatus PostTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetShapeAttrsInfo() override;

    uint64_t GetTilingKey() const override
    {
        return 0;
    }

protected:
    int64_t totalLength_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t rotaryMode_ = 0;
    int64_t partialStart_ = 0;
    int64_t partialEnd_ = 0;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
};

} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_INPLACE_PARTIAL_ROTARY_MUL_GRAD_H
