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
 * \brief Tiling for InplacePartialRotaryMulGrad (arch35 / ascend950).
 *
 * Supports only BSND layout and B/S/N broadcast of cos/sin, and only the x == nullptr scenario
 * (dcos / dsin are not computed).
 */

#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_INPLACE_PARTIAL_ROTARY_MUL_GRAD_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_INPLACE_PARTIAL_ROTARY_MUL_GRAD_H

#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "../op_kernel/arch35/inplace_partial_rotary_mul_grad_tiling_data.h"
#include "platform/platform_info.h"

namespace optiling {

// Field order / types must match the kernel-side InplacePartialRotaryMulGradTilingData struct exactly.
BEGIN_TILING_DATA_DEF(InplacePartialRotaryMulGradTiling)
TILING_DATA_FIELD_DEF(uint64_t, b);
TILING_DATA_FIELD_DEF(uint64_t, s);
TILING_DATA_FIELD_DEF(uint64_t, n);
TILING_DATA_FIELD_DEF(uint64_t, d);
TILING_DATA_FIELD_DEF(uint64_t, partialStart);
TILING_DATA_FIELD_DEF(uint64_t, sliceLen);
TILING_DATA_FIELD_DEF(uint64_t, rotaryMode);
TILING_DATA_FIELD_DEF(uint64_t, dyDtypeCode);
TILING_DATA_FIELD_DEF(uint64_t, cosDtypeCode);
TILING_DATA_FIELD_DEF(uint64_t, cosStrideB);
TILING_DATA_FIELD_DEF(uint64_t, cosStrideS);
TILING_DATA_FIELD_DEF(uint64_t, cosStrideN);
TILING_DATA_FIELD_DEF(uint64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, blockFactorM);
TILING_DATA_FIELD_DEF(uint64_t, ubFactorM);
TILING_DATA_FIELD_DEF(uint64_t, isFullBroadcast);
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

private:
    ge::graphStatus CheckParam();
    ge::graphStatus InitTilingData();
    int64_t DTypeCode(ge::DataType dt) const;

private:
    InplacePartialRotaryMulGradTilingData* tilingData_{nullptr};
    int64_t b_{0};
    int64_t s_{0};
    int64_t n_{0};
    int64_t d_{0};
    int64_t partialStart_{0};
    int64_t sliceLen_{0};
    int64_t rotaryMode_{0};
    int64_t cosB_{0};
    int64_t cosS_{0};
    int64_t cosN_{0};
    int64_t cosStrideB_{0};
    int64_t cosStrideS_{0};
    int64_t cosStrideN_{0};
    int64_t isFullBroadcast_{0};
    int64_t dyDtypeCode_{0};
    int64_t cosDtypeCode_{0};
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
};

} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_INPLACE_PARTIAL_ROTARY_MUL_GRAD_H
