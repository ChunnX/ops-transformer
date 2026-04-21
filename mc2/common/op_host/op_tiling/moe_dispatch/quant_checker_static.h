/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * [PSEUDOCODE] MOE Dispatch Tiling 重构 - StaticQuantChecker (quantMode=1)
 * 配套文档：dispatch_analysis_opus_3/architecture_v0.md §5.3
 *
 * 特征：
 *   scales      : 必需 (1D, dtype=float, dim0=expectedDim0)
 *   dynamicScales: 不需要
 *   tokenMsgSize : AlignUp(h, ALIGN_256)
 */

#ifndef MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_QUANT_CHECKER_STATIC_H
#define MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_QUANT_CHECKER_STATIC_H

#include "quant_checker_base.h"
#include "defs.h"

namespace moe_dispatch_quant {

using namespace moe_dispatch_defs;

class StaticQuantChecker : public QuantCheckerBase {
public:
    StaticQuantChecker() : QuantCheckerBase(STATIC_QUANT) {}

    bool ShouldHaveScales(DataType /*xDtype*/) const override {
        // 静态量化无论 xDtype 均需要 scales
        return true;
    }

    graphStatus CheckScales(TilingContext* ctx, const TensorDesc* desc,
                            uint32_t expectedDim0, uint32_t /*h*/) const override {
        // TODO: 校验 scales 为 1D
        // TODO: 校验 scales dtype == DT_FLOAT
        // TODO: 校验 scales dim0 == expectedDim0
        return ge::GRAPH_SUCCESS;
    }

    graphStatus CheckDynamicScales(TilingContext* /*ctx*/, const TensorDesc* /*desc*/,
                                   uint32_t /*expectedDim0*/, uint32_t /*h*/) const override {
        // 静态量化不需要 dynamicScales
        return ge::GRAPH_SUCCESS;
    }

    uint32_t GetDynamicScalesDimNum() const override { return 0; }

    uint32_t CalcTokenMsgSize(uint32_t h) const override {
        // AlignUp(h, ALIGN_256)
        return (h + ALIGN_256 - 1) / ALIGN_256 * ALIGN_256;
    }
};

}  // namespace moe_dispatch_quant

#endif  // MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_QUANT_CHECKER_STATIC_H
