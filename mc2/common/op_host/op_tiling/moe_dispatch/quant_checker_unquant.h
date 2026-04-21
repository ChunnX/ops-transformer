/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * [PSEUDOCODE] MOE Dispatch Tiling 重构 - UnquantChecker (quantMode=0)
 * 配套文档：dispatch_analysis_opus_3/architecture_v0.md §5.3
 *
 * 特征：
 *   scales      : 不需要
 *   dynamicScales: 不需要
 *   tokenMsgSize : AlignUp(h, ALIGN_256)
 */

#ifndef MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_QUANT_CHECKER_UNQUANT_H
#define MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_QUANT_CHECKER_UNQUANT_H

#include "quant_checker_base.h"
#include "defs.h"

namespace moe_dispatch_quant {

using namespace moe_dispatch_defs;

class UnquantChecker : public QuantCheckerBase {
public:
    UnquantChecker() : QuantCheckerBase(UNQUANT) {}

    bool ShouldHaveScales(DataType /*xDtype*/) const override {
        return false;  // 无量化不需要 scales
    }

    graphStatus CheckScales(TilingContext* /*ctx*/, const TensorDesc* /*desc*/,
                            uint32_t /*expectedDim0*/, uint32_t /*h*/) const override {
        // 无量化不应调用此函数，逻辑上调用方不会传 scales
        return ge::GRAPH_SUCCESS;
    }

    graphStatus CheckDynamicScales(TilingContext* /*ctx*/, const TensorDesc* /*desc*/,
                                   uint32_t /*expectedDim0*/, uint32_t /*h*/) const override {
        // 无量化不需要 dynamicScales
        return ge::GRAPH_SUCCESS;
    }

    uint32_t GetDynamicScalesDimNum() const override { return 0; }

    uint32_t CalcTokenMsgSize(uint32_t h) const override {
        // AlignUp(h, ALIGN_256)
        return (h + ALIGN_256 - 1) / ALIGN_256 * ALIGN_256;
    }
};

}  // namespace moe_dispatch_quant

#endif  // MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_QUANT_CHECKER_UNQUANT_H
