/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * [PSEUDOCODE] MOE Dispatch Tiling 重构 - PerTokenChecker (quantMode=2)
 * 配套文档：dispatch_analysis_opus_3/architecture_v0.md §5.3
 * 配套伪代码：dispatch_analysis_opus_3/pseudocode_v0.md §1.4
 *
 * 特征：
 *   scales       : 按 xDtype 判断是否需要（xDtype=INT8 时需要）
 *   dynamicScales: 2D, dtype=float, dim1=CeilDiv(h, ALIGN_128)
 *   tokenMsgSize : AlignUp(AlignUp(h, ALIGN_32) + AlignUp(h, ALIGN_128), ALIGN_512)
 */

#ifndef MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_QUANT_CHECKER_PERTOKEN_H
#define MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_QUANT_CHECKER_PERTOKEN_H

#include "quant_checker_dynamic_base.h"
#include "defs.h"

namespace moe_dispatch_quant {

using namespace moe_dispatch_defs;

class PerTokenChecker : public DynamicQuantCheckerBase {
public:
    PerTokenChecker() : DynamicQuantCheckerBase(PERTOKEN_DYNAMIC_QUANT) {}

    bool ShouldHaveScales(DataType xDtype) const override {
        // TODO: 参照原 v2 QUANT_MODE_MAP：xDtype=INT8 时需要 scales
        return xDtype == DT_INT8;
    }

    uint32_t GetDynamicScalesDimNum() const override { return 2; }

    DataType GetDynamicScalesDtype() const override { return DT_FLOAT; }

    uint32_t CalcDynamicScalesDim1(uint32_t h) const override {
        // CeilDiv(h, ALIGN_128)
        return (h + ALIGN_128 - 1) / ALIGN_128;
    }

    uint32_t CalcTokenMsgSize(uint32_t h) const override {
        // AlignUp(AlignUp(h, ALIGN_32) + AlignUp(h, ALIGN_128), ALIGN_512)
        uint32_t part1 = (h + ALIGN_32 - 1) / ALIGN_32 * ALIGN_32;
        uint32_t part2 = (h + ALIGN_128 - 1) / ALIGN_128 * ALIGN_128;
        uint32_t sum   = part1 + part2;
        return (sum + ALIGN_512 - 1) / ALIGN_512 * ALIGN_512;
    }
};

}  // namespace moe_dispatch_quant

#endif  // MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_QUANT_CHECKER_PERTOKEN_H
