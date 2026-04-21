/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * [PSEUDOCODE] MOE Dispatch Tiling 重构 - PerGroupChecker (quantMode=3)
 * 配套文档：dispatch_analysis_opus_3/architecture_v0.md §5.3
 *
 * 特征：
 *   scales       : 按 xDtype 判断（同 PerToken）
 *   dynamicScales: 2D, dtype=float, dim1=CeilDiv(h,32) 偶数对齐 (EVEN_ALIGN)
 *   tokenMsgSize : AlignUp(AlignUp(h, ALIGN_128) + CeilDiv(h, ALIGN_128), ALIGN_512)
 */

#ifndef MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_QUANT_CHECKER_PERGROUP_H
#define MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_QUANT_CHECKER_PERGROUP_H

#include "quant_checker_dynamic_base.h"
#include "defs.h"

namespace moe_dispatch_quant {

using namespace moe_dispatch_defs;

class PerGroupChecker : public DynamicQuantCheckerBase {
public:
    PerGroupChecker() : DynamicQuantCheckerBase(PERGROUP_DYNAMIC_QUANT) {}

    bool ShouldHaveScales(DataType xDtype) const override {
        // TODO: 参照原 v2 QUANT_MODE_MAP
        return xDtype == DT_INT8;
    }

    uint32_t GetDynamicScalesDimNum() const override { return 2; }

    DataType GetDynamicScalesDtype() const override { return DT_FLOAT; }

    uint32_t CalcDynamicScalesDim1(uint32_t h) const override {
        // CeilDiv(h, PERGROUP_BLOCK_SIZE) 向上偶数对齐
        uint32_t raw = (h + PERGROUP_BLOCK_SIZE - 1) / PERGROUP_BLOCK_SIZE;
        return (raw + EVEN_ALIGN - 1) / EVEN_ALIGN * EVEN_ALIGN;
    }

    uint32_t CalcTokenMsgSize(uint32_t h) const override {
        // AlignUp(AlignUp(h, ALIGN_128) + CeilDiv(h, ALIGN_128), ALIGN_512)
        uint32_t part1 = (h + ALIGN_128 - 1) / ALIGN_128 * ALIGN_128;
        uint32_t part2 = (h + ALIGN_128 - 1) / ALIGN_128;
        uint32_t sum   = part1 + part2;
        return (sum + ALIGN_512 - 1) / ALIGN_512 * ALIGN_512;
    }
};

}  // namespace moe_dispatch_quant

#endif  // MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_QUANT_CHECKER_PERGROUP_H
