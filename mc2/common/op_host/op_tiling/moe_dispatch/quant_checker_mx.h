/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * [PSEUDOCODE] MOE Dispatch Tiling 重构 - MxQuantChecker (quantMode=4)
 * 配套文档：dispatch_analysis_opus_3/architecture_v0.md §4.4 §5.3
 * 配套伪代码：dispatch_analysis_opus_3/pseudocode_v0.md §1.4
 *
 * 特征：
 *   scales       : 禁止（MX 量化不使用 scales）
 *   dynamicScales: 1D, dtype=构造参数（v2-A5: DT_E8M0，teardown: DT_FLOAT）
 *   tokenMsgSize : AlignUp(AlignUp(h, ALIGN_256) + CeilDiv(h, MX_BLOCK_SIZE), ALIGN_512)
 *
 * arch 差异通过构造参数传入 —— 解耦的核心。
 * 见 architecture_v0.md §4.4 "构造参数化处理 arch 差异"。
 */

#ifndef MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_QUANT_CHECKER_MX_H
#define MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_QUANT_CHECKER_MX_H

#include "quant_checker_dynamic_base.h"
#include "defs.h"

namespace moe_dispatch_quant {

using namespace moe_dispatch_defs;

class MxQuantChecker : public DynamicQuantCheckerBase {
public:
    /**
     * @param dynScalesDtype dynamicScales 的期望 dtype。
     *   调用方按 arch 传入：
     *   - v2-A5 主流程：DT_E8M0
     *   - teardown 主流程：DT_FLOAT
     */
    explicit MxQuantChecker(DataType dynScalesDtype)
        : DynamicQuantCheckerBase(MX_QUANT), dynScalesDtype_(dynScalesDtype) {}

    bool ShouldHaveScales(DataType /*xDtype*/) const override {
        return false;  // MX 量化禁止 scales
    }

    graphStatus CheckScales(TilingContext* /*ctx*/, const TensorDesc* /*desc*/,
                            uint32_t /*expectedDim0*/, uint32_t /*h*/) const override {
        // MX 量化禁止 scales，调用方不应传入
        return ge::GRAPH_SUCCESS;
    }

    uint32_t GetDynamicScalesDimNum() const override { return 1; }  // MX 为 1D

    DataType GetDynamicScalesDtype() const override { return dynScalesDtype_; }  // 由构造参数决定

    // MX 的 1D dynamicScales 不需要 CalcDynamicScalesDim1
    uint32_t CalcDynamicScalesDim1(uint32_t /*h*/) const override { return 0; }

    uint32_t CalcTokenMsgSize(uint32_t h) const override {
        // AlignUp(AlignUp(h, ALIGN_256) + CeilDiv(h, MX_BLOCK_SIZE), ALIGN_512)
        uint32_t part1 = (h + ALIGN_256 - 1) / ALIGN_256 * ALIGN_256;
        uint32_t part2 = (h + MX_BLOCK_SIZE - 1) / MX_BLOCK_SIZE;
        uint32_t sum   = part1 + part2;
        return (sum + ALIGN_512 - 1) / ALIGN_512 * ALIGN_512;
    }

private:
    DataType dynScalesDtype_;  // 由调用方按 arch 传入
};

}  // namespace moe_dispatch_quant

#endif  // MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_QUANT_CHECKER_MX_H
