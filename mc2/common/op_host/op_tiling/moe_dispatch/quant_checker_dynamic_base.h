/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * [PSEUDOCODE] MOE Dispatch Tiling 重构 - DynamicQuantCheckerBase
 * 配套文档：dispatch_analysis_opus_3/architecture_v0.md §4.4 §5.3
 *
 * 抽取 PerToken/PerGroup/MX 共有的 dynamicScales 校验骨架。
 * 子类仅 override dim/dtype/计算路径差异。
 */

#ifndef MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_QUANT_CHECKER_DYNAMIC_BASE_H
#define MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_QUANT_CHECKER_DYNAMIC_BASE_H

#include "quant_checker_base.h"
#include "defs.h"

namespace moe_dispatch_quant {

using namespace moe_dispatch_defs;

/**
 * 动态量化共有骨架基类（PerToken/PerGroup/MX 的公共父类）。
 *
 * CheckDynamicScales 实现统一骨架：
 *   1. desc 不为 null 时才校验（dynamicScales 为可选输出）
 *   2. 按 GetDynamicScalesDimNum() 校验维度数
 *   3. 按 GetDynamicScalesDtype() 校验 dtype
 *   4. 按 expectedDim0 校验 dim0
 *   5. 维度数为 2 时按 CalcDynamicScalesDim1(h) 校验 dim1
 */
class DynamicQuantCheckerBase : public QuantCheckerBase {
public:
    explicit DynamicQuantCheckerBase(uint32_t mode) : QuantCheckerBase(mode) {}

    graphStatus CheckDynamicScales(TilingContext* ctx,
                                   const TensorDesc* desc,
                                   uint32_t expectedDim0,
                                   uint32_t h) const override {
        if (desc == nullptr) {
            return ge::GRAPH_SUCCESS;  // dynamicScales 为可选，不存在时跳过
        }
        // TODO: 1. CheckDimNum(desc->GetShape(), GetDynamicScalesDimNum())
        // TODO: 2. CheckDtypeIn(desc->GetDataType(), {GetDynamicScalesDtype()}, 1)
        // TODO: 3. CheckRangeU32(desc->GetShape().GetDim(0), expectedDim0, expectedDim0, "dynScales.dim0")
        // TODO: 4. if (GetDynamicScalesDimNum() == 2)
        //             CheckRangeU32(desc->GetShape().GetDim(1), CalcDynamicScalesDim1(h), CalcDynamicScalesDim1(h), "dynScales.dim1")
        return ge::GRAPH_SUCCESS;
    }

    // ShouldHaveScales 由各子类实现（PerToken/PerGroup 按 xDtype 判断，MX 返回 false）
    bool ShouldHaveScales(DataType xDtype) const override = 0;

    // CheckScales 由各子类实现（或继承默认 1D float 校验）
    graphStatus CheckScales(TilingContext* ctx, const TensorDesc* desc,
                            uint32_t expectedDim0, uint32_t h) const override {
        // 默认：与 StaticQuantChecker 相同的 1D float 校验
        // TODO: 校验 scales 1D，dtype=float，dim0=expectedDim0
        return ge::GRAPH_SUCCESS;
    }
};

}  // namespace moe_dispatch_quant

#endif  // MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_QUANT_CHECKER_DYNAMIC_BASE_H
