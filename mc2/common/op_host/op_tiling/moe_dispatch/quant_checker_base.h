/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * [PSEUDOCODE] MOE Dispatch Tiling 重构公共层 - 量化校验基类
 * 配套文档：dispatch_analysis_opus_3/architecture_v0.md §4.4 §5.3
 * 配套伪代码：dispatch_analysis_opus_3/pseudocode_v0.md §1.3
 */

#ifndef MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_QUANT_CHECKER_BASE_H
#define MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_QUANT_CHECKER_BASE_H

#include <cstdint>
#include "defs.h"
// TODO: include framework headers (TilingContext, graphStatus, DataType, TensorDesc, etc.)

namespace moe_dispatch_quant {

using namespace moe_dispatch_defs;

/**
 * 量化校验抽象基类。
 *
 * 每个 quantMode 对应一个子类，arch/算子差异通过构造参数传入而非继承。
 *
 * ASCII 类图（见 architecture_v0.md §4.4）：
 *   QuantCheckerBase
 *   ├─ UnquantChecker        (mode=0) scales 不需要
 *   ├─ StaticQuantChecker    (mode=1) scales 必需(1D)
 *   └─ DynamicQuantCheckerBase
 *      ├─ PerTokenChecker    (mode=2) dynScales 2D, dtype=float
 *      ├─ PerGroupChecker    (mode=3) dynScales 2D, dtype=float
 *      └─ MxQuantChecker     (mode=4) dynScales 1D, dtype=构造参数
 */
class QuantCheckerBase {
public:
    explicit QuantCheckerBase(uint32_t mode) : mode_(mode) {}
    virtual ~QuantCheckerBase() = default;

    uint32_t mode() const { return mode_; }

    // ---- 各子类必须实现 ----

    // scales 是否必须存在（受 xDtype 影响）
    // UnquantChecker: 不需要；StaticQuantChecker: 必需；MxQuantChecker: 禁止
    virtual bool ShouldHaveScales(DataType xDtype) const = 0;

    // scales 完整校验（dim + dtype + shape）
    virtual graphStatus CheckScales(TilingContext* ctx,
                                    const TensorDesc* desc,
                                    uint32_t expectedDim0,
                                    uint32_t h) const = 0;

    // dynamicScales 完整校验
    virtual graphStatus CheckDynamicScales(TilingContext* ctx,
                                           const TensorDesc* desc,
                                           uint32_t expectedDim0,
                                           uint32_t h) const = 0;

    // tokenMsgSize 计算（各子类公式不同，见 architecture_v0.md §5.3 表格）
    virtual uint32_t CalcTokenMsgSize(uint32_t h) const = 0;

    // ---- 默认实现，子类可 override ----

    // dynamicScales 维度数（Unquant/Static 返回 0 表示不需要）
    virtual uint32_t GetDynamicScalesDimNum() const { return 0; }

    // dynamicScales dtype
    virtual DataType GetDynamicScalesDtype() const { return DT_FLOAT; }

    // dynamicScales dim1 计算（仅 2D 子类有意义）
    virtual uint32_t CalcDynamicScalesDim1(uint32_t h) const { return 0; }

protected:
    uint32_t mode_;
};

// ========== 公共辅助函数（供各 checker 子类内部使用）==========

// dynamicScales dim0 期望值计算（跨 quantMode 通用）
// 见 architecture_v0.md §5.3
inline uint32_t GetExpectedDynamicScalesDim0(uint32_t A, uint32_t tpWorldSize) {
    // TODO: 原始公式来自 v2 校验逻辑，待确认
    return A * tpWorldSize;
}

// QUANT_MODE_MAP 查表校验（expandX dtype 与 quantMode 的关系）
// 见原 v2 tiling 中 CheckQuantModeMap 逻辑
inline graphStatus CheckQuantModeMap(uint32_t quantMode, DataType expandXDtype) {
    // TODO: 按 quantMode → expandX 合法 dtype 映射表查表
    // 校验失败时 OPS_LOG_E 并返回 GRAPH_FAILED
    return ge::GRAPH_SUCCESS;
}

}  // namespace moe_dispatch_quant

#endif  // MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_QUANT_CHECKER_BASE_H
