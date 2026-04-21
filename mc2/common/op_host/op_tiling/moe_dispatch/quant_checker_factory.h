/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * [PSEUDOCODE] MOE Dispatch Tiling 重构 - QuantChecker 工厂函数
 * 配套文档：dispatch_analysis_opus_3/architecture_v0.md §4.4 §5.3
 * 配套伪代码：dispatch_analysis_opus_3/pseudocode_v0.md §1.5
 *
 * 使用方式（见 architecture_v0.md §4.4 "各算子使用方式"）：
 *   // v2-A5 主流程（MX 的 dynamicScales dtype = e8m0）
 *   auto checker = CreateQuantChecker(tc.quantMode, DT_E8M0);
 *
 *   // teardown 主流程（MX 的 dynamicScales dtype = float）
 *   auto checker = CreateQuantChecker(tc.quantMode, DT_FLOAT);
 *
 *   // setup/其他算子（无 arch 差异，使用默认）
 *   auto checker = CreateQuantChecker(tc.quantMode);
 */

#ifndef MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_QUANT_CHECKER_FACTORY_H
#define MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_QUANT_CHECKER_FACTORY_H

#include <memory>
#include "quant_checker_base.h"
#include "quant_checker_unquant.h"
#include "quant_checker_static.h"
#include "quant_checker_pertoken.h"
#include "quant_checker_pergroup.h"
#include "quant_checker_mx.h"
#include "defs.h"
// TODO: include framework header for OPS_LOG_E

namespace moe_dispatch_quant {

using namespace moe_dispatch_defs;

/**
 * 根据 quantMode 创建对应 QuantChecker 实例。
 *
 * @param quantMode       量化模式（0-4）
 * @param mxDynScalesDtype MX_QUANT(mode=4) 的 dynamicScales dtype，
 *                         其他 mode 忽略此参数
 * @return 对应子类实例；quantMode 非法时返回 nullptr（调用方负责报错）
 */
inline std::unique_ptr<QuantCheckerBase> CreateQuantChecker(
        uint32_t quantMode,
        DataType mxDynScalesDtype = DT_FLOAT) {
    switch (quantMode) {
        case UNQUANT:
            return std::make_unique<UnquantChecker>();
        case STATIC_QUANT:
            return std::make_unique<StaticQuantChecker>();
        case PERTOKEN_DYNAMIC_QUANT:
            return std::make_unique<PerTokenChecker>();
        case PERGROUP_DYNAMIC_QUANT:
            return std::make_unique<PerGroupChecker>();
        case MX_QUANT:
            return std::make_unique<MxQuantChecker>(mxDynScalesDtype);
        default:
            // TODO: OPS_LOG_E 输出非法 quantMode 错误
            return nullptr;
    }
}

}  // namespace moe_dispatch_quant

#endif  // MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_QUANT_CHECKER_FACTORY_H
