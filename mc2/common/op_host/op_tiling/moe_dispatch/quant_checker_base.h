/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_checker_base.h
 * \brief MOE Distribute Dispatch 量化校验基类定义
 */

#ifndef MOE_DISPATCH_QUANT_CHECKER_BASE_H
#define MOE_DISPATCH_QUANT_CHECKER_BASE_H

#include <cstdint>
#include "defs.h"
#include "utils.h"
#include "register/tilingdata_base.h"

namespace moe_dispatch {

// ========== Tensor 信息结构体（解耦对 TensorDesc 类型依赖）==========
struct ScalesTensorInfo {
    ge::DataType dtype;
    uint32_t dimNum;
    uint64_t dim0;
    uint64_t dim1;
};

class QuantCheckerBase {
public:
    explicit QuantCheckerBase(moe_dispatch_defs::QuantMode mode) : mode_(mode)
    {
    }
    virtual ~QuantCheckerBase() = default;

    // 禁止拷贝和赋值
    QuantCheckerBase(const QuantCheckerBase &) = delete;
    QuantCheckerBase &operator=(const QuantCheckerBase &) = delete;
    QuantCheckerBase(QuantCheckerBase &&) = default;
    QuantCheckerBase &operator=(QuantCheckerBase &&) = default;

    // ========== 模式查询 ==========
    moe_dispatch_defs::QuantMode mode() const
    {
        return mode_;
    }
    uint32_t modeValue() const
    {
        return static_cast<uint32_t>(mode_);
    }

    // ========== 各子类必须实现 ==========

    // scales 是否必须存在（受 xDtype 影响）
    virtual bool ShouldHaveScales(ge::DataType xDtype) const = 0;

    // scales 完整校验（dim + dtype + shape）
    virtual ge::graphStatus CheckScales(gert::TilingContext *ctx, const ScalesTensorInfo &info, uint32_t expectedDim0,
                                        uint32_t h) const = 0;

    // dynamicScales 完整校验
    virtual ge::graphStatus CheckDynamicScales(gert::TilingContext *ctx, const ScalesTensorInfo &info,
                                               uint32_t expectedDim0, uint32_t h) const = 0;

    // tokenMsgSize 计算
    virtual uint32_t CalcTokenMsgSize(uint32_t h) const = 0;

    // ========== 默认实现可 override ==========

    // scales 维度数校验（参数化：不同模式要求不同）
    // expandXDtype: expandX 输出的 dtype，用于静态量化 hif8 特殊校验
    // 返回期望的维度数，0 表示允许多种维度（由调用方进一步校验）
    virtual uint32_t GetExpectedScalesDimNum(ge::DataType expandXDtype) const
    {
        (void)expandXDtype;
        return moe_dispatch_defs::TWO_DIMS; // 默认 2D
    }

    // scales 是否允许多维度（静态量化特有：允许 1D 或 2D）
    virtual bool AllowScalesMultiDim() const
    {
        return false;
    }

    // dynamicScales 维度数
    virtual uint32_t GetDynamicScalesDimNum() const
    {
        return 0U;
    }

    // dynamicScales dtype
    virtual ge::DataType GetDynamicScalesDtype() const
    {
        return ge::DT_FLOAT;
    }

    // dynamicScales dim1 计算（仅 2D 时需要）
    virtual uint32_t CalcDynamicScalesDim1(uint32_t h) const
    {
        return 0U;
    }

    // dynamicScales dim0 期望值计算（通用）
    uint32_t GetExpectedDynamicScalesDim0(uint32_t A, uint32_t tpWorldSize) const
    {
        return A * tpWorldSize;
    }

protected:
    moe_dispatch_defs::QuantMode mode_;
};

} // namespace moe_dispatch

#endif // MOE_DISPATCH_QUANT_CHECKER_BASE_H
