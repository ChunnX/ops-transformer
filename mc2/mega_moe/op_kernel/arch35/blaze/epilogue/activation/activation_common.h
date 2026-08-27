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
 * \file activation_common.h
 * \brief A5（arch35）门控激活公共数据结构、参数和常量。
 */

#ifndef MEGA_MOE_ARCH35_ACTIVATION_COMMON_H
#define MEGA_MOE_ARCH35_ACTIVATION_COMMON_H

#if defined(__DAV_C310__)
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../../../common/mega_moe_constants.h"

namespace MegaMoeImpl {
namespace Activation {

// 具体激活实现按 FP32 寄存器宽度划分主循环。
constexpr uint32_t VECTOR_LENGTH_FP32 = AscendC::VECTOR_REG_WIDTH / sizeof(float);

// 输入统一转换为 FP32，供各激活的寄存器计算流程使用。
static constexpr AscendC::MicroAPI::CastTrait CAST_INPUT_TO_FP32 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

// 当前激活结果统一写入 BF16 UB；量化流程在激活之后独立执行。
static constexpr AscendC::MicroAPI::CastTrait CAST_FP32_TO_BF16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

/*
 * 激活直接调用接口所需的高层数据。
 *
 * 此结构只描述 UB 地址和矩阵布局，不保存 RegTensor、MaskReg、AddrReg
 * 等 VecScope 内对象，避免把具体激活的寄存器生命周期泄漏到公共接口。
 */
template <typename InputType>
struct GatedActivationArgs {
    __ubuf__ InputType *gate = nullptr;
    __ubuf__ InputType *up = nullptr;
    __ubuf__ bfloat16_t *output = nullptr;
    __ubuf__ float *topkWeights = nullptr;

    uint32_t validRowCount = 0U;
    uint32_t validColumnCount = 0U;
    uint32_t inputRowStrideElements = 0U;
    uint32_t outputRowStrideElements = 0U;
};

// Epilogue 向激活调度层传递的通用参数。
struct ActivationParams {
    float clampLimit = 0.0F;
    float alpha = 0.0F;
    float beta = 0.0F;
    MegaMoeActMode mode = MegaMoeActMode::SWIGLU;
    MegaMoeActSubMode subMode = MegaMoeActSubMode::DEFAULT;
};

// 各直接调用接口使用最小参数集，防止实现依赖无关配置。
struct SwiGluParams {
    float clampLimit = 0.0F;
};

struct SituParams {
    float clampLimit = 0.0F;
    float beta = 0.0F;
    float invBeta = 0.0F;
    float alpha = 0.0F;
    float invAlpha = 0.0F;
};

struct SwiGluOaiParams {
    float clampLimit = 0.0F;
    float alpha = 0.0F;
    float beta = 0.0F;
};

/*
 * VecScope 外的标量循环规划。
 *
 * validRowCount 和 validColumnCount 描述逻辑有效范围，两个 RowStride
 * 描述 UB 物理布局，单位均为元素。非交织尾块的输入物理行距由调用方
 * 按生产端布局传入（当前为 TileN），不得用 validColumnCount 替代。
 * 具体激活在自己的 VecScope 内根据每行规划创建 mask 并完成补零。
 */
struct ActivationLoopPlan {
    uint32_t validRowCount = 0U;
    uint32_t validColumnCount = 0U;
    uint32_t inputRowStrideElements = 0U;
    uint32_t outputRowStrideElements = 0U;
    uint32_t fullVectorLoopCountPerRow = 0U;
    uint32_t tailElementCountPerRow = 0U;
    uint32_t outputPaddingElementCountPerRow = 0U;
};

template <typename InputType>
__aicore__ inline ActivationLoopPlan BuildActivationLoopPlan(const GatedActivationArgs<InputType> &args)
{
    ActivationLoopPlan plan;
    plan.validRowCount = args.validRowCount;
    plan.validColumnCount = args.validColumnCount;
    plan.inputRowStrideElements = args.inputRowStrideElements;
    plan.outputRowStrideElements = args.outputRowStrideElements;
    plan.fullVectorLoopCountPerRow = args.validColumnCount / VECTOR_LENGTH_FP32;
    plan.tailElementCountPerRow = args.validColumnCount % VECTOR_LENGTH_FP32;
    plan.outputPaddingElementCountPerRow = args.outputRowStrideElements > args.validColumnCount
                                               ? args.outputRowStrideElements - args.validColumnCount
                                               : 0U;
    return plan;
}

} // namespace Activation
} // namespace MegaMoeImpl

#endif // defined(__DAV_C310__)
#endif // MEGA_MOE_ARCH35_ACTIVATION_COMMON_H
