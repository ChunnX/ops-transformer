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
    // 当前 Block Epilogue 使用 CeilAlign(validColumnCount, 32)；直接调用接口保留
    // 更广输出物理行距下的两段写回和补零能力。
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

struct SituGluParams {
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

    // 以下字段均为尾块的标量派生结果。UpdateMask 接收可写引用，具体激活需复制 mask 计数后再调用。
    uint32_t tailColumnOffsetElements = 0U;
    uint32_t tailComputeMaskElementCount = 0U;
    uint32_t tailStoreMaskElementCount = 0U;
    uint32_t additionalPaddingStoreMaskElementCount = 0U;
    // 0/1 标志：是否需要尾 Vector 计算，为保持原 VecScope for-loop 形态直接作为循环上界。
    uint16_t needTailVectorCompute = 0U;
    // 0/1 标志：是否需要额外补零写回，为保持原 VecScope for-loop 形态直接作为循环上界。
    uint16_t needAdditionalPaddingStore = 0U;
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
    if (plan.tailElementCountPerRow > 0U) {
        plan.tailColumnOffsetElements = plan.fullVectorLoopCountPerRow * VECTOR_LENGTH_FP32;
        plan.tailComputeMaskElementCount = plan.tailElementCountPerRow;
        plan.needTailVectorCompute = 1U;

        const uint32_t tailAndPaddingElementCount =
            plan.tailElementCountPerRow + plan.outputPaddingElementCountPerRow;
        if (tailAndPaddingElementCount <= VECTOR_LENGTH_FP32) {
            plan.tailStoreMaskElementCount = tailAndPaddingElementCount;
        } else {
            plan.tailStoreMaskElementCount = VECTOR_LENGTH_FP32;
            plan.additionalPaddingStoreMaskElementCount = tailAndPaddingElementCount - VECTOR_LENGTH_FP32;
            plan.needAdditionalPaddingStore = 1U;
        }
    }
    return plan;
}

} // namespace Activation
} // namespace MegaMoeImpl

#endif // defined(__DAV_C310__)
#endif // MEGA_MOE_ARCH35_ACTIVATION_COMMON_H
