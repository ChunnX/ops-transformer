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
 * \file fa_static_init_output.h
 * \brief 静态 Tensor 版 GM 填充接口，替代 CANN matmul::InitOutput。
 *
 * CANN 原版 InitOutput 内部通过 GetTPipePtr()->FetchEventID(V_MTE3) 动态申请 eventId，
 * 与 flash_attn 静态 Tensor 编程中编译期写死的 eventId 冲突，批跑时导致 aicore 卡死。
 * 本实现使用固定 V_MTE3 eventId，Duplicate + DataCopyPad 逻辑与 CANN 版一致。
 */
#ifndef FA_STATIC_INIT_OUTPUT_H
#define FA_STATIC_INIT_OUTPUT_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#else
#include "kernel_operator.h"
#endif

using namespace AscendC;

namespace fa_static {

// V_MTE3 专用通道：避开 FA stage1(0/1)、FD output(2/4)、FA stage1 反向 MTE3_V(3/5)
static constexpr uint32_t INIT_OUTPUT_V_MTE3 = 8U;
static constexpr uint32_t INIT_OUTPUT_MAX_REPEAT = 255U;
static constexpr uint32_t INIT_OUTPUT_BLK_BYTE = 32U;
// CANN InitOutput 单次 burst 上限 255×32B；由 vec block InitBuffers 在 UB 尾部静态申请
static constexpr uint32_t INIT_OUTPUT_SCRATCH_BYTES = INIT_OUTPUT_MAX_REPEAT * INIT_OUTPUT_BLK_BYTE;

// 将 GM 区间填充为 value（V Duplicate → MTE3 DataCopyPad to GM）
// scratchBytes：InitBuffers 中分配的静态 UB 区，禁止 PopStackBuffer(LCM) 以免与静态布局冲突
template <typename T>
__aicore__ inline void InitOutput(GlobalTensor<T> gmWorkspaceAddr, uint32_t size, T value,
                                  LocalTensor<uint8_t> scratchBytes)
{
#if (__NPU_ARCH__ == 2201) || (__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102)
#if (__NPU_ARCH__ != 5102)
    if ASCEND_IS_AIC {
        return;
    }
#endif
    if (size == 0U) {
        return;
    }
    LocalTensor<T> popBuffer = scratchBytes.template ReinterpretCast<T>();
    const uint32_t maxBurstSize = (INIT_OUTPUT_MAX_REPEAT * INIT_OUTPUT_BLK_BYTE) / sizeof(T);
    const uint32_t scratchElemCap = INIT_OUTPUT_SCRATCH_BYTES / sizeof(T);
    const uint32_t popSize = scratchElemCap >= maxBurstSize ? maxBurstSize : scratchElemCap;
    if (popSize == 0U) {
        return;
    }
    const uint32_t round = size / popSize;
    const uint32_t tail = size % popSize;
    const uint32_t roundSize = round != 0U ? popSize : 0U;

    Duplicate(popBuffer, value, popSize);
    {
        SetFlag<HardEvent::V_MTE3>(INIT_OUTPUT_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(INIT_OUTPUT_V_MTE3);
    }

    DataCopyExtParams repeatParams;
    repeatParams.blockCount = 1U;
    repeatParams.srcStride = 0U;
    repeatParams.dstStride = 0U;

    uint32_t comOffset = 0U;
    repeatParams.blockLen = static_cast<uint32_t>(roundSize * sizeof(T));
    for (uint32_t index = 0U; index < round; ++index) {
        DataCopyPad(gmWorkspaceAddr[comOffset], popBuffer, repeatParams);
        comOffset += roundSize;
    }
    if (tail != 0U) {
        comOffset = round * roundSize;
        repeatParams.blockLen = static_cast<uint32_t>(tail * sizeof(T));
        DataCopyPad(gmWorkspaceAddr[comOffset], popBuffer, repeatParams);
    }
#endif
}

} // namespace fa_static

// DealActSeqLenIsZero 依赖 GmFormat / OffsetCalculator，放在 memory_copy 类型可见之后
#include "../../../common/op_kernel/memory_copy_arch35.h"

namespace fa_static {

template <GmFormat FORMAT, typename OUT_T, typename OffsetCalcType>
__aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, OffsetCalcType &offsetCalculator,
                                           GlobalTensor<OUT_T> &attentionOutGm,
                                           LocalTensor<uint8_t> initOutputScratch)
{
    if constexpr (FORMAT == GmFormat::TNGD) {
        uint32_t s1Count = offsetCalculator.actualSeqLensQParser.GetTBase(bIdx + 1) -
                           offsetCalculator.actualSeqLensQParser.GetTBase(bIdx);
        for (int s1Idx = 0; s1Idx < static_cast<int>(s1Count); s1Idx++) {
            uint64_t attenOutOffset = offsetCalculator.GetOffset(bIdx, n2Idx, 0, s1Idx, 0);
            InitOutput<OUT_T>(attentionOutGm[attenOutOffset], offsetCalculator.GetStrideN2(), 0, initOutputScratch);
        }
    } else if constexpr (FORMAT == GmFormat::NGTD) {
        uint32_t s1Count = offsetCalculator.actualSeqLensQParser.GetTBase(bIdx + 1) -
                           offsetCalculator.actualSeqLensQParser.GetTBase(bIdx);
        uint32_t gSize = offsetCalculator.GetDimG();
        for (int gIdx = 0; gIdx < static_cast<int>(gSize); gIdx++) {
            uint64_t attenOutOffset = offsetCalculator.GetOffset(bIdx, n2Idx, gIdx, 0, 0);
            InitOutput<OUT_T>(attentionOutGm[attenOutOffset], s1Count * offsetCalculator.GetDimD(), 0,
                              initOutputScratch);
        }
    } else if constexpr (FORMAT == GmFormat::BNGSD) {
        uint64_t attenOutOffset = offsetCalculator.GetOffset(bIdx, n2Idx, 0, 0, 0);
        InitOutput<OUT_T>(attentionOutGm[attenOutOffset], offsetCalculator.GetStrideN2(), 0, initOutputScratch);
    } else if constexpr (FORMAT == GmFormat::BSNGD) {
        uint32_t s1Size = offsetCalculator.GetDimS1();
        for (int s1Idx = 0; s1Idx < static_cast<int>(s1Size); s1Idx++) {
            uint64_t attenOutOffset = offsetCalculator.GetOffset(bIdx, n2Idx, 0, s1Idx, 0);
            InitOutput<OUT_T>(attentionOutGm[attenOutOffset], offsetCalculator.GetStrideN2(), 0, initOutputScratch);
        }
    } else if constexpr (FORMAT == GmFormat::NGBSD) {
        uint32_t gSize = offsetCalculator.GetDimG();
        for (int gIdx = 0; gIdx < static_cast<int>(gSize); gIdx++) {
            uint64_t attenOutOffset = offsetCalculator.GetOffset(bIdx, n2Idx, gIdx, 0, 0);
            InitOutput<OUT_T>(attentionOutGm[attenOutOffset], offsetCalculator.GetStrideB(), 0, initOutputScratch);
        }
    }
}

} // namespace fa_static

#endif // FA_STATIC_INIT_OUTPUT_H
