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
 * \brief 静态 Tensor 版 GM 填充接口，自包含替代 CANN 内部 matmul::InitOutput。
 *
 *  背景：CANN 的 matmul::InitOutput 内部使用
 *      event_t e = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
 *      SetFlag<V_MTE3>(e); WaitFlag<V_MTE3>(e);
 *  即通过 TPipe 动态申请 eventId；而 flash_attn 已完成静态 Tensor 适配，
 *  全部 eventId 在编译期固定，二者并存会触发 eventId 冲突→aicore 异常。
 *
 *  方案：参考 fa_static_buffer.h 的"自包含、不依赖 common"思路，
 *      把 InitOutput / DealActSeqLenIsZero 的最小实现搬入算子自身目录，
 *      使用编译期固定的 V_MTE3 flagId（INIT_OUTPUT_V_MTE3），
 *      并由调用方显式传入 scratch UB（编译期 LocalTensor 构造），
 *      实现完全静态、可预测的同步与内存管理。
 */
#ifndef FA_STATIC_INIT_OUTPUT_H
#define FA_STATIC_INIT_OUTPUT_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "../../../common/op_kernel/memory_copy_arch35.h" // GmFormat 枚举

using namespace AscendC;

namespace fa_static {

// ===== 编译期常量 =====
// V_MTE3 通道的固定 flagId：避开 flash_attn 内部已占用的 0~5/8/9，10 在 V_MTE3 上空闲。
//   若后续在 V_MTE3 通道新增静态 flag，请同步更新本常量并核对无冲突。
static constexpr uint32_t INIT_OUTPUT_V_MTE3 = 10U;

// 单次 DataCopyPad 的最大 burst：MAX_REPEAT_TIMES * ONE_BLK_SIZE 字节
static constexpr uint32_t INIT_OUTPUT_MAX_REPEAT = 255U;
static constexpr uint32_t INIT_OUTPUT_BLK_BYTE   = 32U;

// 调用方必须为本接口预留 ≥ INIT_OUTPUT_SCRATCH_BYTES 的连续 UB（uint8_t 视图）
static constexpr uint32_t INIT_OUTPUT_SCRATCH_BYTES = INIT_OUTPUT_MAX_REPEAT * INIT_OUTPUT_BLK_BYTE; // 8160 B

// -----------------------------------------------------------------------------
// InitOutput：将 GM 区间 [gm, gm+size) 填充为 value
//   scratchBytes：编译期分配的 UB 临时区，字节数 ≥ INIT_OUTPUT_SCRATCH_BYTES
//   仅在 AIV（V_MTE3 可用核）上生效，AIC 上直接 return。
// -----------------------------------------------------------------------------
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
    const uint32_t maxBurstSize   = (INIT_OUTPUT_MAX_REPEAT * INIT_OUTPUT_BLK_BYTE) / sizeof(T);
    const uint32_t scratchElemCap = INIT_OUTPUT_SCRATCH_BYTES / sizeof(T);
    const uint32_t popSize        = scratchElemCap >= maxBurstSize ? maxBurstSize : scratchElemCap;
    if (popSize == 0U) {
        return;
    }

    const uint32_t round     = size / popSize;
    const uint32_t tail      = size % popSize;
    const uint32_t roundSize = round != 0U ? popSize : 0U;

    // 1) V 写满 scratch
    Duplicate(popBuffer, value, popSize);

    // 2) V→MTE3 同步：等 Duplicate 完成再启动 DataCopyPad（UB→GM）
    SetFlag<HardEvent::V_MTE3>(INIT_OUTPUT_V_MTE3);
    WaitFlag<HardEvent::V_MTE3>(INIT_OUTPUT_V_MTE3);

    // 3) 主循环：每轮拷 popSize 个元素
    DataCopyExtParams repeatParams;
    repeatParams.blockCount = 1U;
    repeatParams.srcStride  = 0U;
    repeatParams.dstStride  = 0U;

    uint32_t comOffset = 0U;
    repeatParams.blockLen = static_cast<uint32_t>(roundSize * sizeof(T));
    for (uint32_t index = 0U; index < round; ++index) {
        DataCopyPad(gmWorkspaceAddr[comOffset], popBuffer, repeatParams);
        comOffset += roundSize;
    }

    // 4) 尾块
    if (tail != 0U) {
        comOffset = round * roundSize;
        repeatParams.blockLen = static_cast<uint32_t>(tail * sizeof(T));
        DataCopyPad(gmWorkspaceAddr[comOffset], popBuffer, repeatParams);
    }
#endif
}

// -----------------------------------------------------------------------------
// DealActSeqLenIsZero：actualSeqLen==0 时把对应 batch/n2 的 attentionOut 区段写 0
//   保留与 common/memory_copy.h 同名函数完全一致的语义，只是底层换成 fa_static::InitOutput
// -----------------------------------------------------------------------------
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
