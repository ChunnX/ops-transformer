/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file attenmask_gs1_static_qfa.h
 * \brief quant_flash_attn 静态Tensor版 attenmask 拷入函数（Mutex Lock/Unlock 同步）。
 *        复用 common attenmask_gs1_arch35.h 的类型与纯计算 helper（MaskInfo/枚举/
 *        AttentionmaskDataCopy/ComputeAttenMaskOffset/MaskUbCopyS1G/IsSkipAttentionmask），
 *        仅新增带静态 Mutex ID 的拷贝函数（加 Static 后缀避免重定义）。
 *        qfa 以 isReconstructTemp=true 调用（CheckIsSkipAttenMask 早退分支编译剔除，
 *        跳过判断由外层 AttenMaskCopyIn 的 isSkip 逻辑承担）。
 */
#ifndef ATTENMASK_GS1_STATIC_QFA_H
#define ATTENMASK_GS1_STATIC_QFA_H

#if __has_include("../../../common/op_kernel/arch35/attenmask_gs1_arch35.h")
#include "../../../common/op_kernel/arch35/attenmask_gs1_arch35.h"
#else
#include "../../common/op_kernel/arch35/attenmask_gs1_arch35.h"
#endif

template <typename T, bool isReconstructTemp, uint32_t s2BaseSize>
__aicore__ inline void AttentionmaskCopyInForGsLayoutStatic(LocalTensor<T> &attenMaskUb, GlobalTensor<T> &srcGmAddr,
                                                            MaskInfo &info, bool isPre, uint32_t mte2ToVEventId)
{
    int32_t s1StartIdx = info.gs1StartIdx % info.s1Size;
    int32_t s1EndIdx = (info.gs1StartIdx + info.gs1dealNum - 1) % info.s1Size + 1;
    uint32_t attenMaskS2Stride = AttentionCommon::Align(info.s2dealNum, 32U) + 32 * info.attenMaskDstStride;
    uint32_t headS1Count = (s1StartIdx == 0) ? 0 : (info.s1Size - s1StartIdx);
    // midGCount: gs1dealNum 跨 G 边界时，中间完整 s1Size 行的重复次数
    uint32_t midGCount = (info.gs1dealNum > headS1Count) ? (info.gs1dealNum - headS1Count) / info.s1Size : 0;
    // 无完整G（midGCount==0）时，模板写入会超出 buffer 边界，改为走 if 分支直接从 GM 拷贝
    if (info.gs1dealNum <= info.s1Size || midGCount == 0) {
        Mutex::Lock<PIPE_MTE2>(mte2ToVEventId);
        if (s1StartIdx + info.gs1dealNum > info.s1Size) {
            AttentionmaskDataCopy(attenMaskUb, srcGmAddr, info, s1StartIdx, info.s1Size, isPre);
            LocalTensor<T> attenMaskSecUb = attenMaskUb[(info.s1Size - s1StartIdx) * attenMaskS2Stride];
            AttentionmaskDataCopy(attenMaskSecUb, srcGmAddr, info, 0, s1EndIdx, isPre);
        } else {
            AttentionmaskDataCopy(attenMaskUb, srcGmAddr, info, s1StartIdx, s1EndIdx, isPre);
        }
        Mutex::Unlock<PIPE_MTE2>(mte2ToVEventId);
    } else {
        // 有完整G：先从GM拷贝一个完整G到UB作为模板，再用UB→UB DataCopy填充head/mid/tail
        uint32_t tailS1Size = (info.gs1dealNum - headS1Count) % info.s1Size;

        Mutex::Lock<PIPE_MTE2>(mte2ToVEventId);
        LocalTensor<T> attenMaskSecUb = attenMaskUb[headS1Count * attenMaskS2Stride];
        AttentionmaskDataCopy(attenMaskSecUb, srcGmAddr, info, 0, info.s1Size, isPre);
        Mutex::Unlock<PIPE_MTE2>(mte2ToVEventId);

        Mutex::Lock<PIPE_V>(mte2ToVEventId);
        // head
        if (headS1Count > 0) {
            DataCopy(attenMaskUb, attenMaskUb[info.s1Size * attenMaskS2Stride], headS1Count * attenMaskS2Stride);
        }
        // mid
        for (uint32_t i = 1; i < midGCount; i++) {
            DataCopy(attenMaskUb[(headS1Count + i * info.s1Size) * attenMaskS2Stride], attenMaskSecUb,
                     info.s1Size * attenMaskS2Stride);
        }
        // tail
        if (tailS1Size > 0) {
            DataCopy(attenMaskUb[(headS1Count + midGCount * info.s1Size) * attenMaskS2Stride], attenMaskSecUb,
                     tailS1Size * attenMaskS2Stride);
        }
        Mutex::Unlock<PIPE_V>(mte2ToVEventId);
    }
    PipeBarrier<PIPE_V>();
}

template <typename T, bool isReconstructTemp, uint32_t s2BaseSize>
__aicore__ inline void AttentionmaskCopyInForSgLayoutStatic(LocalTensor<T> &attenMaskUb, GlobalTensor<T> &srcGmAddr,
                                                            MaskInfo &info, bool isPre, uint32_t mte2ToVEventId)
{
    uint32_t s1StartIdx = info.gs1StartIdx / info.gSize;
    uint32_t s1EndIdx = (info.gs1StartIdx + info.gs1dealNum - 1) / info.gSize;
    uint32_t s1Count = s1EndIdx - s1StartIdx + 1;
    uint32_t headGSize = info.gs1dealNum;
    if (s1Count > 1) {
        headGSize = (info.gs1StartIdx % info.gSize == 0) ? 0 : (info.gSize - info.gs1StartIdx % info.gSize);
    }
    uint32_t remainRowCount = info.gs1dealNum - headGSize;
    uint32_t midS1Count = remainRowCount / info.gSize;
    uint32_t tailGSize = remainRowCount % info.gSize;
    uint32_t attenMaskSizeAlign = AttentionCommon::Align(info.s2dealNum, 32U);
    uint32_t attenMaskS2Stride = attenMaskSizeAlign + 32 * info.attenMaskDstStride;

    Mutex::Lock<PIPE_MTE2>(mte2ToVEventId);
    // ub-head
    if (headGSize > 0) {
        AttentionmaskDataCopy(attenMaskUb, srcGmAddr, info, s1StartIdx, s1StartIdx + 1, isPre);
        s1StartIdx++;
    }

    // ub-remain
    if (remainRowCount > 0) {
        uint64_t maskOffset = ComputeAttenMaskOffset(info, s1StartIdx, isPre);
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = midS1Count + (tailGSize > 0);
        dataCopyParams.blockLen = info.s2dealNum;
        dataCopyParams.srcStride = info.attenMaskS1Stride - info.s2dealNum;
        dataCopyParams.dstStride = (info.gSize - 1) * attenMaskS2Stride / 32 + info.attenMaskDstStride;
        // blockLen + padParams.len + dstStride = gSize * s2Size
        DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(attenMaskSizeAlign - info.s2dealNum), 1U};
        DataCopyPad(attenMaskUb[headGSize * attenMaskS2Stride], srcGmAddr[maskOffset], dataCopyParams, padParams);
    }
    Mutex::Unlock<PIPE_MTE2>(mte2ToVEventId);

    Mutex::Lock<PIPE_V>(mte2ToVEventId);
    MaskUbCopyS1G<T, s2BaseSize>(attenMaskUb, headGSize, info.gSize, tailGSize, midS1Count);
    Mutex::Unlock<PIPE_V>(mte2ToVEventId);
    PipeBarrier<PIPE_V>();
}

template <typename T, MaskFormat maskFormat, bool isReconstructTemp, uint32_t s2BaseSize>
__aicore__ inline void AttentionmaskCopyInStatic(LocalTensor<T> &attenMaskUb, GlobalTensor<T> &srcGmAddr,
                                                 MaskInfo &info, bool isPre, uint32_t mte2ToVEventId)
{
    if constexpr (maskFormat == MaskFormat::GS) {
        AttentionmaskCopyInForGsLayoutStatic<T, isReconstructTemp, s2BaseSize>(attenMaskUb, srcGmAddr, info, isPre,
                                                                               mte2ToVEventId);
    } else if constexpr (maskFormat == MaskFormat::SG) {
        AttentionmaskCopyInForSgLayoutStatic<T, isReconstructTemp, s2BaseSize>(attenMaskUb, srcGmAddr, info, isPre,
                                                                               mte2ToVEventId);
    }
}
#endif
