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
 * \file vector.h
 * \brief
 */
#ifndef VECTOR_H
#define VECTOR_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../fa_kernel_public.h"

using namespace fa_kernel::util;
using namespace AscendC;
using namespace AscendC::Reg;
using AscendC::LocalTensor;

namespace fa_base_vector {

enum SparseMode : uint8_t {
    DEFAULT_MASK = 0,
    ALL_MASK,
    LEFT_UP_CAUSAL,
    RIGHT_DOWN_CAUSAL,
    BAND,
    TREE = 9,
};

enum LAYOUT_Q {
    GS,
    SG,
};

enum MaskDataType : uint8_t {
    MASK_BOOL,
    MASK_INT8,
    MASK_UINT8,
};

struct MaskInfo {
    uint32_t s1StartIdx;
    uint32_t s1dealNum;
    uint32_t s1Size;
    uint32_t s2StartIdx;
    uint32_t s2dealNum;
    uint32_t s2Size;
    uint32_t nBaseSize;

    int64_t preToken = 0;
    int64_t nextToken = 0;

    // for bss & bs
    uint32_t attnMaskBatchStride;
    uint32_t attnMaskS1Stride; // 这个变量名与A3 代码不一致，统一整改
    uint32_t attnMaskDstStride = 0;

    LAYOUT_Q layout;
    MaskDataType attnMaskType;
    SparseMode sparseMode;
    uint32_t maskValue;
};

__aicore__ inline uint64_t ComputeAttnMaskOffsetCompress(MaskInfo &info, uint32_t s1StartIdx)
{
    int64_t nextToken = 0; // sparse2 本身原点就是左上角
    if (info.sparseMode == RIGHT_DOWN_CAUSAL) {
        nextToken =
            static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size); // 统一以左上角为原点计算token
    }
    uint64_t offset = 0;
    int64_t delta = nextToken + s1StartIdx - info.s2StartIdx;
    uint32_t attnMaskSizeAlign = AlignUp(info.s2dealNum, 32U);
    if (delta < 0) {
        offset = (-delta) < static_cast<int64_t>(info.s1dealNum) ? (-delta) : info.s1dealNum; // min (-delta, s1Size)
    } else {
        offset = (delta < static_cast<int64_t>(attnMaskSizeAlign) ? delta : attnMaskSizeAlign) *
                 info.attnMaskS1Stride; // min(delta, s2inner)
    }
    return offset;
}

__aicore__ inline uint64_t ComputeAttnMaskOffset(MaskInfo &info, uint32_t s1StartIdx = 0)
{
    return ComputeAttnMaskOffsetCompress(info, s1StartIdx);
}

__aicore__ inline bool IsSkipAttentionmask(MaskInfo &info)
{
    if (info.sparseMode == DEFAULT_MASK || info.sparseMode == ALL_MASK) {
        return false;
    }

    int32_t s1StartIdx = info.s1StartIdx;

    int64_t nextToken = 0; // sparse2 本身原点就在左上角
    if (info.sparseMode == RIGHT_DOWN_CAUSAL) {
        nextToken =
            static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size); // 统一以左上角为原点计算Token
    }

    if (static_cast<int64_t>(info.s2StartIdx + info.s2dealNum) <= static_cast<int64_t>(s1StartIdx) + nextToken) {
        return true;
    }
    return false;
}

template <typename T>
__aicore__ inline void AttentionmaskDataCopy(LocalTensor<T> &attnMaskUb, GlobalTensor<T> &srcGmAddr, MaskInfo &info,
                                             uint32_t s1StartIdx, uint32_t s1EndIdx)
{
    uint32_t attnMaskSizeAlign = AlignUp(info.s2dealNum, 32U);
    uint64_t maskOffset = ComputeAttnMaskOffset(info, s1StartIdx);
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = s1EndIdx - s1StartIdx;
    dataCopyParams.blockLen = info.s2dealNum;
    dataCopyParams.srcStride = info.attnMaskS1Stride - info.s2dealNum;
    dataCopyParams.dstStride = info.attnMaskDstStride;
    DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(attnMaskSizeAlign - info.s2dealNum), 1U};

    DataCopyPad(attnMaskUb, srcGmAddr[maskOffset], dataCopyParams, padParams);
}

template <typename T, bool isReconstructTemp, uint32_t s2BaseSize>
__aicore__ inline void AttentionmaskCopyInForSgLayout(LocalTensor<T> &attnMaskUb, GlobalTensor<T> &srcGmAddr,
                                                      MaskInfo &info)
{
    uint32_t s1StartIdx = info.s1StartIdx;
    uint32_t s1EndIdx = info.s1StartIdx + info.s1dealNum - 1;
    uint32_t s1Count = s1EndIdx - s1StartIdx + 1;
    uint32_t headS1Size = info.s1dealNum;
    if (s1Count > 1) {
        headS1Size = 0;
    }
    uint32_t remainRowCount = info.s1dealNum - headS1Size;
    uint32_t midS1Count = remainRowCount;
    uint32_t attnMaskSizeAlign = AlignUp(info.s2dealNum, 32U);
    uint32_t attnMaskS2Stride = attnMaskSizeAlign + 32 * info.attnMaskDstStride;

    // ub-head
    if (headS1Size > 0) {
        AttentionmaskDataCopy(attnMaskUb, srcGmAddr, info, s1StartIdx, s1StartIdx + 1);
        s1StartIdx++;
    }

    // ub-remain
    if (remainRowCount > 0) {
        uint64_t maskOffset = ComputeAttnMaskOffset(info, s1StartIdx);
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = midS1Count;
        dataCopyParams.blockLen = info.s2dealNum;
        dataCopyParams.srcStride = info.attnMaskS1Stride - info.s2dealNum;
        dataCopyParams.dstStride = info.attnMaskDstStride;
        DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(attnMaskSizeAlign - info.s2dealNum), 1U};
        DataCopyPad(attnMaskUb[headS1Size * attnMaskS2Stride], srcGmAddr[maskOffset], dataCopyParams, padParams);
    }
}

template <typename T, LAYOUT_Q layoutQ, bool isReconstructTemp, uint32_t s2BaseSize>
__aicore__ inline void AttentionmaskCopyIn(LocalTensor<T> &attnMaskUb, GlobalTensor<T> &srcGmAddr, MaskInfo &info)
{
    AttentionmaskCopyInForSgLayout<T, isReconstructTemp, s2BaseSize>(attnMaskUb, srcGmAddr, info);
}

} // namespace fa_base_vector
#endif
