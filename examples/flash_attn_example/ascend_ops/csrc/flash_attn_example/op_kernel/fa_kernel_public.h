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
 * \file fa_kernel_public.h
 * \brief
 */
#ifndef FA_KERNEL_PUBLIC_H
#define FA_KERNEL_PUBLIC_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "kernel_tiling/kernel_tiling.h"
#include "./stdarg.h"
#include <type_traits>
#include "../fia_tiling_data.h"

using namespace AscendC;
using AscendC::AIC;
using AscendC::AIV;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::SetFlag;
using AscendC::ShapeInfo;
using AscendC::SoftmaxConfig;
using AscendC::WaitFlag;
using AscendC::DataFormat;
using AscendC::DataCopyParams;
using AscendC::DataCopyPadParams;
using AscendC::BinaryRepeatParams;
using AscendC::IsSameType;
using AscendC::HardEvent;
using AscendC::TQue;
using AscendC::QuePosition;
using matmul::MatmulType;

namespace fa_kernel {
namespace hardware {
constexpr int32_t BLOCK_BYTES = 32;
constexpr uint16_t REG_BYTES = 256;
constexpr int32_t FLOAT_BASE_SIZE = 8;
constexpr uint64_t FLOAT_BYTES = 4;
constexpr uint64_t KB_TO_BYTES = 1024;
constexpr uint64_t L0C_SIZE = 256;
constexpr uint32_t CV_RATIO = 2;
constexpr uint32_t FA_BYTE_BLOCK = 32;
constexpr uint64_t BYTE_BLOCK = 32UL;
constexpr uint32_t NEGATIVE_MIN_VALUE_FP32 = 0xFF7FFFFF;
} // namespace hardware

namespace config {
enum class DTemplateType {
    Aligned16 = 16,
    Aligned32 = 32,
    Aligned48 = 48,
    Aligned64 = 64,
    Aligned80 = 80,
    Aligned96 = 96,
    Aligned128 = 128,
    Aligned160 = 160,
    Aligned192 = 192,
    Aligned256 = 256,
    Aligned512 = 512,
    Aligned576 = 576,
    Aligned768 = 768,
    NotAligned,
};

enum class S1TemplateType {
    Aligned16 = 16,
    Aligned64 = 64,
    Aligned128 = 128,
    Aligned256 = 256,
    Aligned512 = 512,
    NotAligned,
};

enum class S2TemplateType {
    Aligned16 = 16,
    Aligned32 = 32,
    Aligned64 = 64,
    Aligned128 = 128,
    Aligned256 = 256,
    Aligned512 = 512,
    Aligned1024 = 1024,
    NotAligned,
};
} // namespace config

namespace layout {
enum class LayOutTypeEnum {
    None = 0,
    LAYOUT_BSH = 1,
    LAYOUT_SBH = 2,
    LAYOUT_BNSD = 3,
    LAYOUT_TND = 4,
    LAYOUT_NTD_TND = 5,
    LAYOUT_NTD = 6,
    LAYOUT_NBSD = 7
};

enum class FA_LAYOUT : uint32_t
{
    BSH = 0,
    BSND = 0,
    BNSD = 1,
    NZ = 2,
    TND = 3,
    NBSD = 4,
    NTD = 5
};

enum class FaKernelType : uint8_t {
    NO_QUANT = 0,
    ANTI_QUANT,
    FULL_QUANT
};

enum class TASK_DEAL_MODE : uint32_t
{
    DEAL_ZERO = 0,
    SKIP = 1,
    CREATE_TASK = 2,
    SKIP_S1OUT = 3,
    SKIP_ZERO = 4,
    S2_END = 5,
    NOT_START = 6,
};
} // namespace layout

struct CombineParamsX {
    uint32_t combineCoreEnable;
    uint32_t combineBN2Idx;
    uint32_t combineMIdx;
    uint32_t combineS2SplitNum;
    uint32_t mStart;
    uint32_t mLen;
    uint32_t combineWorkspaceIdx;
};

struct RunInfoX {
    uint32_t loop = 0;
    uint32_t mloop = 0;
    bool isValid = false;
    bool isFirstS2Loop = false;
    bool isLastS2Loop = false;

    uint32_t bIdx = 0;
    uint32_t n2Idx = 0;
    uint32_t gS1Idx = 0;
    uint32_t gIdx = 0;
    uint32_t s1Idx = 0;
    uint32_t s2Idx = 0;
    uint32_t realN2Idx = 0;
    uint64_t actS1Size = 1;
    uint64_t actS2Size = 1;
    uint32_t actMSize = 0;
    uint32_t actMSizeAlign32;
    uint32_t actVecMSize;
    uint32_t vecMbaseIdx;

    uint32_t actSingleLoopS2Size = 0;
    uint32_t actSingleLoopS2SizeAlign;
    bool isS2SplitCore = false;
    uint32_t faTmpOutWsPos = 0;

    int64_t preTokensLeftUp = 0;
    int64_t nextTokensLeftUp = 0;

    uint64_t qPaddingBeginOffset = 0;
    uint64_t kvPaddingBeginOffset = 0;
};

struct CommonConstInfo {
    uint32_t bSize;
    uint64_t t1Size;
    uint64_t t2Size;
    uint32_t dSize;
    uint32_t dSizeV;
    uint32_t dBasicBlock;
    uint32_t gSize;
    uint32_t n2Size;
    uint32_t realGSize;
    uint32_t realN2Size;
    uint64_t s1Size;
    uint64_t s2Size;
    uint64_t actualSeqLenSize;
    uint64_t actualSeqLenKVSize;

    uint32_t bN2Start;
    uint32_t bN2End;
    uint32_t gS1OStart;
    uint32_t gS1OEnd;
    uint32_t s2OStart;
    uint32_t s2OEnd;
    uint32_t coreFirstTmpOutWsPos;

    uint32_t sparseMode;
    uint32_t attnMaskBatch;
    uint32_t attnMaskS1Size;
    uint32_t attnMaskS2Size;
    int64_t preTokens;
    int64_t nextTokens;
    bool isRowInvalidOpen;
    bool isExistRowInvalid;
    float softmaxScale;

    uint32_t aicIdx;
    uint32_t aivIdx;
    uint8_t subBlockIdx;
    uint32_t coreNum;

    uint32_t totalSize;

    uint32_t accumOutSize;
    uint32_t logSumExpSize;

    layout::FA_LAYOUT outputLayout;

    bool isQHasLeftPadding;
    bool isKVHasLeftPadding;
    int64_t queryRightPaddingSize;
    int64_t kvRightPaddingSize;
    uint32_t blockSize;
};

template <layout::FaKernelType>
struct ConstInfo_t;

template <>
struct ConstInfo_t<layout::FaKernelType::NO_QUANT> : CommonConstInfo {};

struct ConstInfo {
    static constexpr float FLOAT_ZERO = 0;
    uint32_t bN2Start = 0U;
    uint32_t gS1Start = 0U;
    uint32_t s2Start = 0U;
    uint32_t bN2End = 0U;
    uint32_t gS1End = 0U;
    uint32_t s2End = 0U;

    uint32_t preLoadNum = 0U;
    uint32_t nBufferMBaseSize = 0U;
    uint32_t syncV1NupdateC2 = 0U;
    uint32_t syncV0C1 = 0U;
    uint32_t syncC1V1 = 0U;
    uint32_t syncV1C2 = 0U;
    uint32_t syncC2V2 = 0U;
    uint32_t syncC2V1 = 0U;

    float softmaxScale = 0;
    uint32_t mmResUbSize = 0U;
    uint32_t vec1ResUbSize = 0U;
    uint32_t bmm2ResUbSize = 0U;
    uint64_t batchSize = 0ULL;
    uint64_t gSize = 0ULL;
    uint64_t qHeadNum = 0ULL;
    uint64_t kvHeadNum = 0ULL;
    uint64_t headDim = 0;
    uint64_t headDimAlign = 0;
    uint64_t kvSeqSize = 0ULL;
    uint64_t qSeqSize = 1ULL;
    int64_t preToken = 0;
    int64_t nextToken = 0;
    uint64_t attnMaskBatchStride = 0ULL;
    uint32_t splitKVNum = 0U;
    layout::FA_LAYOUT outputLayout;
    uint32_t subBlockNum = 2;

    uint32_t attnMaskStride = 0ULL;
    uint32_t sparseMode = 0;
    uint32_t tndFDCoreArrLen = 0U;
    uint32_t coreStartKVSplitPos = 0U;

    uint32_t actualLenQDims = 0U;
    uint32_t actualLenDims = 0U;
    uint32_t mBaseSize = 1ULL;
    uint32_t s2BaseSize = 1ULL;
    uint32_t l2CacheOffFlag = 0;

    bool attnMaskFlag = false;
    bool accumQSeqFlag = false;
    bool accumKVSeqFlag = false;
    bool needInit = false;
    bool isRowInvalid = false;
    bool isExistRowInvalid = false;

    bool batchContinuous = true;

    bool isLegacyIfa = false;

    bool headS2Split = false;
    bool tailS2Split = false;
};

namespace util {
constexpr uint16_t SHIFT_NUM_6 = 6;
constexpr uint16_t ADD_NUM_63 = 63;

__aicore__ constexpr uint16_t Align64Func(uint16_t data)
{
    return (data + ADD_NUM_63) >> SHIFT_NUM_6 << SHIFT_NUM_6;
}

__aicore__ inline int64_t ClipSInnerToken(int64_t sInnerToken, int64_t minValue, int64_t maxValue)
{
    sInnerToken = sInnerToken > minValue ? sInnerToken : minValue;
    sInnerToken = sInnerToken < maxValue ? sInnerToken : maxValue;
    return sInnerToken;
}

template <typename T1, typename T2> __aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a > b) ? (a) : (b);
}

template <typename T1, typename T2> __aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}
} // namespace util
} // namespace fa_kernel

#endif // FA_KERNEL_PUBLIC_H
