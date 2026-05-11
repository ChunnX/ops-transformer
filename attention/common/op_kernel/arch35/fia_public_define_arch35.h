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
 * \file fia_public_define_arch35.h
 * \brief
 */
#ifndef FIA_PUBLIC_DEFINE_ARCH35_H
#define FIA_PUBLIC_DEFINE_ARCH35_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../fia_public_define.h"

using namespace AscendC;
using AscendC::AIC;
using AscendC::AIV;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::SetFlag;
using AscendC::ShapeInfo;
using AscendC::SoftmaxConfig;
using AscendC::WaitFlag;

namespace AttentionCommon {

enum class FiaKernelType : uint8_t {
    NO_QUANT = 0,
    ANTI_QUANT,
    FULL_QUANT
};

#if 0
struct FDparams {
    uint32_t *bN2IdxOfFdHead;
    uint32_t *gS1IdxOfFdHead;
    uint32_t *s2SplitNumOfFdHead;
    uint32_t *gS1SplitNumOfFdHead;
    uint32_t *gS1LastPartSizeOfFdHead;
    uint32_t *gS1IdxEndOfFdHead;
    uint32_t *gS1IdxEndOfFdHeadSplit;
    uint32_t usedVecNumOfFd;
    uint32_t gS1BaseSizeOfFd;
};


#endif
struct FDparamsX {
    uint32_t fdCoreEnable;
    uint32_t fdBN2Idx;
    uint32_t fdMIdx;
    uint32_t fdS2SplitNum;
    uint32_t mStart;
    uint32_t mLen;
    uint32_t fdWorkspaceIdx;
};

struct RunInfoX {
    uint32_t loop = 0;
    uint32_t mloop = 0;
    bool isValid = false;
    bool isChangeBatch = false;
    bool isFirstS2Loop = false;
    bool isLastS2Loop = false;

    uint32_t bIdx = 0;
    uint32_t n2Idx = 0;
    uint32_t gS1Idx = 0;
    uint32_t gIdx = 0;
    uint32_t s1Idx = 0;
    uint32_t s2Idx = 0;
    uint64_t actS1Size = 1;   // 当前处理head的S1轴实际大小
    uint64_t actS2Size = 1;   // 当前处理head的S2轴实际大小
    uint32_t actMSize = 0;    // GS1方向上的长度
    uint32_t actMSizeAlign32; // GS1 方向上长度对齐
    uint32_t actVecMSize;     // VEC 视角, 基本块GS1方向长度
    uint32_t vecMbaseIdx;     // VEC 对应的M 轴起始位置,V0 为0， V1 为 V0的actVecMSize

    uint32_t actSingleLoopS2Size = 0; // S2方向长度
    uint32_t actSingleLoopS2SizeAlign;
    // uint32_t curS2LoopTimes = 0;
    bool isS2SplitCore = false;
    uint32_t faTmpOutWsPos = 0; // FA阶段，S2外切，需要写到workspace时，写出到第几块M*D的GM块

    int64_t preTokensLeftUp = 0;
    int64_t nextTokensLeftUp = 0;

    uint64_t qPaddingBeginOffset = 0;
    uint64_t kvPaddingBeginOffset = 0;
};

struct CommonConstInfo {
    /* 轴长度 */
    uint32_t bSize = 0;
    uint64_t t1Size = 0;
    uint64_t t2Size = 0;
    uint32_t dSize = 0;
    uint32_t dSizeV = 0;
    uint32_t dBasicBlock = 0;
    uint32_t dSizeRope = 0;
    uint32_t gSize = 0; /* g轴的大小 */
    uint32_t n2Size = 0;
    uint64_t s1Size = 0;             /* s1总大小 */
    uint64_t s2Size = 0;             /* s2总大小 */
    uint64_t actualSeqLenSize = 0;   /* 用户输入的actualseq的长度 */
    uint64_t actualSeqLenKVSize = 0; /* 用户输入的actualseq_kv的长度 */

    /* FA kernel meta */
    uint32_t bN2Start = 0;
    uint32_t bN2End = 0;
    uint32_t gS1OStart = 0;
    uint32_t gS1OEnd = 0;
    uint32_t s2OStart = 0;
    uint32_t s2OEnd = 0;
    uint32_t coreFirstTmpOutWsPos = 0;

    /* mask */
    uint32_t sparseMode = 0; // sparse
    uint32_t attenMaskBatch = 0;
    uint32_t attenMaskS1Size = 0;
    uint32_t attenMaskS2Size = 0;
    int64_t preTokens = 0;
    int64_t nextTokens = 0;
    bool isRowInvalidOpen = false;
    bool isExistRowInvalid = false;
    float scaleValue = 0.0f;

    /* 核信息 */
    uint32_t aicIdx = 0;
    uint32_t aivIdx = 0;
    uint8_t subBlockIdx = 0;
    uint32_t coreNum = 0;

    /* FA中间结果写出workspace信息 */
    uint32_t accumOutSize = 0;
    uint32_t logSumExpSize = 0;

    /* 输出shape */
    FIA_LAYOUT outputLayout = FIA_LAYOUT::BSH;
};

/* 高阶特性 */
struct PAConstInfo {
    uint32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint32_t paLayoutType = 0;
};

struct LseConstInfo {
    bool isSoftmaxLseEnable = false;
};

struct SinkConstInfo {
    bool learnableSinkFlag = false;
};

struct PseConstInfo {
    uint32_t pseShiftByBatch;
    int64_t pseS1Size;
    int64_t pseS2Size;
    uint32_t pseStride;
};

struct TensorListConstInfo {
    bool isKvContinuous = false; /* 是否为tensorlist */
};

struct PostQuantConstInfo {
    bool isPostQuantPerChnl;
    bool isPostQuantBF16;
    bool isPostQuantOffsetExist;
    float postQuantScaleValue;
    float postQuantOffsetValue;
};

struct LeftPaddingConstInfo {
    bool isQHasLeftPadding;
    bool isKVHasLeftPadding;
    int64_t queryRightPaddingSize;
    int64_t kvRightPaddingSize;
};

struct SysPrefixConstInfo {
    bool isActualSharedPrefixLenNull = true;
    int64_t actualKVPrefixSize = 0; /* 保存prefix实际长度 */
    int64_t kvPrefixSize = 0;       /* 保存prefix shape完整长度 */
    int64_t prefixLoopCount = 0;    /* 保存prefix参与的S2方向循环次数 */
};

template <FiaKernelType>
struct ConstInfo_t;

template <>
struct ConstInfo_t<FiaKernelType::NO_QUANT> : CommonConstInfo,
                                              PAConstInfo,
                                              LseConstInfo,
                                              SinkConstInfo,
                                              PseConstInfo,
                                              TensorListConstInfo,
                                              PostQuantConstInfo,
                                              LeftPaddingConstInfo,
                                              SysPrefixConstInfo {};

template <>
struct ConstInfo_t<FiaKernelType::FULL_QUANT> : CommonConstInfo,
                                                PAConstInfo,
                                                LseConstInfo,
                                                TensorListConstInfo {};

// struct FusedTransposeInfo {
//     // 以下是FlashDecode分支区分的信息
//     uint32_t n2Idx = 0;
//     uint32_t bIdx = 0;

//     // 以下是需要用公式计算的信息
//     uint32_t s1StartIdx = 0;
//     uint32_t s1EndIdx = 0;
//     uint32_t s1Count = 0;
//     uint32_t gStartIdx = 0;
//     uint32_t gEndIdx = 0;
//     uint32_t gCount = 0;
// };

// struct MSplitInfo {
//     uint32_t nBufferIdx = 0U;
//     uint32_t nBufferStartM = 0U;
//     uint32_t nBufferDealM = 0U;
//     uint32_t vecStartM = 0U;
//     uint32_t vecDealM = 0U;
// };

// enum class TASK_DEAL_MODE : uint32_t
// {
//     DEAL_ZERO = 0,
//     SKIP = 1,
//     CREATE_TASK = 2
// };

// template <LayOutTypeEnum LAYOUT, typename CONST_INFO_T>
// __aicore__ inline void GetGS1Idx(uint32_t gS1Idx, uint32_t &gIdx, uint32_t &s1Idx, const CONST_INFO_T &constInfo)
// {
//     // GS1
//     if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD || LAYOUT == LayOutTypeEnum::LAYOUT_NBSD ||
//                   LAYOUT == LayOutTypeEnum::LAYOUT_NTD) {
//         gIdx = gS1Idx / constInfo.qSeqSize;
//         s1Idx = gS1Idx % constInfo.qSeqSize;
//     } else {
//         // S1G
//         s1Idx = gS1Idx / constInfo.gSize;
//         gIdx = gS1Idx % constInfo.gSize;
//     }
// }

// __aicore__ inline int64_t ClipSInnerToken(int64_t sInnerToken, int64_t minValue, int64_t maxValue)
// {
//     sInnerToken = sInnerToken > minValue ? sInnerToken : minValue;
//     sInnerToken = sInnerToken < maxValue ? sInnerToken : maxValue;
//     return sInnerToken;
// }


} // namespace AttentionCommon

#endif // FIA_PUBLIC_DEFINE_H