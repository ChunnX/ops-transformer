/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ifa_public_define.h
 * \brief
 */
#ifndef FIA_PUBLIC_DEFINE_H
#define FIA_PUBLIC_DEFINE_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

using namespace AscendC;
using AscendC::AIC;
using AscendC::AIV;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::SetFlag;
using AscendC::ShapeInfo;
using AscendC::SoftmaxConfig;
using AscendC::WaitFlag;
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

namespace AttentionCommon {

// 将isCheckTiling设置为false
constexpr SoftmaxConfig FIA_SOFTMAX_FLASHV2_CFG = {false};
// 将isCheckTiling设置为false, 输入输出的max&sum&exp的shape为(m, 1)
constexpr SoftmaxConfig FIA_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC = {false, 0, 0, SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC};

enum class FIA_LAYOUT : uint32_t
{
    BSH = 0,
    BSND = 0,
    BNSD = 1,
    NZ = 2,
    TND = 3,
    NBSD = 4,
    NTD = 5
};

template <typename Q_T, typename KV_T, typename OUT_T, typename ORIGIN_T, const bool PAGE_ATTENTION = false,
          const bool FLASH_DECODE = false, FIA_LAYOUT LAYOUT_T = FIA_LAYOUT::BSH, const uint8_t ANTIQUANT_MODE = 0,
          const bool SHARED_PREFIX = false, FIA_LAYOUT KV_LAYOUT_T = FIA_LAYOUT::BSH, typename... Args>
struct FIAType {
    using queryType = Q_T;
    using kvType = KV_T;
    using outputType = OUT_T;
    using orginalType = ORIGIN_T;
    static constexpr bool pageAttention = PAGE_ATTENTION;
    static constexpr bool flashDecode = FLASH_DECODE;
    static constexpr FIA_LAYOUT layout = LAYOUT_T;
    static constexpr uint8_t antiquantMode = ANTIQUANT_MODE;
    static constexpr bool sharedPrefix = SHARED_PREFIX;
    static constexpr FIA_LAYOUT kvLayout = KV_LAYOUT_T;
};

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

struct RunInfo {
    uint32_t loop;
    uint32_t bIdx;
    uint32_t gIdx;
    uint32_t s1Idx;
    uint32_t s2Idx;
    uint32_t bn2IdxInCurCore;
    uint32_t curSInnerLoopTimes;
    uint64_t tndBIdxOffset;
    uint64_t tensorAOffset;
    uint64_t tensorBOffset;
    uint64_t tensorARopeOffset;
    uint64_t tensorBRopeOffset;
    uint64_t attenOutOffset;
    uint64_t attenMaskOffset;
    uint32_t actualSingleProcessSInnerSize;
    uint32_t actualSingleProcessSInnerSizeAlign;
    bool isFirstSInnerLoop;
    bool isChangeBatch;
    uint32_t s2BatchOffset;
    uint32_t gSize;
    uint32_t s1Size;
    uint32_t s2Size;
    uint32_t tndIsS2SplitCore;
    uint32_t tndCoreStartKVSplitPos;
    bool isValid = false;

    static constexpr uint32_t n2Idx = 0;
    uint64_t actS1Size = 1;

    uint32_t gS1Idx;
    uint64_t actS2Size = 1;
    uint32_t actMBaseSize;
    bool isLastS2Loop;
    int32_t preTokensPerBatch = 0;
    int32_t nextTokensPerBatch = 0;
};

struct ConstInfo {
    // CUBE与VEC核间同步的模式
    static constexpr uint32_t FIA_SYNC_MODE2 = 2;
    // BUFFER的字节数
    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    static constexpr uint32_t BUFFER_SIZE_BYTE_64B = 64;
    static constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
    static constexpr uint32_t BUFFER_SIZE_BYTE_512B = 512;
    static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
    static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192;
    static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;
    static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;
    // FP32的0值和极大值
    static constexpr float FLOAT_ZERO = 0;
    static constexpr float FLOAT_MAX = 3.402823466e+38F;

    // preLoad的总次数
    uint32_t preLoadNum = 0U;
    uint32_t nBufferMBaseSize = 0U;
    // CUBE和VEC的核间同步EventID
    uint32_t syncV1NupdateC2 = 0U;
    uint32_t syncV0C1 = 0U;
    uint32_t syncC1V1 = 0U;
    uint32_t syncV1C2 = 0U;
    uint32_t syncC2V2 = 0U;
    uint32_t syncC2V1 = 0U;

    uint32_t mmResUbSize = 0U;   // Matmul1输出结果GM上的大小
    uint32_t vec1ResUbSize = 0U; // Vector1输出结果GM上的大小
    uint32_t bmm2ResUbSize = 0U; // Matmul2输出结果GM上的大小
    uint64_t batchSize = 0ULL;
    uint64_t gSize = 0ULL;
    uint64_t qHeadNum = 0ULL;
    uint64_t kvHeadNum;
    uint64_t headDim;
    uint64_t headDimRope;
    uint64_t headDimAlign;
    uint64_t kvSeqSize = 0ULL;        // kv最大S长度
    uint64_t qSeqSize = 1ULL;         // q最大S长度
    uint32_t kvCacheBlockSize = 0;    // PA场景的block size
    uint32_t maxBlockNumPerBatch = 0; // PA场景的最大单batch block number
    uint32_t splitKVNum = 0U;         // S2核间切分的切分份数
    FIA_LAYOUT outputLayout;          // 输出的Transpose格式
    bool attenMaskFlag = false;
    uint64_t attenMaskSize = 0ULL;
    bool slidingFlag = false;
    bool needInit = false;
    int32_t preToken = 0;
    int32_t nextToken = 0;

    uint32_t actualLenQDims = 0U; // query的actualSeqLength 的维度
    uint32_t actualLenDims = 0U;  // KV 的actualSeqLength 的维度

    // TND
    uint32_t s2Start = 0U; // TND场景下，S2的起始位置
    uint32_t s2End = 0U;   // 单核TND场景下S2循环index上限

    uint32_t bN2Start = 0U;
    uint32_t bN2End = 0U;
    uint32_t gS1Start = 0U;
    uint32_t gS1End = 0U;

    uint32_t tndFDCoreArrLen = 0U;     // TNDFlashDecoding相关分核信息array的长度
    uint32_t coreStartKVSplitPos = 0U; // TNDFlashDecoding kv起始位置

    uint32_t mBaseSize = 1ULL;
    uint32_t s2BaseSize = 1ULL;
};

struct FusedTransposeInfo {
    // 以下是FlashDecode分支区分的信息
    uint32_t n2Idx = 0;
    uint32_t bIdx = 0;

    // 以下是需要用公式计算的信息
    uint32_t s1StartIdx = 0;
    uint32_t s1EndIdx = 0;
    uint32_t s1Count = 0;
    uint32_t gStartIdx = 0;
    uint32_t gEndIdx = 0;
    uint32_t gCount = 0;
};

struct MSplitInfo {
    uint32_t nBufferIdx = 0U;
    uint32_t nBufferStartM = 0U;
    uint32_t nBufferDealM = 0U;
    uint32_t vecStartM = 0U;
    uint32_t vecDealM = 0U;
};

template <FIA_LAYOUT LAYOUT_T>
__aicore__ inline void GetGS1Idx(uint32_t gS1Idx, uint32_t &gIdx, uint32_t &s1Idx, AttentionCommon::ConstInfo &constInfo)
{
    // GS1
    if constexpr (LAYOUT_T == FIA_LAYOUT::BNSD) {
        gIdx = gS1Idx / constInfo.qSeqSize;
        s1Idx = gS1Idx % constInfo.qSeqSize;
    } else {
        // S1G
        s1Idx = gS1Idx / constInfo.gSize;
        gIdx = gS1Idx % constInfo.gSize;
    }
}

} // namespace

#endif // FIA_PUBLIC_DEFINE_H