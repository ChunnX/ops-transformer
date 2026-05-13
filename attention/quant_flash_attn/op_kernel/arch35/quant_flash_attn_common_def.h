/**
 * Copyright (c) 2025 Huawei Technologies Co.,
    Ltd.
 * This program is free software,
    you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS,
    WITHOUT WARRANTIES OF ANY KIND,
    EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
    MERCHANTABILITY,
    OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_flash_attn_common_def.h
 * \brief
 */


#ifndef QUANT_FLASH_ATTN_COMMON_DEF_H_
#define QUANT_FLASH_ATTN_COMMON_DEF_H_

namespace QFA_KERNEL {

constexpr float FLOAT_ZERO = 0;
// // // BUFFER的字节数
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

struct CommonConstInfo {
    /* 轴长度 */
    uint32_t bSize = 0;
    uint64_t t1Size = 0;
    uint64_t t2Size = 0;
    uint32_t dSize = 0;
    uint32_t dSizeV = 0;
    uint32_t dBasicBlock = 0;
    uint32_t gSize = 0; /* g轴的大小 */
    uint32_t n2Size = 0;
    uint64_t s1Size = 0; /* s1总大小 */
    uint64_t s2Size = 0; /* s2总大小 */
    uint32_t qCuSeqLensSize = 0;   /* 用户输入的cu_seqlens_q的长度 */
    uint32_t kvCuSeqLensSize = 0; /* 用户输入的cu_seqlens_kv的长度 */
    uint32_t qSeqUsedSize = 0;   /* 用户输入的seqused_q的长度 */
    uint32_t kvSeqUsedSize = 0; /* 用户输入的seqused_kv的长度 */

    /* FA kernel meta */
    uint32_t bN2Start = 0;
    uint32_t bN2End = 0;
    uint32_t gS1OStart = 0;
    uint32_t gS1OEnd = 0;
    uint32_t s2OStart = 0;
    uint32_t s2OEnd = 0;
    uint32_t coreFirstTmpOutWsPos = 0;

    /* mask */
    uint32_t sparseMode = 0;
    uint32_t attenMaskS1Size = 0;
    uint32_t attenMaskS2Size = 0;
    int64_t preTokens = 0;
    int64_t nextTokens = 0;
    float scaleValue = 0.0;

    /* 核信息 */
    uint32_t aicIdx = 0;
    uint32_t aivIdx = 0;
    uint8_t subBlockIdx = 0;
    uint32_t coreNum = 0;

    /* FA中间结果写出workspace信息 */
    uint32_t accumOutSize = 0;
    uint32_t logSumExpSize = 0;
};

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

struct ConstInfo :
CommonConstInfo, PAConstInfo, LseConstInfo, SinkConstInfo {
};

struct RunInfo {
    uint32_t loop = 0;
    uint32_t mloop = 0;
    bool isValid = false;
    bool isFirstS2Loop = false;
    bool isLastS2Loop = false;
    bool isUpdatePScale = false;
    bool isC2Sync = false;

    uint32_t bIdx = 0;
    uint32_t n2Idx = 0;
    uint32_t gS1Idx = 0;
    uint32_t gIdx = 0;
    uint32_t s1Idx = 0;
    uint32_t s2Idx = 0;
    uint32_t curS2LoopIdx = 0; // 在当前核处理的S2的循环下标
    uint64_t actS1Size = 1;   // 当前处理head的S1轴实际大小
    uint64_t actS2Size = 1;   // 当前处理head的S2轴实际大小
    uint32_t actMSize = 0;    // GS1方向上的长度
    uint32_t actMSizeAlign32; // GS1 方向上长度对齐
    uint32_t actVecMSize = 0;  // VEC 视角, 基本块GS1方向长度
    uint32_t vecMbaseIdx = 0;    // VEC 对应的M 轴起始位置,V0 为0， V1 为 V0的actVecMSize

    uint32_t actSingleLoopS2Size = 0; // S2方向长度
    uint32_t actSingleLoopS2SizeAlign = 0;
    uint32_t curS2LoopTimes = 0;
    bool isS2SplitCore = false;
    uint32_t faTmpOutWsPos = 0;  // FA阶段，S2外切，需要写到workspace时，写出到第几块M*D的GM块

    int64_t preTokensLeftUp = 0;
    int64_t nextTokensLeftUp = 0;
};

// kernel stream related struct
struct FDparams {
    uint32_t fdCoreEnable = 0;
 	uint32_t fdBN2Idx = 0;
    uint32_t fdMIdx = 0;
    uint32_t fdS2SplitNum = 0;
    uint32_t mStart = 0;
    uint32_t mLen = 0;
    uint32_t fdWorkspaceIdx = 0;
};

}

#endif