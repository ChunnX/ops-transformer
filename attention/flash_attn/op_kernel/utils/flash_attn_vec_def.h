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
 * \file flash_attn_vec_def.h
 * \brief
 */


#ifndef FLASH_ATTN_VEC_DEF_H_
#define FLASH_ATTN_VEC_DEF_H_

#include "flash_attn_common_def.h"

struct FDparamsX {
    uint32_t fdCoreEnable;
    uint32_t fdBN2Idx;
    uint32_t fdMIdx;
    uint32_t fdS2SplitNum;
    uint32_t mStart;
    uint32_t mLen;
    uint32_t fdWorkspaceIdx;
};

struct RunInfoXVec {
    uint32_t loop = 0;
    uint32_t mloop = 0;
    bool isValid = false;
    bool isFirstS2Loop = false;
    bool isLastS2Loop = false;

    uint32_t bIdx = 0;
    uint32_t n2Idx = 0;
    uint32_t gS1Idx = 0;
    uint32_t s2Idx = 0;
    uint64_t actS1Size = 1;
    uint64_t actS2Size = 1;
    uint32_t actMSize = 0;
    uint32_t actSingleLoopS2Size = 0;

    int64_t preTokensLeftUp = 0;
    int64_t nextTokensLeftUp = 0;

    uint32_t gIdx = 0;
    uint32_t actMSizeAlign32;
    uint32_t actVecMSize;
    uint32_t vecMbaseIdx;
    uint32_t actSingleLoopS2SizeAlign;
    bool isS2SplitCore = false;
    uint32_t faTmpOutWsPos = 0;
};

struct CommonConstInfoVec {
    uint32_t bSize;
    uint32_t dSize;
    uint32_t dSizeV;
    uint32_t dBasicBlock;
    uint32_t gSize;
    uint32_t n2Size;
    uint64_t s1Size;
    uint64_t s2Size;
    uint64_t cuSeqLensQSize;
    uint64_t cuSeqLensKVSize;
    uint64_t seqUsedQSize;
    uint64_t seqUsedKvSize;
    bool isKvContinuous;
    uint32_t aicIdx;
    uint32_t coreNum;
    uint32_t sparseMode;
    int64_t preTokens;
    int64_t nextTokens;

    uint64_t t1Size;
    uint32_t aivIdx;
    uint8_t subBlockIdx;
    uint32_t attenMaskBatch;
    uint32_t attenMaskS1Size;
    uint32_t attenMaskS2Size;
    bool isExistRowInvalid;
    float scaleValue;
    uint32_t accumOutSize;
    uint32_t logSumExpSize;
    bool needInitOutput;
    FA_LAYOUT outputLayout;
};

struct LseConstInfoVec {
    bool isSoftmaxLseEnable;
};

struct SinkConstInfoVec {
    bool learnableSinkFlag = false;
};

struct ConstInfoXVec : CommonConstInfoVec, LseConstInfoVec, SinkConstInfoVec {};

#endif // FLASH_ATTN_VEC_DEF_H_