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
 * \file flash_attn_cube_def.h
 * \brief
 */

#ifndef FLASH_ATTN_CUBE_DEF_H_
#define FLASH_ATTN_CUBE_DEF_H_

#include "flash_attn_common_def.h"

struct RunInfoXCube {
    bool isValid = false;
    bool isChangeBatch = false;
    bool isFirstS2Loop = false;
    bool isLastS2Loop = false;

    uint32_t bIdx = 0;
    uint32_t n2Idx = 0;
    uint32_t gS1Idx = 0;
    uint32_t s2Idx = 0;
    uint32_t actMSize = 0;
    uint32_t actSingleLoopS2Size = 0;
};

struct CommonConstInfoCube {
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
};

struct PAConstInfoCube {
    uint32_t blockSize;
    uint32_t maxBlockNumPerBatch;
};

struct ConstInfoXCube : CommonConstInfoCube, PAConstInfoCube {};

#endif // FLASH_ATTN_CUBE_DEF_H_