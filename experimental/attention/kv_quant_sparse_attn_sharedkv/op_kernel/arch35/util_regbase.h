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
 * \file util_regbase.h
 * \brief
 */

#ifndef KV_QUANT_SAS_UTIL_REGBASE_H
#define KV_QUANT_SAS_UTIL_REGBASE_H

#include "util.h"

using AscendC::TQue;
using AscendC::QuePosition;

namespace regbaseutil {
constexpr int64_t MAX_PRE_NEXT_TOKENS = 0x7FFFFFFF;

#define COMMON_RUN_PARAM \
    int64_t boIdx; \
    int64_t s1oIdx; \
    int64_t n2oIdx; \
    int64_t goIdx; \
    int64_t s2LoopEndIdx;          /* S2方向的循环控制信息 souter层确定 */ \
    int64_t s2LineStartIdx = 0;    /* S2方向按行的起始位置 */ \
    int64_t s2LineEndIdx;          /* S2方向按行的结束位置 */ \
    int64_t s2CmpLineEndIdx; \
    /* cube视角的sOuter，在SAMEAB场景中cubeSOuterSize为两倍的 halfS1RealSize souter层确定 */ \
    uint32_t s1RealSize; \
    uint32_t halfS1RealSize; \
    uint32_t firstHalfS1RealSize; \
    uint32_t mRealSize; \
    uint32_t halfMRealSize; \
    uint32_t firstHalfMRealSize; \
    int64_t attentionOutOffset;    /* attentionOut的offset souter层确定 */ \
    int32_t actualS1Size;      /* Q的actualSeqLength */ \
    int32_t actualS2Size    /* KV的actualSeqLength */ \

struct RunParamStr {  // 分核与切块需要使用到参数
    COMMON_RUN_PARAM;
    /* 推理新增 */
    int64_t gs1LoopStartIdx;
    int64_t gs1LoopEndIdx;
    // BN循环生产的数据
    int64_t preTokensPerBatch = MAX_PRE_NEXT_TOKENS; // 左上顶点的pretoken
    int64_t nextTokensPerBatch = MAX_PRE_NEXT_TOKENS; // 左上顶点的nexttoken

    // NBS1循环生产的数据
    int64_t sOuterOffset;               // 单个S内 souter的 souterIdx * halfS1RealSize souter层确定
    int64_t cubeSOuterOffset;           // 单个S内 souter的 souterIdx * halfS1RealSize souter层确定
    int64_t mOuterOffset;
    int64_t cubeMOuterOffset;

    // lse 输出offset
    int64_t softmaxLseOffset;       // souter层确定

    int64_t qSNumInOneBlock;
    int64_t oriKvLoopEndIdx;
    int64_t cmpKvLoopEndIdx;
};

struct RunInfo {
    int64_t s2LoopCount; /* s2循环当前的循环index */
    int32_t s2RealSize; /* s2方向基本块的真实长度 */
    int64_t s2StartIdx; /* s2的起始位置，sparse场景下可能不是0 */
    int64_t s2LoopLimit;
    int64_t s1oIdx; /* s1轴的index */
    int64_t attentionOutOffset;
    int64_t oriKvLoopEndIdx;
    uint8_t taskIdMod2;
    uint8_t taskIdMod3;
    uint8_t multiCoreIdxMod2;
};


}

#endif // KV_QUANT_SAS_UTIL_REGBASE_H
