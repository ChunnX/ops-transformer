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

#ifndef SMLA_UTIL_REGBASE_H
#define SMLA_UTIL_REGBASE_H

#include "util.h"

using AscendC::TQue;
using AscendC::QuePosition;

namespace regbaseutil {
constexpr int64_t MAX_PRE_NEXT_TOKENS = 0x7FFFFFFF;
enum class VselrIndexEnum {
    GT_64_AND_LTE_128_INDEX = 0,
    GT_0_AND_LTE_64_INDEX = 1
};

#define RUN_PARAM_COMMON_FIELDS \
    int64_t boIdx; \
    int64_t s1oIdx; \
    int64_t n2oIdx; \
    int64_t goIdx; \
    int64_t gSplitSize;         /* split-G模式下当前AIC处理的G轴行数 */ \
    int64_t s2LoopEndIdx;       /* S2方向的循环控制信息 souter层确定 */ \
    int64_t s2LineStartIdx = 0; /* S2方向按行的起始位置 */ \
    int64_t s2LineOriEndIdx;    /* S2方向按行的结束位置 */ \
    int64_t s2CmpLineEndIdx; \
    int64_t s2LineCmpEndIdx; \
    /* cube视角的sOuter，在SAMEAB场景中cubeSOuterSize为两倍的 halfS1RealSize souter层确定 */ \
    uint32_t s1RealSize; \
    uint32_t halfS1RealSize; \
    uint32_t firstHalfS1RealSize; \
    uint32_t mRealSize; \
    uint32_t halfMRealSize; \
    uint32_t firstHalfMRealSize; \
    int64_t attentionOutOffset; /* attentionOut的offset souter层确定 */ \
    int32_t actualS1Size;       /* Q的actualSeqLength */ \
    int32_t actualS2OriSize;    /* ori_kv的真实使用长度 */ \
    int32_t actualS2CmpSize;    /* cmp_kv的真实使用长度 */ \
    int32_t cmpResidual;        /* cmp的余数，用于mask计算 */ \
    /* BN循环生产的数据 */ \
    int64_t gs1LoopStartIdx; \
    int64_t gs1LoopEndIdx; \
    int64_t preTokensPerBatch = MAX_PRE_NEXT_TOKENS;     /* 左上顶点的pretoken */ \
    int64_t nextTokensPerBatchOri = MAX_PRE_NEXT_TOKENS; /* ori 左上顶点的nexttoken */ \
    int64_t nextTokensPerBatchCmp = MAX_PRE_NEXT_TOKENS; /* cmp 左上顶点的nexttoken */ \
    /* NBS1循环生产的数据 */ \
    int64_t sOuterOffset;     /* 单个S内 souter的 souterIdx * halfS1RealSize souter层确定 */ \
    int64_t cubeSOuterOffset; /* 单个S内 souter的 souterIdx * halfS1RealSize souter层确定 */ \
    int64_t qSNumInOneBlock; \
    int64_t oriKvLoopEndIdx; \
    int64_t cmpKvLoopEndIdx; \
    int64_t firstFdDataWorkspaceIdx = 0; \
    int64_t s2SplitIdx = 0; \
    int64_t baseBlockNumPerReductionBlock = 1; /* s2方向一个规约块有几个基本块 */ \
    bool isCrossCoreSplit = false; \
    bool isFirstS2SplitCore = true

#define RUN_PARAM_TOPK_LSE_FIELDS \
    /* lse/topk len 相关参数, 仅在非HIGH_PERF模式存在 */ \
    int64_t softmaxLseOffset /* lse 输出offset, souter层确定 */

template <bool HIGH_PERF = false>
struct RunParamStr { // 分核与切块需要使用到参数
    RUN_PARAM_COMMON_FIELDS;
    RUN_PARAM_TOPK_LSE_FIELDS;
};

template <>
struct RunParamStr<true> { // HIGH_PERF: 无topk len且无lse场景, 剔除topk/lse相关字段
    RUN_PARAM_COMMON_FIELDS;
};

#define RUN_INFO_COMMON_FIELDS \
    uint64_t s2StartIdx; /* s2的起始位置，sparse场景下可能不是0 */ \
    int64_t s2EndIdx; \
    int64_t s2LoopCount; /* s2循环当前的循环index */ \
    int64_t s2LoopLimit; \
    int64_t s1oIdx = 0; /* s1轴的index */ \
    int64_t boIdx = 0;  /* b轴的index */ \
    int64_t n2oIdx = 0; /* n2轴的index */ \
    int64_t goIdx = 0;  /* g轴的index */ \
    int64_t multiCoreInnerIdx = 0; \
    int64_t attentionOutOffset; \
    int64_t s2SplitIdx = 0; \
    int64_t qSNumInOneBlock; \
    int64_t oriKvLoopEndIdx; \
    int64_t firstFdDataWorkspaceIdx = 0; \
    int64_t reduceBlockId = 0; \
    int32_t mRealSize; \
    int32_t halfMRealSize; \
    int32_t firstHalfMRealSize; \
    int32_t s2RealSize; /* s2方向基本块的真实长度 */ \
    int32_t vec2MBaseSize; \
    int32_t actualS1Size; /* 非TND场景=总s1Size, Tnd场景下当前batch对应的s1 */ \
    uint8_t taskIdMod2; \
    uint8_t taskIdMod3; \
    uint8_t multiCoreIdxMod2 = 0; \
    bool isCmp; \
    bool isCrossCoreSplit = false; \
    bool isFirstS2SplitCore = true; \
    bool isFirstBase = true; \
    bool isLastBase = true; \
    bool needReduce = false

#define RUN_INFO_TOPK_LSE_FIELDS \
    /* lse 输出offset, 仅在非HIGH_PERF模式存在 */ \
    int64_t softmaxLseOffset

template <bool HIGH_PERF = false>
struct RunInfo {
    RUN_INFO_COMMON_FIELDS;
    RUN_INFO_TOPK_LSE_FIELDS;
};

template <>
struct RunInfo<true> { // HIGH_PERF: 无topk len且无lse场景, 剔除topk/lse相关字段
    RUN_INFO_COMMON_FIELDS;
};

#define CONST_INFO_COMMON_FIELDS \
    /* 全局的基本块信息 */ \
    uint32_t bSize; \
    uint32_t needInit; \
    uint32_t s1BaseSize; \
    uint32_t s2BaseSize; \
    int64_t dSize;       /* query d 512 */ \
    int64_t dSizeV;      /* key d 512 */ \
    int64_t dSizeVInput; /* key input Dsize 608 = rope + nope + scale + pad */ \
    int64_t dSizeNope;   /* key nope d 448 */ \
    int64_t dSizeRope;   /* key rope d 64 */ \
    int64_t gSize;       /* g轴的大小 */ \
    int64_t n2Size; \
    int64_t s1Size; /* s1总大小 */ \
    int64_t s2Size; /* s2总大小 */ \
    int64_t cmpS2Size; \
    /* 轴的乘积 */ \
    int64_t gDv; \
    int64_t n2G; \
    int64_t n2GDv; \
    /* matmul跳读参数 */ \
    int64_t mm1Ka; \
    /* dq 或者attentionOut的Stride */ \
    int64_t attentionOutStride; \
    uint32_t aivIdx; \
    uint32_t oriSparseBlockCount; \
    uint32_t cmpSparseBlockCount; \
    uint32_t alignedOriSparseBlockCount; \
    uint32_t alignedCmpSparseBlockCount; \
    /* service mm1 mm2 pageAttention */ \
    uint32_t oriBlockSize; \
    uint32_t cmpBlockSize; \
    uint32_t oriMaxBlockNumPerBatch; \
    uint32_t cmpMaxBlockNumPerBatch; \
    int32_t oriWinLeft; \
    int32_t oriWinRight; \
    uint32_t sparseBlockSize; \
    uint32_t cmpRatio; \
    float softmaxScale; \
    uint32_t oriKvStride; \
    uint32_t cmpKvStride; \
    uint32_t oriMaskMode; \
    uint32_t cmpMaskMode; \
    uint8_t subBlockIdx

#define CONST_INFO_TOPK_LSE_FIELDS \
    /* topk len/lse 相关参数, 仅在非HIGH_PERF模式存在 */ \
    bool hasOriTopkLength; \
    bool hasCmpTopkLength; \
    bool isSoftmaxLseEnable

template <bool HIGH_PERF = false>
struct ConstInfo {
    CONST_INFO_COMMON_FIELDS;
    CONST_INFO_TOPK_LSE_FIELDS;
};

template <>
struct ConstInfo<true> { // HIGH_PERF: 无topk len且无lse场景, 剔除topk/lse相关字段
    CONST_INFO_COMMON_FIELDS;
};
} // namespace regbaseutil

#endif // SMLA_UTIL_REGBASE_H
