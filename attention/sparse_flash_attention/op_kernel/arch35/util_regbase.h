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

#ifndef UTIL_REGBASE_H
#define UTIL_REGBASE_H

#include "util.h"

using AscendC::TQue;
using AscendC::QuePosition;

namespace regbaseutil {
constexpr int64_t MAX_PRE_NEXT_TOKENS = 0x7FFFFFFF;

#define COMMON_RUN_PARAM \
    int64_t sfaBoIdx; \
    int64_t sfaS1oIdx; \
    int64_t sfaN2oIdx; \
    int64_t sfaGoIdx; \
    int64_t sfaS2LoopEndIdx;          /* S2方向的循环控制信息 souter层确定 */ \
    int64_t sfaS2LineStartIdx = 0;    /* S2方向按行的起始位置 */ \
    int64_t sfaS2LineEndIdx;          /* S2方向按行的结束位置 */ \
    int64_t sfaS2CmpLineEndIdx; \
    /* cube视角的sOuter，在SAMEAB场景中cubeSOuterSize为两倍的 sfaHalfS1RealSize souter层确定 */ \
    uint32_t sfaS1RealSize; \
    uint32_t sfaHalfS1RealSize; \
    uint32_t sfaFirstHalfS1RealSize; \
    uint32_t sfaMRealSize; \
    uint32_t sfaHalfMRealSize; \
    uint32_t sfaFirstHalfMRealSize; \
    int64_t sfaAttentionOutOffset;    /* attentionOut的offset souter层确定 */ \
    int32_t sfaActualS1Size;      /* Q的actualSeqLength */ \
    int32_t sfaActualS2Size;    /* KV的actualSeqLength */ \
    int64_t sfaTensorQOffset; \
    int64_t sfaTensorQRopeOffset; \
    int64_t sfaQBOffset; \
    int64_t sfaQRopeBOffset;

struct RunParamStr {  // 分核与切块需要使用到参数
    COMMON_RUN_PARAM;
    /* 推理新增 */
    int64_t gs1LoopStartIdx;
    int64_t gs1LoopEndIdx;
    // BN循环生产的数据
    int64_t sfaPreTokensPerBatch = MAX_PRE_NEXT_TOKENS; // 左上顶点的pretoken
    int64_t sfaNextTokensPerBatch = MAX_PRE_NEXT_TOKENS; // 左上顶点的nexttoken

    // NBS1循环生产的数据
    int64_t sfaSOuterOffset;               // 单个S内 souter的 souterIdx * sfaHalfS1RealSize souter层确定
    int64_t cubeSOuterOffset;           // 单个S内 souter的 souterIdx * sfaHalfS1RealSize souter层确定
    int64_t sfaMOuterOffset;
    int64_t cubeMOuterOffset;

    // lse 输出offset
    int64_t softmaxLseOffset;       // souter层确定

    int64_t qSNumInOneBlock;
    int64_t kvLoopEndIdx;
};

#define COMMON_RUN_INFO \
    int64_t sfaS2StartIdx; /* s2的起始位置，sparse场景下可能不是0 */ \
    int64_t sfaS2EndIdx; \
    int64_t sfaS2LoopCount; /* s2循环当前的循环index */ \
    int64_t sfaS2LoopLimit; \
    int64_t sfaS1oIdx = 0; /* s1轴的index */ \
    int64_t sfaLoop = 0; /* for v0 perload sfaLoop */ \
    int64_t sfaBoIdx = 0; /* b轴的index */ \
    int64_t sfaN2oIdx = 0; /* n2轴的index */ \
    int64_t sfaGoIdx = 0; /* g轴的index */ \
    int32_t sfaS1RealSize; \
    int32_t sfaHalfS1RealSize; /* vector侧实际的s1基本块大小，如果Cube基本块=128，那么halfS1RealSize=64 */ \
    int32_t sfaFirstHalfS1RealSize; /* 当s1RealSize不是2的整数倍时，v0比v1少计算一行，计算subblock偏移的时候需要使用v0的s1 size */ \
    int32_t sfaMRealSize; \
    int32_t sfaHalfMRealSize; \
    int32_t sfaFirstHalfMRealSize; \
    int32_t sfaS2RealSize; /* s2方向基本块的真实长度 */ \
    int64_t sfaS2AlignedSize; /* s2方向基本块对齐到16之后的长度 */ \
    int32_t sfaVec2S1BaseSize; /* vector2侧开循环之后，经过切分的S1大小，例如把64切分成两份32 */ \
    int32_t sfaVec2S1RealSize; /* vector2侧开循环之后，经过切分的S1的尾块大小，例如把63切分成两份32和31，第二份的实际大小是31 */ \
    int32_t sfaVec2MBaseSize; \
    int32_t sfaVec2MRealSize; \
    int64_t sfaTaskId; \
    int64_t sfaMultiCoreInnerIdx = 0; \
    int64_t sfaAttentionOutOffset; \
    int32_t sfaActualS1Size; /* 非TND场景=总s1Size, Tnd场景下当前batch对应的s1 */ \
    int32_t sfaActualS2Size; /* 非TND场景=总s2Size, Tnd场景下当前batch对应的s2 */ \
    int64_t sfaPreTokensPerBatch; /* vector2 左上顶点的pretoken */ \
    int64_t sfaNextTokensPerBatch; /* vector2 左上顶点的nexttoken */ \
    uint8_t sfaTaskIdMod2; \
    uint8_t sfaTaskIdMod3; \
    uint8_t sfaMultiCoreIdxMod2 = 0; \
    uint8_t sfaMultiCoreIdxMod3 = 0; \
    int64_t sfaSOuterOffset; \
    int64_t sfaMOuterOffset; \
    int64_t sfaQueryOffset; \
    int64_t sfaQueryRopeOffset

struct RunInfo {
    COMMON_RUN_INFO;
    // 推理新增
    // lse 输出offset
    int64_t softmaxLseOffset;

    int64_t qSNumInOneBlock;
    int64_t kvLoopEndIdx;
};

#define COMMON_CONST_INFO \
    /* 全局的基本块信息 */ \
    uint32_t sfaBSize; \
    uint32_t sfaNeedInit; \
    uint32_t sfaS1BaseSize; \
    uint32_t sfaS2BaseSize; \
    int64_t sfaDSize; /* query d 512 */ \
    int64_t sfaDSizeV; /* key d 512 */ \
    int64_t sfaDSizeVInput; /* key inpue d 656 = rope + nope + scale + pad */ \
    int64_t sfaDSizeNope; /* key nope d 448 */ \
    int64_t sfaDSizeRope; /* key rope d 64 */ \
    int64_t sfaTileSize; /* 64 */ \
    int64_t sfaSparseMode; \
    int64_t sfaGSize; /* g轴的大小 */ \
    int64_t sfaN2Size; \
    int64_t sfaS1Size; /* s1总大小 */ \
    int64_t sfaS2Size; /* s2总大小 */ \
    /* 轴的乘积 */ \
    int64_t sfaS1D; \
    int64_t sfaGS1D; \
    int64_t sfaN2GS1D; \
    int64_t sfaS2D; \
    int64_t sfaN2S2D; \
    int64_t sfaS1Dv; \
    int64_t sfaGS1Dv; \
    int64_t sfaN2GS1Dv; \
    int64_t sfaS2Dv; \
    int64_t sfaN2S2Dv; \
    int64_t sfaS1S2; \
    int64_t sfaGS1; \
    int64_t sfaGD; \
    int64_t sfaN2D; \
    int64_t sfaBN2D; \
    int64_t sfaGDv; \
    int64_t sfaN2Dv; \
    int64_t sfaBN2Dv; \
    int64_t sfaN2G; \
    int64_t sfaN2GD; \
    int64_t sfaBN2GD; \
    int64_t sfaN2GDv; \
    int64_t sfaBN2GDv; \
    int64_t sfaGS2; \
    int64_t sfaS1Dr; \
    int64_t sfaGS1Dr; \
    int64_t sfaN2GS1Dr; \
    int64_t sfaS2Dr; \
    int64_t sfaN2S2Dr; \
    int64_t sfaGDr; \
    int64_t sfaN2Dr; \
    int64_t sfaBN2Dr; \
    int64_t sfaN2GDr; \
    int64_t sfaBN2GDr; \
    int32_t sfaS2BaseN2D; \
    int32_t sfaS1BaseN2GD; \
    int64_t sfaS2BaseBN2D; \
    int64_t sfaS1BaseBN2GD; \
    int32_t sfaS1BaseD; \
    int32_t sfaS2BaseD; \
    int64_t sfaS2BaseN2Dv; \
    int64_t sfaS2BaseBN2Dv; \
    int64_t sfaS1BaseN2GDv; \
    int64_t sfaS1BaseBN2GDv; \
    int32_t sfaS1BaseDv; \
    int32_t sfaS2BaseDv; \
    bool sfaReturnSoftmaxLse; \
    /* matmul跳读参数 */ \
    int64_t sfaMm1Ka; \
    /* dq 或者attentionOut的Stride */ \
    int64_t sfaAttentionOutStride; \
    uint32_t sfaAicIdx; \
    uint32_t sfaAivIdx; \
    uint8_t sfaLayoutType; \
    uint8_t sfaSubBlockIdx; \
    /* 分核相关 */ \
    uint32_t sfaS2Start; \
    uint32_t sfaS2End; \
    uint32_t sfaBN2Start; \
    uint32_t sfaBN2End; \
    uint32_t sfaGS1Start; \
    uint32_t sfaGS1End

#define INFER_CONST_INFO \
    /* 推理 */ \
    bool sfaIsActualLenDimsNull; /* 判断是否有actualseq */ \
    bool sfaIsActualLenDimsKVNull; /* 判断是否有actualseq_kv */ \
    bool sfaIsSoftmaxLseEnable; \
    bool sfaRsvd1; \
    uint32_t sfaSparseBlockCount; \
    uint32_t sfaActualSeqLenSize; /* 用户输入的actualseq的长度 */ \
    uint32_t sfaActualSeqLenKVSize; /* 用户输入的actualseq_kv的长度 */ \
    /* service mm1 mm2 pageAttention */ \
    uint32_t sfaOriBlockSize; \
    uint32_t sfaCmpBlockSize; \
    uint32_t sfaPaLayoutType; \
    uint32_t sfaOriMaxBlockNumPerBatch; \
    uint32_t sfaCmpMaxBlockNumPerBatch; \
    int32_t sfaOriWinLeft; \
    int32_t sfaOriWinRight; \
    uint32_t sfaSparseBlockSize; \
    uint32_t sfaCmpRatio; \
    float sfaSoftmaxScale

#define CV_SHARED_PARAMS \
    /* base params */ \
    uint32_t sfaS1BaseSize; \
    uint32_t sfaS2BaseSize; \
    uint32_t sfaBSize;  \
    uint32_t sfaN2Size;  \
    uint32_t sfaGSize;  \
    uint32_t sfaS1Size;  \
    uint32_t sfaS2Size;  \
    uint32_t sfaDSize : 10;  \
    int64_t sfaDSizeVInput : 12;  \
    uint32_t sfaNeedInit : 4; \
    uint32_t sfaLayoutType : 4;  \
    uint32_t sfaIsActualSeqLengthsNull : 1; \
    uint32_t sfaIsActualSeqLengthsKVNull : 1; \
    uint32_t sfaSparseBlockCount; \
    float sfaSoftmaxScale; \
    uint32_t sfaCmpRatio : 9; \
    uint32_t sfaDSizeRope : 11; \
    uint32_t sfaOriMaskMode : 6; \
    uint32_t sfaCmpMaskMode : 6; \
    int32_t sfaOriWinLeft; \
    int32_t sfaOriWinRight; \
    uint32_t sfaTileSize : 8; \
    /* pa params */  \
    uint32_t sfaOriBlockSize : 12; \
    uint32_t sfaCmpBlockSize : 12; \
    uint32_t sfaOriMaxBlockNumPerBatch; \
    uint32_t sfaCmpMaxBlockNumPerBatch; \
    uint32_t sfaUsedCoreNum; \
    bool sfaReturnSoftmaxLse

struct ConstInfo {
    COMMON_CONST_INFO;
    INFER_CONST_INFO;
};

/* only support b32 or b64 */
struct CVSharedParams {
    CV_SHARED_PARAMS;
};
}

#endif // UTIL_REGBASE_H
