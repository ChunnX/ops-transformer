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
    int64_t qsfaBoIdx; \
    int64_t qsfaS1oIdx; \
    int64_t qsfaN2oIdx; \
    int64_t qsfaGoIdx; \
    int64_t qsfaS2LoopEndIdx;          /* S2方向的循环控制信息 souter层确定 */ \
    int64_t qsfaS2LineStartIdx = 0;    /* S2方向按行的起始位置 */ \
    int64_t qsfaS2LineEndIdx;          /* S2方向按行的结束位置 */ \
    int64_t qsfaS2CmpLineEndIdx; \
    /* cube视角的sOuter，在SAMEAB场景中cubeSOuterSize为两倍的 qsfaHalfS1RealSize souter层确定 */ \
    uint32_t qsfaS1RealSize; \
    uint32_t qsfaHalfS1RealSize; \
    uint32_t qsfaFirstHalfS1RealSize; \
    uint32_t qsfaMRealSize; \
    uint32_t qsfaHalfMRealSize; \
    uint32_t qsfaFirstHalfMRealSize; \
    int64_t qsfaAttentionOutOffset;    /* attentionOut的offset souter层确定 */ \
    int32_t qsfaActualS1Size;      /* Q的actualSeqLength */ \
    int32_t qsfaActualS2Size    /* KV的actualSeqLength */ \

struct RunParamStr {  // 分核与切块需要使用到参数
    COMMON_RUN_PARAM;
    /* 推理新增 */
    int64_t gs1LoopStartIdx;
    int64_t gs1LoopEndIdx;
    // BN循环生产的数据
    int64_t qsfaPreTokensPerBatch = MAX_PRE_NEXT_TOKENS; // 左上顶点的pretoken
    int64_t qsfaNextTokensPerBatch = MAX_PRE_NEXT_TOKENS; // 左上顶点的nexttoken

    // NBS1循环生产的数据
    int64_t qsfaSOuterOffset;               // 单个S内 souter的 souterIdx * qsfaHalfS1RealSize souter层确定
    int64_t qsfaCubeSOuterOffset;           // 单个S内 souter的 souterIdx * qsfaHalfS1RealSize souter层确定
    int64_t qsfaMOuterOffset;
    int64_t cubeMOuterOffset;

    // lse 输出offset
    int64_t softmaxLseOffset;       // souter层确定

    int64_t qSNumInOneBlock;
    int64_t kvLoopEndIdx;
};

#define COMMON_RUN_INFO \
    int64_t qsfaS2StartIdx; /* s2的起始位置，sparse场景下可能不是0 */ \
    int64_t qsfaS2EndIdx; \
    int64_t qsfaS2LoopCount; /* s2循环当前的循环index */ \
    int64_t qsfaS2LoopLimit; \
    int64_t qsfaS1oIdx = 0; /* s1轴的index */ \
    int64_t qsfaLoop = 0; /* for v0 perload qsfaLoop */ \
    int64_t qsfaBoIdx = 0; /* b轴的index */ \
    int64_t qsfaN2oIdx = 0; /* n2轴的index */ \
    int64_t qsfaGoIdx = 0; /* g轴的index */ \
    int32_t qsfaS1RealSize; \
    int32_t qsfaHalfS1RealSize; /* vector侧实际的s1基本块大小，如果Cube基本块=128，那么halfS1RealSize=64 */ \
    int32_t qsfaFirstHalfS1RealSize; /* 当s1RealSize不是2的整数倍时，v0比v1少计算一行，计算subblock偏移的时候需要使用v0的s1 size */ \
    int32_t qsfaMRealSize; \
    int32_t qsfaHalfMRealSize; \
    int32_t qsfaFirstHalfMRealSize; \
    int32_t qsfaS2RealSize; /* s2方向基本块的真实长度 */ \
    int64_t qsfaS2AlignedSize; /* s2方向基本块对齐到16之后的长度 */ \
    int32_t qsfaVec2S1BaseSize; /* vector2侧开循环之后，经过切分的S1大小，例如把64切分成两份32 */ \
    int32_t qsfaVec2S1RealSize; /* vector2侧开循环之后，经过切分的S1的尾块大小，例如把63切分成两份32和31，第二份的实际大小是31 */ \
    int32_t qsfaVec2MBaseSize; \
    int32_t qsfaVec2MRealSize; \
    int64_t qsfaTaskId; \
    int64_t qsfaMultiCoreInnerIdx = 0; \
    int64_t qsfaAttentionOutOffset; \
    int32_t qsfaActualS1Size; /* 非TND场景=总s1Size, Tnd场景下当前batch对应的s1 */ \
    int32_t qsfaActualS2Size; /* 非TND场景=总s2Size, Tnd场景下当前batch对应的s2 */ \
    int64_t qsfaPreTokensPerBatch; /* vector2 左上顶点的pretoken */ \
    int64_t qsfaNextTokensPerBatch; /* vector2 左上顶点的nexttoken */ \
    uint8_t qsfaTaskIdMod2; \
    uint8_t qsfaTaskIdMod3; \
    uint8_t qsfaMultiCoreIdxMod2 = 0; \
    uint8_t qsfaMultiCoreIdxMod3 = 0; \
    int64_t qsfaSOuterOffset; \
    int64_t qsfaMOuterOffset

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
    uint32_t qsfaBSize; \
    uint32_t qsfaNeedInit; \
    uint32_t qsfaS1BaseSize; \
    uint32_t qsfaS2BaseSize; \
    int64_t qsfaDSize; /* query d 512 */ \
    int64_t qsfaDSizeV; /* key d 512 */ \
    int64_t qsfaDSizeVInput; /* key inpue d 656 = rope + nope + scale + pad */ \
    int64_t qsfaDSizeNope; /* key nope d 448 */ \
    int64_t qsfaDSizeRope; /* key rope d 64 */ \
    int64_t qsfaTileSize; /* 64 */ \
    int64_t qsfaSparseMode; \
    int64_t qsfaGSize; /* g轴的大小 */ \
    int64_t qsfaN2Size; \
    int64_t qsfaS1Size; /* s1总大小 */ \
    int64_t qsfaS2Size; /* s2总大小 */ \
    /* 轴的乘积 */ \
    int64_t qsfaS1D; \
    int64_t qsfaGS1D; \
    int64_t qsfaN2GS1D; \
    int64_t qsfaS2D; \
    int64_t qsfaN2S2D; \
    int64_t qsfaS1Dv; \
    int64_t qsfaGS1Dv; \
    int64_t qsfaN2GS1Dv; \
    int64_t qsfaS2Dv; \
    int64_t qsfaN2S2Dv; \
    int64_t qsfaS1S2; \
    int64_t qsfaGS1; \
    int64_t qsfaGD; \
    int64_t qsfaN2D; \
    int64_t qsfaBN2D; \
    int64_t qsfaGDv; \
    int64_t qsfaN2Dv; \
    int64_t qsfaBN2Dv; \
    int64_t qsfaN2G; \
    int64_t qsfaN2GD; \
    int64_t qsfaBN2GD; \
    int64_t qsfaN2GDv; \
    int64_t qsfaBN2GDv; \
    int64_t qsfaGS2; \
    int64_t qsfaS1Dr; \
    int64_t qsfaGS1Dr; \
    int64_t qsfaN2GS1Dr; \
    int64_t qsfaS2Dr; \
    int64_t qsfaN2S2Dr; \
    int64_t qsfaGDr; \
    int64_t qsfaN2Dr; \
    int64_t qsfaBN2Dr; \
    int64_t qsfaN2GDr; \
    int64_t qsfaBN2GDr; \
    int32_t qsfaS2BaseN2D; \
    int32_t qsfaS1BaseN2GD; \
    int64_t qsfaS2BaseBN2D; \
    int64_t qsfaS1BaseBN2GD; \
    int32_t qsfaS1BaseD; \
    int32_t qsfaS2BaseD; \
    int64_t qsfaS2BaseN2Dv; \
    int64_t qsfaS2BaseBN2Dv; \
    int64_t qsfaS1BaseN2GDv; \
    int64_t qsfaS1BaseBN2GDv; \
    int32_t qsfaS1BaseDv; \
    int32_t qsfaS2BaseDv; \
    /* matmul跳读参数 */ \
    int64_t qsfaMm1Ka; \
    /* dq 或者attentionOut的Stride */ \
    int64_t qsfaAttentionOutStride; \
    uint32_t qsfaAivIdx; \
    uint8_t qsfaLayoutType; \
    uint8_t qsfaSubBlockIdx; \
    /* 分核相关 */ \
    uint32_t qsfaS2Start; \
    uint32_t qsfaS2End; \
    uint32_t qsfaBN2Start; \
    uint32_t qsfaBN2End; \
    uint32_t qsfaGS1Start; \
    uint32_t qsfaGS1End

#define INFER_CONST_INFO \
    /* 推理 */ \
    bool qsfaIsActualLenDimsNull; /* 判断是否有actualseq */ \
    bool qsfaIsActualLenDimsKVNull; /* 判断是否有actualseq_kv */ \
    bool qsfaIsSoftmaxLseEnable; \
    bool qsfaRsvd1; \
    uint32_t qsfaSparseBlockCount; \
    uint32_t qsfaActualSeqLenSize; /* 用户输入的actualseq的长度 */ \
    uint32_t qsfaActualSeqLenKVSize; /* 用户输入的actualseq_kv的长度 */ \
    /* service mm1 mm2 pageAttention */ \
    uint32_t qsfaBlockSize; \
    uint32_t qsfaPaLayoutType; \
    uint32_t qsfaMaxBlockNumPerBatch; \
    uint32_t qsfaSparseBlockSize; \
    float qsfaSoftmaxScale

#define CV_SHARED_PARAMS \
    /* base params */ \
    uint32_t qsfaS1BaseSize; \
    uint32_t qsfaS2BaseSize; \
    uint32_t qsfaBSize;  \
    uint32_t qsfaN2Size;  \
    uint32_t qsfaGSize;  \
    uint32_t qsfaS1Size;  \
    uint32_t qsfaS2Size;  \
    uint32_t qsfaDSize : 10;  \
    int64_t qsfaDSizeVInput : 12;  \
    uint32_t qsfaNeedInit : 4; \
    uint32_t qsfaLayoutType : 4;  \
    uint32_t qsfaIsActualSeqLengthsNull : 1; \
    uint32_t qsfaIsActualSeqLengthsKVNull : 1; \
    uint32_t qsfaSparseBlockCount; \
    float qsfaSoftmaxScale; \
    uint32_t qsfaDSizeRope : 11; \
    uint32_t qsfaMaskMode : 6; \
    uint32_t qsfaTileSize : 8; \
    /* pa params */  \
    uint32_t qsfaBlockSize : 12; \
    uint32_t qsfaMaxBlockNumPerBatch; \
    uint32_t qsfaUsedCoreNum


struct ConstInfo {
    COMMON_CONST_INFO;
    INFER_CONST_INFO;
};

/* only support b32 or b64 */
struct CVSharedParams {
    CV_SHARED_PARAMS;
};
}

#endif // KV_QUANT_SAS_UTIL_REGBASE_H
