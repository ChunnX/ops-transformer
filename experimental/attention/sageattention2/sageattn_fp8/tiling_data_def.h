/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTENTION_TILING_DATA_DEF_H
#define FLASH_ATTENTION_TILING_DATA_DEF_H

constexpr uint32_t MAX_CORE_NUM = 64;

class InputParamsRegbase {
public:
    int64_t batch;
    int64_t qHeads;
    int64_t kvHeads;
    int64_t groupSize;
    int64_t qSeqlen;
    int64_t kvSeqlen;
    int64_t embed;
    float scaleValue;
    uint8_t attenMaskCompressMode; // SPARSE_MODE_NO_MASK: 0, SPARSE_MODE_LEFT_UP: 1, SPARSE_MODE_RIGHT_DOWN : 2

    // ---- SageAttention2 量化参数（INT8 QK + FP8 PV）----
    uint8_t qkQuantGran;    // 0=per_warp, 1=per_thread（AscendC 无同名内置，自研分块映射）
    uint8_t pvAccumDtype;   // 0=fp32, 1=fp32+fp32, 2=fp32+fp16（默认）
    uint8_t smoothKFlag;    // 1 表示对 K 减均值（outlier smoothing）
    uint8_t smoothVFlag;    // 1 表示对 V 减均值（fp32+fp32/fp32+fp16 下被忽略）
    uint8_t returnLseFlag;  // 1 表示输出 softmax LSE [B, Hq, Sq]
    uint16_t headDimOg;     // 输出裁回 pad 前 head_dim
    uint32_t qScaleGroups;  // Q scale 组数（per_thread: ceil(Sq/32)*8；per_warp: ceil(Sq/32)）
    uint32_t kScaleGroups;  // K scale 组数（per_thread: ceil(Sk/64)*4；per_warp: ceil(Sk/64)）
    float vScaleMax;        // fp32+fp16=2.25，其余=448.0
    float quantEps;         // 1e-7（scale = max|x|/127 + eps）

    // PFA
    uint8_t isActualSeqLengthsNull;
    uint8_t isActualSeqLengthsKVNull;
    uint32_t actualSeqLengthsSize;
    uint32_t actualSeqLengthsKVSize;

    uint32_t headNumRatio;
    uint32_t blockSize;
    uint32_t blockTableDim2;
    uint32_t paBlockNumSum;
    uint32_t attenMaskQSeqlen;
    uint32_t attenMaskKvSeqlen;
};

class MultiCoreParamsRegbase {
public:
    int32_t coreNum;
    int64_t totalSize;
    int64_t qSeqlenOuterSize;
    int64_t splitFactorSize;
    int64_t splitFactorTailSize;
    uint32_t bnAxisStartIdx[MAX_CORE_NUM];
    int64_t sparseStartIdx[MAX_CORE_NUM];
};

class FATilingData {
public:
    InputParamsRegbase inputParamsRegbase;
    MultiCoreParamsRegbase multiCoreParamsRegbase;
};
#endif
