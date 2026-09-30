/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GLI_KERNEL_TILING_DATA_STRUCT_H
#define GLI_KERNEL_TILING_DATA_STRUCT_H

// Kernel 侧 tiling 布局必须与 op_host BEGIN_TILING_DATA_DEF 字段顺序、类型一致。
// 设备 opc -include 的 header 只有 GET_TILING_DATA 宏；kernel UT 框架会生成带 class 的同名类型。
#if !defined(__CCE_KT_TEST__) && !defined(__GENERICLIGHTNINGINDEXERTILINGDATA_HEADER__) && \
    !defined(GENERIC_LIGHTNING_INDEXER_TILING_DATA_DEFINED)
#define GENERIC_LIGHTNING_INDEXER_TILING_DATA_DEFINED
#pragma pack(push, 8)
struct GenericLightningIndexerTilingData {
    uint32_t bSize;
    uint32_t gSize;
    uint32_t s1Size;
    int64_t s2Size;
    uint32_t sparseCount;
    uint32_t topk;
    uint32_t poolSize;
    uint32_t usedCoreNum;
    uint32_t s1BaseSize;
    uint32_t mBaseSize;
    uint32_t mBaseSizeMax;
    uint32_t s2BaseSize;
    uint32_t trunkLen;
    uint32_t maskMode;
    uint32_t quantMode;
    uint32_t returnValue;
    uint32_t layoutQ;
    uint32_t layoutK;
    uint32_t blockSize;
    uint32_t maxBlockNumPerBatch;
    uint32_t keyStride0;
    uint32_t keyDequantScaleStride0;
    uint64_t wsOffScore;
    uint64_t wsOffLdScore;
    uint64_t wsOffLdIdx;
    float qkScale;
    uint32_t expandMode;
    uint32_t hasWeightedSum;
    uint32_t actType;
    uint32_t selectMode;
    uint32_t keyReduceMode;
    uint32_t initBlocks;
    uint32_t localBlocks;
    uint32_t hasOutputIdxOffset;
    int32_t maxSeqlenQ;
    uint32_t seqQCuFormat;
    uint32_t seqKCuFormat;
    uint32_t seqDtypeFlags;
    uint32_t msaScoreNT;
    uint64_t wsOffKCast;
};
#pragma pack(pop)
#endif

#endif // GLI_KERNEL_TILING_DATA_STRUCT_H
