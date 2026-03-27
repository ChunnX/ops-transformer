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
 * \file fused_causal_conv1d_cut_bh_struct.h
 * \brief FusedCausalConv1dCutBH tiling struct
 */

#ifndef FUSED_CAUSAL_CONV1D_CUT_BH_STRUCT_H
#define FUSED_CAUSAL_CONV1D_CUT_BH_STRUCT_H


struct FusedCausalConv1dCutBHTilingData{
// Core distribution parameters
int64_t usedCoreNum;              // Total used core number
int64_t dimCoreCnt;               // Number of cores for dim direction
int64_t batchCoreCnt;             // Number of cores for batch direction

// Dim tiling parameters inter-core (non-uniform split)
int64_t dimMainCoreCnt;           // Number of big dim cores (base+1 blocks of 128)
int64_t dimTailCoreCnt;           // Number of small dim cores (base blocks of 128)
int64_t mainCoredimLen;             // Big core dim size: (base+1) * 128
int64_t tailCoredimLen;              // Small core dim size: base * 128

// Batch tiling parameters inter-core (non-uniform split)
int64_t batchMainCoreCnt;         // Number of big batch cores
int64_t batchTailCoreCnt;         // Number of small batch cores
int64_t mainCoreBatchNum;             // Batch size for big cores
int64_t tailCoreBatchNum;         // Batch size for small cores

// Intra-core tiling parameters UB loop
int64_t loopNumBS;                // Loops in BS direction for big cores
int64_t loopNumDim;               // Loops in Dim direction for big cores
int64_t ubMainFactorBS;               // UB BS factor for big cores
int64_t ubTailFactorBS;           // UB BS tail factor for big cores
int64_t ubMainFactorDim;              // UB Dim factor for big cores
int64_t ubTailFactorDim;          // UB Dim tail factor for big cores
int64_t tailBlockloopNumBS;       // Loops in BS direction for tail cores
int64_t tailBlockloopNumDim;      // Loops in Dim direction for tail cores
int64_t tailBlockubFactorBS;      // UB BS factor for tail cores
int64_t tailBlockubTailFactorBS;  // UB BS tail factor for tail cores
int64_t tailBlockubFactorDim;     // UB Dim factor for tail cores
int64_t tailBlockubTailFactorDim; // UB Dim tail factor for tail cores

// Shape information for kernel use
int64_t batchSize;                // Batch size
int64_t seqLen;                   // Sequence length for 3D input
int64_t cuSeqLen;                 // Cumulative sequence length for 2D input
int64_t dim;                      // Dimension size
int64_t kernelSize;               // Kernel size K
int64_t stateLen;                 // State length: second dimension of cacheState (K-1+m)
int64_t xStride;                  // Stride for x tensor - seqLen
int64_t cacheStride0;             // Stride for cacheState tensor - batch
int64_t cacheStride1;             // Stride for cacheState tensor - stateLen
int64_t padSlotId;               // padding batch which will not be calculated
int64_t xInputMode;               // Input mode: 0 for 3D 1 for 2D
int64_t hasAcceptTokenNum;        // Whether acceptTokenNum input is provided: 0 for false, 1 for true
int64_t residualConnection;       // Whether use residual connection: 0 for false, 1 for true
};

#endif
