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
 * \file moe_tutel_combine_gates_tiling_struct.h
 * \brief MoeTutelCombineGates TilingData definition.
 */

#ifndef MOE_TUTEL_COMBINE_GATES_TILING_STRUCT_H_
#define MOE_TUTEL_COMBINE_GATES_TILING_STRUCT_H_

#include <cstdint>

// ---------------------------------------------------------------------------
// MoeTutelCombineGatesTilingData — global TilingData (all fields in elements)
//
// Fields (units = number of elements; no field counts cores — the launched core
// count travels through SetBlockDim/blockDim, and taskNumPerBlock already folds
// the AIV core count in):
//   batchSize       — B = y_grad.shape[0]; also dim 0 of indices / locations /
//                     gates_grad. Consumed in kernel Init() (taskNum = samples * B)
//                     and Process() (validity bound of indices).
//   taskNumPerBlock — number of tasks handled by one core
//                     = CeilAlign(CeilDiv(taskNum, aivCoreNum), OFFSET=256),
//                     taskNum = samples * batchSize. Consumed in kernel Init():
//                     startOffset = blockIdx * taskNumPerBlock,
//                     endOffset   = min((blockIdx + 1) * taskNumPerBlock, taskNum).
//   samples         — S = x.shape[0]; also the last dim of indices / locations /
//                     gates_grad. Consumed in Init() (taskNum, x global buffer
//                     size = samples * hidden) and Process() (token row
//                     (taskOffset + i) % samples).
//   hidden          — H = x.shape[1] = y_grad.shape[2], the reduction axis
//                     length. Consumed in Init() (GM sizes, UB buffer sizes) and
//                     Process() (CopyIn length, Mul/ReduceSum count).
//   capacity        — C = y_grad.shape[1] (no attribute: implied by shape); upper
//                     bound of the illegal-routing test locations[t] >= capacity.
//                     Consumed in Init() (y_grad global buffer size =
//                     batchSize * capacity * hidden) and Process()
//                     (slot = indices[t] * capacity + locations[t]).
//   ubRowElems      — element count of ONE input-dtype H-long UB route (xBuf_ and
//                     yBuf_, the two routes holding T = bf16 / fp16 / fp32 rows).
//                     Already aligned by the host to ONE WHOLE VECTOR REGISTER in
//                     elements of the route's own dtype —
//                     CeilAlign(max(hidden, 1), max(blockSize, vectorSize) / sizeof(T))
//                     which on Ascend950PR (dav_3510: blockSize = 32 B,
//                     vectorSize = GetVecLen() = 256 B) is CeilAlign(h, 128) on the
//                     16b branches (bf16 / fp16; 256 / 2) and CeilAlign(h, 64) on the
//                     fp32 branch (256 / 4) — and NEVER below 1 element, so the kernel
//                     passes it straight into TPipe::InitBuffer — the kernel derives no
//                     UB size of its own (TilingData.md §3.2, P-08). Consumed in
//                     Init() only; it is a capacity, never a count (Kernel.md §9.3).
//   ubFp32Elems     — element count of ONE fp32 H-long UB route. The 16b branches
//                     size their four fp32 routes with it (xCastBuf_ / yCastBuf_ /
//                     mulBuf_ / workBuf_); the fp32 branch sizes its two (mulBuf_ /
//                     workBuf_). Aligned exactly like ubRowElems — one whole vector
//                     register in fp32 elements,
//                     CeilAlign(max(hidden, 1), max(blockSize, vectorSize) / 4) —
//                     which is CeilAlign(h, 64) on BOTH branches on dav_3510. That
//                     granularity also SUBSUMES the compensated reduction's former
//                     explicit `>= one vector register` floor (V = GetVecLen() / 4 = 64
//                     fp32 elements): S6/S7's partial tail chunk loads one whole
//                     V-wide aligned window per carrier, and CeilAlign(h, V) >= V for
//                     every h >= 1. Consumed in Init() (route sizes) and as the
//                     reduction VF's bufElems window bound — the same value on the host
//                     side is what CalcUbNeedBytes() charges, so budget and allocation
//                     cannot disagree (HostTiling.md §4.3).
//
//   WHY THE GRANULARITY IS max(blockSize, vectorSize) AND NOT blockSize (R4-6/R4-7):
//   on dav_3510 the aligned vector load form carries NO mask
//   (Reg::LoadAlign(vreg, addr) — the call sites are in the 16b/32b kernel headers),
//   so a consumer of an H-long route touches exactly one whole vector register per
//   repeat: window = vectorSize bytes = vectorSize / sizeof(dtype) elements. The host
//   must therefore guarantee ceil(hidden / W) * W <= routeElems for EVERY legal
//   hidden, which holds iff routeElems is hidden CeilAlign'ed to W = vectorSize /
//   sizeof(dtype) — hence the two grids above (blockSize / sizeof(dtype) is a divisor
//   of both: 128 % 16 == 0, 64 % 8 == 0, 64 % 64 == 0, so the UB allocation
//   block-size requirement is implied by the alignment).
//
//   THE SAME RULE APPLIES TO THE SMALL (innerOffset-sized) ROUTES the vector engine
//   READS: every route whose capacity is read by a whole-register vector load must be
//   at least max(INNER_OFFSET * sizeof(dtype), vectorSize) bytes — host side that is
//   CeilAlign(INNER_OFFSET * sizeof(dtype), max(blockSize, vectorSize)), i.e. 256 B on
//   dav_3510. It covers sumBuf_ on both branches (source of the 16b S10 narrowing Cast
//   and of the fp32 path's UB -> UB DataCopy) and zTmpBuf_ on the 16b branch (staging
//   for the MTE3 write-back). idxBuf_ / locBuf_ / tmpBuf_ / outBuf_ are only ever
//   scalar-read or mask-stored, so they keep the plain blockSize alignment.
//
//   Consequence for the contract: the host's H upper bound is 12608 on the 16b
//   branches and 15808 on the fp32 branch (pre-R4-6/R4-7 values were 12680 / 15848).
//
// Every other kernel-side quantity is derived inside the kernel from the first five
// (taskNum = (int64_t)samples * batchSize; innerOffset = 16 compile-time constant;
// innerTask = min(innerOffset, taskNum - taskOffset); row stride = hidden), so no
// further field exists — see TilingData.md §2/§3.2.
// The two UB-size fields are the ONE deliberate exception: they carry quantities the
// host already computes for its budget, so exposing them is what removes the second,
// independently maintained sizing formula from the kernel.
// ---------------------------------------------------------------------------
struct MoeTutelCombineGatesTilingData {
    int32_t batchSize;       // B: y_grad.shape[0]                      [elements]
    int32_t taskNumPerBlock; // tasks per core = CeilAlign(CeilDiv(taskNum, aivCoreNum), 256)
    int32_t samples;         // S: x.shape[0]                           [elements]
    int32_t hidden;          // H: reduction axis length = x.shape[1]   [elements]
    int32_t capacity;        // C: y_grad.shape[1]                      [elements]
    int32_t ubRowElems;      // elems of one input-dtype H-long UB route (xBuf_, yBuf_);
                             // CeilAlign(max(H,1), max(blockSize,vectorSize)/sizeof(T))
                             // = CeilAlign(H,128) on 16b, CeilAlign(H,64) on fp32 (dav_3510)
    int32_t ubFp32Elems;     // elems of one fp32 H-long UB route; CeilAlign(max(H,1),
                             // max(blockSize,vectorSize)/4) = CeilAlign(H,64) both branches
                             // (>= one vector register) on dav_3510
};

#endif // MOE_TUTEL_COMBINE_GATES_TILING_STRUCT_H_
