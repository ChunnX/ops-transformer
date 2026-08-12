/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHUNK_GATED_DELTA_RULE_STAGE1_VF_H
#define CHUNK_GATED_DELTA_RULE_STAGE1_VF_H

#include "kernel_tensor.h"

namespace ChunkGatedDeltaRule {
using namespace AscendC;
using namespace MicroAPI;

/*!
 * InverseAIVVF: forward-substitution inverse of the N x N lower-triangular (unit-diagonal) attn block.
 *   inv[0] = e_0; for i=1..N-1: inv[i] = e_i - sum_{j<i} attn[i,j] * inv[j]
 * attnUb/invResUb layout: [halfChunkSize rows, chunkSize cols]; subBlock uses column offset = `offset`,
 *   i.e. element (i, col) is at base + offset + i*chunkSize + col. Only first N columns per row are valid.
 * e_i is generated in registers via UpdateMask + MaskXor + Duplicate<MERGING>, eliminating the need
 *   for a pre-filled identity buffer (eiUb) and the associated V_S/S_V cross-pipe synchronization.
 */
template <uint32_t N>
__simd_vf__ inline void InverseAIVVFImpl(__ubuf__ float *attnUb, __ubuf__ float *invResUb,
                                         uint32_t offset, uint32_t chunkSize)
{
    // N<=64, 单寄存器容纳一行; UpdateMask 生成前 N 个 lane 有效的掩码
    uint32_t maskLen = N;
    MaskReg maskN = UpdateMask<float>(maskLen);

    // 单 lane 掩码生成: maskLaneI 每轮只选中 lane i
    // i=0: CreateMask<VL1> 直接得到 lane 0 的掩码
    // i>0: UpdateMask(i+1) XOR UpdateMask(i) 得到 lane i 的掩码
    MaskReg maskLaneI = CreateMask<float, MaskPattern::VL1>();
    MaskReg maskPrev;
    MaskMov(maskPrev, maskLaneI);

    // inv[0] = e_0: 全零, lane 0 置 1.0
    RegTensor<float> inv0;
    Duplicate(inv0, static_cast<float>(0.0));
    Duplicate<float, MaskMergeMode::MERGING>(inv0, static_cast<float>(1.0), maskLaneI);
    StoreAlign<float, StoreDist::DIST_NORM_B32>(invResUb + offset, inv0, maskN);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    // 前代法: 逐行求解 inv[i] = e_i - sum_{j<i} attn[i,j] * inv[j]
    RegTensor<uint32_t> idxReg;
    RegTensor<float> acc;
    for (uint16_t i = 1; i < static_cast<uint16_t>(N); ++i) {
        // acc 清零, 准备累加当前行 i 的内积
        Duplicate(acc, static_cast<float>(0.0));
        // 加载 attn 第 i 行(对角块内, 含 offset 列偏移)
        RegTensor<float> li;
        LoadAlign(li, attnUb + offset + i * chunkSize);
        // 累加 sum_{j<i} attn[i,j] * inv[j]: Gather 取出 li[j] 广播, 与已求出的 inv[j] 乘加
        for (uint16_t j = 0; j < i; ++j) {
            Duplicate(idxReg, j);
            RegTensor<float> lijBrc;
            Gather(lijBrc, li, idxReg);
            RegTensor<float> invj;
            LoadAlign(invj, invResUb + offset + j * chunkSize);
            MulAddDst(acc, invj, lijBrc, maskN);
        }
        // 生成 lane i 的单 lane 掩码: maskCur(0..i) XOR maskPrev(0..i-1) = lane i
        uint32_t curLen = static_cast<uint32_t>(i) + 1;
        MaskReg maskCur = UpdateMask<float>(curLen);
        MaskXor(maskLaneI, maskCur, maskPrev, maskN);
        MaskMov(maskPrev, maskCur);
        // 生成 e_i: 全零, lane i 置 1.0
        RegTensor<float> ei_i;
        Duplicate(ei_i, static_cast<float>(0.0));
        Duplicate<float, MaskMergeMode::MERGING>(ei_i, static_cast<float>(1.0), maskLaneI);
        // inv[i] = e_i - acc, 写回结果第 i 行
        RegTensor<float> invi;
        Sub(invi, ei_i, acc, maskN);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(invResUb + offset + i * chunkSize, invi, maskN);
        // store->load 屏障: 保证本行写入对后续行加载可见(下一轮 i 需读取 inv[i])
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
}

template <uint32_t N>
__aicore__ inline void InverseAIVVF(const LocalTensor<float> &attnUb, const LocalTensor<float> &invResUb,
                                    uint32_t offset, uint32_t chunkSize)
{
    __ubuf__ float *attn = reinterpret_cast<__ubuf__ float *>(attnUb.GetPhyAddr());
    __ubuf__ float *inv = reinterpret_cast<__ubuf__ float *>(invResUb.GetPhyAddr());
    InverseAIVVFImpl<N>(attn, inv, offset, chunkSize);
}
} // namespace ChunkGatedDeltaRule
#endif // CHUNK_GATED_DELTA_RULE_STAGE1_VF_H
