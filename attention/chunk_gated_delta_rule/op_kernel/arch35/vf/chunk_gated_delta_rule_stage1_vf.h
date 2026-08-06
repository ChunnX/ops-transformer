/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHUNK_GATED_DELTA_RULE_STAGE1_VF_H
#define CHUNK_GATED_DELTA_RULE_STAGE1_VF_H

#include "kernel_operator.h"

namespace ChunkGatedDeltaRule {
using namespace AscendC;

/*!
 * \brief InverseAIVVF: 使用前代法(forward substitution)对 N×N 下三角单位对角阵求逆。
 *
 * 数学原理：
 *   attn 为 N×N 下三角矩阵（对角线为 1，即单位下三角阵），其逆矩阵 inv 同样为下三角。
 *   前代法逐行求解：
 *     inv[0]   = e_0
 *     inv[i]   = e_i - sum_{j=0}^{i-1} attn[i,j] * inv[j]    (i = 1, ..., N-1)
 *   其中 e_i 为单位矩阵的第 i 行（仅第 i 个元素为 1，其余为 0）。
 *
 * 数据布局（UB buffer，float 类型，行主序）：
 *   attnUb  : [halfChunkSize 行, chunkSize 列]，subBlock 列偏移 = offset
 *             元素 (i, col) 地址 = base + offset + i * chunkSize + col
 *             仅前 N 列有效（对角块边长 N）。
 *   invResUb: 与 attnUb 相同布局，存储逆矩阵结果。
 *   eiUb    : 单位矩阵，第 i 行 = e_i，行 stride = chunkSize（无列偏移）。
 *             eiUb 在 InverseCompute 调用前由调用方一次性初始化为 N×N 单位阵。
 *
 * 实现说明：
 *   - 使用 __aicore__ inline + __VEC_SCOPE__：前代法循环内存在 UB store-load 依赖
 *     （inv[i] 的计算依赖 inv[0..i-1]，后者已写入 UB），须由 LocalMemBar 保证一致性。
 *     __simd_vf__ 的 VF 调度器会破坏此跨迭代 store-load 依赖，故不可使用。
 *   - 每行计算流程：加载 attn 第 i 行 → 内层循环 Gather attn[i,j] 并累加 inv[j]*attn[i,j]
 *     → 加载 e_i → 减法得到 inv[i] → 写回 UB → LocalMemBar 保证后续行可读到最新值。
 *   - maskN 限制有效 lane 数为 N（对角块边长），超出部分不参与计算。
 *
 * \tparam N           对角块边长（INVERSE_SHAPE = 32）
 * \param attnUb       attn 矩阵 UB 基址（含 subBlock 列偏移由 offset 指定）
 * \param invResUb     逆矩阵结果 UB 基址
 * \param eiUb         预初始化的单位矩阵 UB 基址
 * \param offset       subBlock 列偏移
 * \param chunkSize    行 stride（当前固定为 64）
 */
template <uint32_t N>
__aicore__ inline void InverseAIVVF(__ubuf__ float *attnUb, __ubuf__ float *invResUb,
                                   __ubuf__ float *eiUb, uint32_t offset, uint32_t chunkSize)
{
    __VEC_SCOPE__
    {
        // maskN：仅前 N 个 lane 有效，屏蔽对角块以外的列
        uint32_t maskLen = N;
        MicroAPI::MaskReg maskN = MicroAPI::UpdateMask<float>(maskLen);

        // 第 0 行：inv[0] = e_0，直接从预初始化单位阵加载
        MicroAPI::RegTensor<float> inv0;
        MicroAPI::LoadAlign(inv0, eiUb);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(invResUb + offset, inv0, maskN);
        // 写入 inv[0] 后须屏障，保证后续行计算时能从 UB 读到正确的 inv[0]
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

        MicroAPI::RegTensor<uint32_t> idxReg;
        MicroAPI::RegTensor<float> acc;
        for (uint16_t i = 1; i < static_cast<uint16_t>(N); ++i) {
            // acc = sum_{j=0}^{i-1} attn[i,j] * inv[j]
            MicroAPI::Duplicate(acc, static_cast<float>(0.0));
            MicroAPI::RegTensor<float> li;
            MicroAPI::LoadAlign(li, attnUb + offset + i * chunkSize); // 加载 attn 第 i 行
            for (uint16_t j = 0; j < i; ++j) {
                MicroAPI::Duplicate(idxReg, j);                        // 将标量 j 广播为 gather 索引
                MicroAPI::RegTensor<float> lijBrc;
                MicroAPI::Gather(lijBrc, li, idxReg);                  // 从 attn 第 i 行取出第 j 个元素并广播
                MicroAPI::RegTensor<float> invj;
                MicroAPI::LoadAlign(invj, invResUb + offset + j * chunkSize); // 从 UB 读取已算出的 inv[j]
                MicroAPI::MulAddDst(acc, invj, lijBrc, maskN);         // acc += inv[j] * attn[i,j]
            }
            // inv[i] = e_i - acc
            MicroAPI::RegTensor<float> ei_i;
            MicroAPI::LoadAlign(ei_i, eiUb + i * chunkSize);           // 从单位阵加载 e_i
            MicroAPI::RegTensor<float> invi;
            MicroAPI::Sub(invi, ei_i, acc, maskN);                     // inv[i] = e_i - acc
            MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(invResUb + offset + i * chunkSize,
                                                                             invi, maskN);
            // 写入 inv[i] 后屏障，保证后续行计算时能从 UB 读到正确的 inv[i]
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        }
    }
}
} // namespace ChunkGatedDeltaRule
#endif // CHUNK_GATED_DELTA_RULE_STAGE1_VF_H
