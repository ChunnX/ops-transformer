/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 * You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pool_key_indexer_vf_expand.h
 * \brief pool 索引 -> token 索引的向量化展开(arch35 AIV)。
 *
 * 展开(out[m] = ps * idx[m/ps] + m%ps)原为逐 pool 标量循环:
 * sc 次 GetValue + sc*ps 次 SetValue/行(poolSize 非 8 倍数时全标量,
 * 8 倍数时每轮 1 次 GetValue + 2 次小向量 + PipeBarrier)。
 * 本文件用 64 lane 寄存器 gather(vselr)方案替代:
 *   每组 LoadAlign 64 个池索引进寄存器, 组内 ps 个 64 lane 输出 chunk 用
 *   vselr 按相对索引选取(vselr 为寄存器选择指令, 单周期级; 实测 UB 表
 *   vgather2 单次 ~600 cycle, 不可用)。组内第 j 个 chunk 的选源索引 =
 *   j*64/ps + r/ps ∈ [0,64), 依赖 pow2 ps 下 (chunkBase+r)/ps 与
 *   chunkBase/ps + r/ps 的可分解性(ps|64 时整除成立)。
 *   约束: 仅支持 pow2 ps ∈ [2,64](ps|64 保证 chunkBase%ps==0,
 *   lane 偏移 r%ps 与 chunk 无关); ps=128 时 64 lane chunk 不跨满
 *   pool 窗口, 偏移需逐 chunk 修正(多一条 Adds 会触发寄存器溢出,
 *   实测性能翻倍劣化), 故 ps=128 由调用方走原有 Duplicate+Add 路径。
 * rIdx/rOff 两个 64 元素模板一次构建、每行经寄存器复用,
 * 每 64 输出 5 条向量指令(Adds/vselr/Mul/Add/StoreAlign),
 * 无标量循环、无逐轮 PipeBarrier、无 UB 随机访问。
 * 池索引表越界防护: LoadAlign 按 64 池对齐读, 尾组读到 [rounds, +64)
 * 的陈旧值仅落入 lane >= effExpand, 由调用方 -1 覆盖, 且读取偏移
 * 上界 rounds+63 < topkCountAlign256_+64(表容量), 恒在界内。
 */

#ifndef POOL_KEY_INDEXER_VF_EXPAND_H
#define POOL_KEY_INDEXER_VF_EXPAND_H

#include "kernel_operator.h"

namespace pkiexpand {
using namespace AscendC;

// pow2 pool_size ∈ [2,64] 的向量化展开。
//   tokenIndices: 输出行缓冲(expandOutLocal_, 基址 256B 对齐)
//   poolIdxTable: 池索引表(TopK 输出的 indicesOutLocal_, 基址对齐)
//   tpl:         [0,64)=r/ps 模板; [64,128)=r%ps 模板(均 256B 段对齐)
//   effExpand:   有效展开元素数 = Min(validS2Len, sparseCount) * ps (<= topk)
//   ps:          pool size(2 的幂, 且 <= 64, 由调用方保证)
// 尾块(lane >= effExpand)的展开结果为垃圾值, 由调用方负责用
// -1 Duplicate 覆盖 [effExpand, topk) 区间(垃圾不会越过 outputLen 暴露)。
__aicore__ inline void ExpandPow2PoolIndices(const LocalTensor<int32_t> &tokenIndices,
                                             const LocalTensor<uint32_t> &poolIdxTable, const LocalTensor<int32_t> &tpl,
                                             uint32_t effExpand, uint32_t ps)
{
    __ubuf__ uint32_t *out = (__ubuf__ uint32_t *)tokenIndices.GetPhyAddr();
    __ubuf__ uint32_t *table = (__ubuf__ uint32_t *)poolIdxTable.GetPhyAddr();
    __ubuf__ uint32_t *rIdxTpl = (__ubuf__ uint32_t *)tpl.GetPhyAddr();      // r / ps
    __ubuf__ uint32_t *rOffTpl = (__ubuf__ uint32_t *)tpl.GetPhyAddr() + 64; // r % ps

    if (effExpand == 0) {
        return;
    }
    // __VEC_SCOPE__ 内向量循环归纳变量/条件必须为 uint16_t(bisheng 约束);
    // chunkNum <= Align(4096,64)/64 = 64, ps <= 64, 均不溢出
    uint16_t chunkNum = static_cast<uint16_t>((effExpand + 63U) / 64U);
    const uint16_t psU16 = static_cast<uint16_t>(ps);
    // 每组: LoadAlign 64 个池索引 -> 覆盖 64*ps 个输出 = ps 个 64 lane chunk
    uint16_t groupNum = static_cast<uint16_t>((chunkNum + psU16 - 1U) / psU16);

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<uint32_t> rIdx;
        MicroAPI::RegTensor<uint32_t> rOff;
        MicroAPI::RegTensor<uint32_t> psReg;
        MicroAPI::RegTensor<uint32_t> poolReg;
        MicroAPI::RegTensor<uint32_t> gatherIdx;
        MicroAPI::RegTensor<uint32_t> gathered;

        MicroAPI::LoadAlign<uint32_t>(rIdx, rIdxTpl);
        MicroAPI::LoadAlign<uint32_t>(rOff, rOffTpl);
        MicroAPI::Duplicate<uint32_t>(psReg, ps, maskAll);

        for (uint16_t g = 0; g < groupNum; g++) {
            // 本组 64 个池索引装入寄存器(256B 对齐读)
            MicroAPI::LoadAlign<uint32_t>(poolReg, table + static_cast<uint32_t>(g) * 64U);
            // 本组实际输出 chunk 数: 尾组可能不足 ps 个(g < groupNum 保证 > 0)
            uint16_t chunksInGroup = static_cast<uint16_t>(chunkNum - static_cast<uint32_t>(g) * psU16);
            if (chunksInGroup > psU16) {
                chunksInGroup = psU16;
            }
            for (uint16_t j = 0; j < chunksInGroup; j++) {
                uint32_t c = static_cast<uint32_t>(g) * psU16 + j; // 全局输出 chunk 号
                // 组内相对选源索引基址: j*64/ps ∈ [0,64)(pow2 ps 可分解性),
                // lane 索引 = j*64/ps + r/ps, 恒落在 poolReg 的 64 lane 内;
                // ps <= 64 时 ps|64, chunkBase%ps==0, lane 偏移恒为 r%ps
                uint32_t relBase = (static_cast<uint32_t>(j) * 64U) / ps;
                MicroAPI::Adds(gatherIdx, rIdx, relBase, maskAll);
                MicroAPI::Gather(gathered, poolReg, gatherIdx);
                MicroAPI::Mul(gathered, gathered, psReg, maskAll);
                MicroAPI::Add(gathered, gathered, rOff, maskAll);
                MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(out + c * 64U, gathered, maskAll);
            }
        }
    }
}
} // namespace pkiexpand

#endif // POOL_KEY_INDEXER_VF_EXPAND_H
