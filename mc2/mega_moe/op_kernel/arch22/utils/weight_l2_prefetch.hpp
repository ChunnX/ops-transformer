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
 * \file weight_l2_prefetch.hpp
 * \brief A3(910_93) 权重 L2 预取：fire-and-forget 哑拷贝把权重字节拉进 L2（数据不消费），
 *        供 GEMM 真实 GM->L1 装载命中。AIV 走 GM->UB、AIC 走 GM->L1；scratch 为 16KB 环写
 *        独占区（AIV 在 UB 顶部洞、AIC 在 L1 功能区之上），各核按 (coreIdx, coreNum) 交错
 *        认领，对目标区间精确覆盖。当前两个窗口：
 *        - 窗口 0：全体 AIV 在 AllGather 自旋窗预取 W1 固定区（仅专家 1），在飞块由函数内
 *          全流水屏障收口后才放行轮就绪 flag；
 *        - W1 冷启动：AIC 在 flag0（轮就绪）wait 后、GMM1 组 0 wait 前预取 W1 仅专家 0，
 *          与 pull g0 并发排空，空专家按 cumsum 过滤。
 *        两窗字节集与时间窗均不重叠（自旋在 flag0 前、g0-pull 在后）。
 *        BS 超过 MAX_BS 时全部窗口不发射（大 BS 冷读已被深流水掩盖）。
 */

#ifndef MEGA_MOE_WEIGHT_L2_PREFETCH_HPP
#define MEGA_MOE_WEIGHT_L2_PREFETCH_HPP

#include "kernel_operator.h"

#include "get_tensor_addr.hpp"

// 窗口 0：AIV 在 AllGather 自旋窗预取 W1 固定区（仅专家 1）。
constexpr bool WEIGHT_L2_PREFETCH_ENABLE_WINDOW0 = false;
// W1 冷启动：AIC 在 flag0 后、GMM1 组 0 wait 前预取 W1 仅专家 0（g0-pull 窗）。
constexpr bool WEIGHT_L2_PREFETCH_ENABLE_W1_AIC_COLDSTART = true;

// 固定区 [W0_FIRST_EXPERT, +W0_EXPERTS)（当前 [1,2) 即仅专家 1，与冷启动的专家 0 互补）；
constexpr uint32_t WEIGHT_L2_PREFETCH_W0_FIRST_EXPERT = 1U;
constexpr uint32_t WEIGHT_L2_PREFETCH_W0_EXPERTS = 1U;

// 覆盖前 EXPERTS 个专家（当前仅专家 0），每专家全量预取
constexpr uint32_t WEIGHT_L2_PREFETCH_W0_AIC_COLDSTART_EXPERTS = 1U;

// AIV scratch 位于 UB 顶部洞：下界 179552（W4A8 swiglu 功能区上界），上界让出 UB 顶部常驻区
constexpr uint32_t WEIGHT_L2_PREFETCH_UB_OFFSET = 179552U;
constexpr uint32_t WEIGHT_L2_PREFETCH_CHUNK_BYTES = 16U * 1024U;
constexpr uint32_t WEIGHT_L2_PREFETCH_AIV_MIN_CHUNK_BYTES = 2048U;
// 融合场景框架在 L1/UB 各预留 256B
constexpr uint32_t WEIGHT_L2_PREFETCH_ARCH_RESERVED_BYTES = 256U;
// BS 门限：超过此值全部窗口不发射（大 BS 冷读已被深流水掩盖）
constexpr int64_t WEIGHT_L2_PREFETCH_MAX_BS = 512;

static_assert(WEIGHT_L2_PREFETCH_UB_OFFSET % 32U == 0U, "prefetch scratch must be 32B aligned");

// 预取游标：单一权重矩阵的专家区间，区间内按 worker 交错认领 chunk，走完即停。
struct WeightL2PrefetchCursor {
    __gm__ uint8_t *chunkAddr = nullptr; // 本 worker 在当前区域内的下一 chunk 地址
    __gm__ uint8_t *regionEnd = nullptr; // 当前区域结束地址（exclusive）
    uint32_t regionIndex = 0U;           // 当前目标专家（每次 Retarget 重置）
    uint32_t regionLimit = 0U;           // 本次 Retarget 的专家区间上界（exclusive）
    uint32_t regionCount = 0U;           // expertCount；regionIndex>=regionCount 表示耗尽
    uint32_t workerIdx = 0U;
    uint32_t workerCount = 1U;
    uint64_t issuedBytes = 0U; // 本游标累计已发射字节
    GM_ADDR weightList = nullptr;
    bool isPerExpertWeightTensor = false; // listLen!=1：每专家独立 tensor；否则单堆叠 tensor 步进
    uint32_t expertCount = 0U;
    uint64_t bytesPerExpert = 0U; // 调用方按 dtype 算好字节直接传入（规避 0.5 字节/element 的整除）
    AscendC::LocalTensor<uint8_t> scratch;                // [WEIGHT_L2_PREFETCH_UB_OFFSET, +chunkBytes) 的单缓冲
    uint32_t chunkBytes = WEIGHT_L2_PREFETCH_CHUNK_BYTES; // AIC 路径固定 16KB；AIV 按 UB 洞自适应
    bool active = false;
};

// 定位 regionIndex 对应专家的区域基址与本 worker 的起始 chunk。
__aicore__ inline void WeightL2PrefetchSetupRegion(WeightL2PrefetchCursor &cursor, uint64_t skipInRegion)
{
    if (cursor.regionIndex >= cursor.regionCount) {
        return;
    }
    uint32_t expertIdx = cursor.regionIndex;
    uint64_t regionBytes = cursor.bytesPerExpert;
    __gm__ uint8_t *regionBase =
        GetTensorAddr<uint8_t>(cursor.isPerExpertWeightTensor ? expertIdx : 0U, cursor.weightList);
    if (!cursor.isPerExpertWeightTensor) {
        // listLen==1 堆叠模式：与 kernel 的 gmGroupOffsetB 元素步进同源。
        regionBase += static_cast<uint64_t>(expertIdx) * regionBytes;
    }
    cursor.regionEnd = regionBase + regionBytes;
    // worker c 从偏移 (skip + c*CHUNK) mod regionBytes 起步；回绕重叠由 L2 去重，无害。
    uint64_t startOffset =
        regionBytes == 0U ? 0U :
                            (skipInRegion + static_cast<uint64_t>(cursor.workerIdx) * cursor.chunkBytes) % regionBytes;
    cursor.chunkAddr = regionBase + startOffset;
}

// 推进到下一专家区域。
__aicore__ inline void WeightL2PrefetchAdvanceRegion(WeightL2PrefetchCursor &cursor)
{
    ++cursor.regionIndex;
    WeightL2PrefetchSetupRegion(cursor, 0U);
}

// 重定位游标到专家区间 [firstExpert, lastExpert)。
__aicore__ inline void WeightL2PrefetchRetarget(WeightL2PrefetchCursor &cursor, uint32_t firstExpert,
                                                uint32_t lastExpert)
{
    if (!cursor.active) {
        return;
    }
    if (firstExpert > cursor.regionCount) {
        firstExpert = cursor.regionCount;
    }
    if (lastExpert > cursor.regionCount) {
        lastExpert = cursor.regionCount;
    }
    if (lastExpert < firstExpert) {
        lastExpert = firstExpert;
    }
    cursor.regionIndex = firstExpert;
    cursor.regionLimit = lastExpert;
    WeightL2PrefetchSetupRegion(cursor, 0U);
}

// 初始化游标；chunkBytes 为单次拷贝与交错步长
__aicore__ inline void WeightL2PrefetchInit(WeightL2PrefetchCursor &cursor, GM_ADDR weightList,
                                            bool isPerExpertWeightTensor, uint64_t bytesPerExpert, uint32_t expertCount,
                                            uint32_t workerIdx, uint32_t workerCount,
                                            AscendC::LocalTensor<uint8_t> scratchTensor,
                                            uint32_t chunkBytes = WEIGHT_L2_PREFETCH_CHUNK_BYTES)
{
    cursor.weightList = weightList;
    cursor.isPerExpertWeightTensor = isPerExpertWeightTensor;
    cursor.expertCount = expertCount;
    cursor.regionCount = expertCount;
    cursor.workerIdx = workerIdx;
    cursor.workerCount = workerCount > 0U ? workerCount : 1U;
    cursor.bytesPerExpert = bytesPerExpert;
    cursor.chunkBytes = chunkBytes & ~31U; // 32B 对齐向下取整（DataCopyPad 块粒度）
    cursor.scratch = scratchTensor;
    cursor.scratch.SetSize(cursor.chunkBytes);
    cursor.active =
        (workerCount > 0U && expertCount > 0U && cursor.chunkBytes >= WEIGHT_L2_PREFETCH_AIV_MIN_CHUNK_BYTES);
    cursor.regionIndex = 0U;
    cursor.regionLimit = 0U;
    WeightL2PrefetchSetupRegion(cursor, 0U);
}

// 发射一个 chunk 的 GM->UB 哑拷贝（不等待完成）；返回 false 表示游标耗尽（区间走完）。
__aicore__ inline bool WeightL2PrefetchTryIssue(WeightL2PrefetchCursor &cursor)
{
    while (cursor.active && cursor.regionIndex < cursor.regionLimit) {
        if (cursor.chunkAddr >= cursor.regionEnd) {
            WeightL2PrefetchAdvanceRegion(cursor);
            continue;
        }
        uint64_t bytesToRegionEnd = static_cast<uint64_t>(cursor.regionEnd - cursor.chunkAddr);
        uint32_t copyBytes =
            bytesToRegionEnd < cursor.chunkBytes ? static_cast<uint32_t>(bytesToRegionEnd) : cursor.chunkBytes;
        AscendC::GlobalTensor<uint8_t> weightGm;
        weightGm.SetGlobalBuffer(cursor.chunkAddr);
        AscendC::DataCopyPad(cursor.scratch, weightGm, {1, static_cast<uint16_t>(copyBytes), 0, 0}, {});
        cursor.issuedBytes += copyBytes;
        // 下一 chunk 跳过其它 worker 认领的 chunk（步长 = workerCount 个 chunk）。
        cursor.chunkAddr += static_cast<uint64_t>(cursor.workerCount) * cursor.chunkBytes;
        return true;
    }
    return false;
}

__aicore__ inline void WeightL2PrefetchBurstAll(WeightL2PrefetchCursor &cursor)
{
    while (WeightL2PrefetchTryIssue(cursor)) {
    }
}

/*
 * AIC 侧区间哑预取：GM->L1 环写，[firstExpert, lastExpert) 逐专家全量，各核按
 * (GetBlockIdx(), GetBlockNum()) 交错认领精确覆盖。
 */
__aicore__ inline void WeightL2PrefetchAicRange(GM_ADDR weightList, bool isPerExpertWeightTensor,
                                                uint64_t bytesPerExpert, uint32_t firstExpert, uint32_t lastExpert,
                                                AscendC::LocalTensor<uint8_t> l1Scratch)
{
    if (bytesPerExpert == 0U || firstExpert >= lastExpert) {
        return;
    }
    uint64_t limit = bytesPerExpert;
    uint32_t coreIdx = AscendC::GetBlockIdx();
    uint32_t coreNum = AscendC::GetBlockNum();
    l1Scratch.SetSize(WEIGHT_L2_PREFETCH_CHUNK_BYTES);
    AscendC::GlobalTensor<uint8_t> weightGm;
    for (uint32_t expertIdx = firstExpert; expertIdx < lastExpert; ++expertIdx) {
        // per-expert 模式取列表项；stacked 模式按专家字节步进（与 gmGroupOffsetB 同源）。
        __gm__ uint8_t *regionBase = GetTensorAddr<uint8_t>(isPerExpertWeightTensor ? expertIdx : 0U, weightList);
        if (!isPerExpertWeightTensor) {
            regionBase += static_cast<uint64_t>(expertIdx) * bytesPerExpert;
        }
        // 各核交错认领 16KB chunk，环写同一块保留区。
        for (uint64_t offset = static_cast<uint64_t>(coreIdx) * WEIGHT_L2_PREFETCH_CHUNK_BYTES; offset < limit;
             offset += static_cast<uint64_t>(coreNum) * WEIGHT_L2_PREFETCH_CHUNK_BYTES) {
            uint64_t remain = limit - offset;
            uint32_t chunkBytes = remain < WEIGHT_L2_PREFETCH_CHUNK_BYTES ? static_cast<uint32_t>(remain) :
                                                                            WEIGHT_L2_PREFETCH_CHUNK_BYTES;
            weightGm.SetGlobalBuffer(regionBase + offset);
            AscendC::DataCopy(l1Scratch, weightGm, chunkBytes);
        }
    }
}

// W1 冷启动：覆盖前 EXPERTS 个专家的 W1，由调用方在 flag0 后发射（g0-pull 窗排空）。
__aicore__ inline void WeightL2PrefetchAicColdStartW1(GM_ADDR weightList, bool isPerExpertWeightTensor,
                                                      uint64_t bytesPerExpert, uint32_t expertPerRank,
                                                      AscendC::LocalTensor<uint8_t> l1Scratch)
{
    if (!WEIGHT_L2_PREFETCH_ENABLE_W1_AIC_COLDSTART) {
        return;
    }
    uint32_t expertCount = expertPerRank < WEIGHT_L2_PREFETCH_W0_AIC_COLDSTART_EXPERTS ?
                               expertPerRank :
                               WEIGHT_L2_PREFETCH_W0_AIC_COLDSTART_EXPERTS;
    WeightL2PrefetchAicRange(weightList, isPerExpertWeightTensor, bytesPerExpert, 0U, expertCount, l1Scratch);
}

#endif // MEGA_MOE_WEIGHT_L2_PREFETCH_HPP
