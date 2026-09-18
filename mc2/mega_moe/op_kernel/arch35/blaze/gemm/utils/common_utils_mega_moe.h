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
 * \file common_utils_mega_moe.h
 * \brief MXFP8FP4 grouped-matmul pipeline constants extracted from blaze for mega_moe.
 */
#pragma once

namespace Blaze {
namespace Gemm {

constexpr uint64_t MX_FP8FP4_L1_K_CONFIG_256 = 256UL;
constexpr uint64_t MX_FP8FP4_L1_K_CONFIG_512 = 512UL;
constexpr uint64_t MX_FP8FP4_L1_K_DYNAMIC_CONFIG_N_THRESHOLD = 128UL;
constexpr uint64_t MX_FP8FP4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD = 256UL;
constexpr uint64_t MX_FP8FP4_SCALE_K_L1_SIZE = 4096UL;
constexpr uint64_t DOUBLE_BUFFER = 2UL;
constexpr uint64_t SYNC_MODE4 = 4;

// 动态 L1 分配结果；静态路径保留原有类内常量。
struct MegaMoeMxL1Allocation {
    static constexpr uint64_t MAX_A_BUFFER_NUM = 6;
    static constexpr uint64_t MAX_B_BUFFER_NUM = 8;
    uint64_t aBufferNum = 0;
    uint64_t bBufferNum = 0;
    uint64_t tileK = 0;
    uint64_t scaleK = 0;
    uint64_t aSlotBytes = 0;
    uint64_t bSlotBytes = 0;
    uint64_t scaleASlotBytes = 0;
    uint64_t scaleBSlotBytes = 0;
    uint64_t aOffsets[MAX_A_BUFFER_NUM] = {};
    uint64_t bOffsets[MAX_B_BUFFER_NUM] = {};
    uint64_t scaleAOffsets[2] = {};
    uint64_t scaleBOffsets[2] = {};
};

// 动态对象保存专家循环复用的分配结果；静态特化不保存布局。
template <bool UseDynamicL1Layout>
struct MegaMoeMxL1Storage {
    MegaMoeMxL1Allocation allocation_;
    __aicore__ MegaMoeMxL1Storage() = default;
    __aicore__ MegaMoeMxL1Storage(const MegaMoeMxL1Allocation &allocation) : allocation_(allocation) {}
};

template <>
struct MegaMoeMxL1Storage<false> {
};

__aicore__ constexpr uint64_t AlignL1AllocationSize(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

/*
 * 为 A8W4 MoE 计算 L1 分配；A 和转换后的 B 均为一字节元素，scaleA/scaleB 各固定双缓冲。
 * 调用者提供正的 tile 维度，KTile 为 64 的倍数，A/B 单 tile 各不超过 64 KiB。
 *
 * 1. 按两个 GMM 的最大 K 估算完整 scale 预算，假设 A/B 槽数相同，确定 A 槽数。
 * 2. 固定 A 槽数，仅为 scale 保留双份最小 KTile 窗口，尽可能增加 B 槽数。
 * 3. 将最终余量分给双份 scaleA/scaleB，两者使用相同的 K 覆盖范围。
 * 4. 依次安排 A、B、scale 区域，并按交错顺序轮转 B 槽。
 *
 * 第一步的完整 scale 预算只用于选择 A 槽数，不保证 scale 常驻整个 K。
 * 预算过大时保留至少一个 A 槽，实际 scale 覆盖范围仍由第三步的剩余空间决定。
 */
template <uint64_t M, uint64_t N, uint64_t KTile>
__aicore__ inline MegaMoeMxL1Allocation CalcMegaMoeMxL1Allocation(uint64_t maxGmmK)
{
    constexpr uint64_t L1_CAPACITY_BYTES = 512 * 1024;
    constexpr uint64_t SCALE_STAGE_COUNT = 2;
    constexpr uint64_t A_ROW_ALIGNMENT = 16;
    constexpr uint64_t B_COLUMN_BUDGET_ALIGNMENT = 32;
    constexpr uint64_t SLOT_BUDGET_ALIGNMENT_BYTES = 512;
    constexpr uint64_t SCALE_K_ALIGNMENT = 64;
    constexpr uint64_t K_ELEMENTS_PER_SCALE_BYTE = 32;
    // 容量预算沿用原来的取整规则，不代表各搬运指令的最小地址对齐要求。
    // 当前 M=N=KTile=256，M/N 和槽容量取整均不增加额外空间。
    const uint64_t tileM = AlignL1AllocationSize(M, A_ROW_ALIGNMENT);
    const uint64_t tileN = AlignL1AllocationSize(N, B_COLUMN_BUDGET_ALIGNMENT);
    const uint64_t aBufferBytes = AlignL1AllocationSize(tileM * KTile, SLOT_BUDGET_ALIGNMENT_BYTES);
    const uint64_t bBufferBytes = AlignL1AllocationSize(tileN * KTile, SLOT_BUDGET_ALIGNMENT_BYTES);

    // 步骤 1：按一份完整 K 的 scaleA+scaleB 预算，估算 A/B 同槽数时可用的 A 槽数。
    const uint64_t maxScaleBytesPerRow = AlignL1AllocationSize(maxGmmK, SCALE_K_ALIGNMENT) / K_ELEMENTS_PER_SCALE_BYTE;
    const uint64_t fullKScalePairBytes =
        AlignL1AllocationSize(tileM * maxScaleBytesPerRow, SLOT_BUDGET_ALIGNMENT_BYTES) +
        AlignL1AllocationSize(tileN * maxScaleBytesPerRow, SLOT_BUDGET_ALIGNMENT_BYTES);
    uint64_t aBufferNum = fullKScalePairBytes < L1_CAPACITY_BYTES ?
        (L1_CAPACITY_BYTES - fullKScalePairBytes) / (aBufferBytes + bBufferBytes) : 0;
    aBufferNum = aBufferNum == 0 ? 1 : aBufferNum;
    aBufferNum = aBufferNum < MegaMoeMxL1Allocation::MAX_A_BUFFER_NUM ? aBufferNum : MegaMoeMxL1Allocation::MAX_A_BUFFER_NUM;

    // 步骤 2：每个 scale 槽先仅覆盖一个 KTile；固定 A 后，尽可能将空间分给 B。
    const uint64_t minScaleAStageBytes = AlignL1AllocationSize(
        tileM * (KTile / K_ELEMENTS_PER_SCALE_BYTE), SLOT_BUDGET_ALIGNMENT_BYTES);
    const uint64_t minScaleBStageBytes = AlignL1AllocationSize(
        tileN * (KTile / K_ELEMENTS_PER_SCALE_BYTE), SLOT_BUDGET_ALIGNMENT_BYTES);
    const uint64_t minScaleTotalBytes = SCALE_STAGE_COUNT * (minScaleAStageBytes + minScaleBStageBytes);
    uint64_t bBufferNum = (L1_CAPACITY_BYTES - aBufferNum * aBufferBytes - minScaleTotalBytes) / bBufferBytes;
    bBufferNum = bBufferNum < MegaMoeMxL1Allocation::MAX_B_BUFFER_NUM ? bBufferNum : MegaMoeMxL1Allocation::MAX_B_BUFFER_NUM;

    // 步骤 3：将余量分给两对 scale 槽，覆盖范围按 KTile 对齐且不超过任务所需的 K tile 数。
    const uint64_t remainingL1Bytes = L1_CAPACITY_BYTES - aBufferNum * aBufferBytes - bBufferNum * bBufferBytes;
    uint64_t scaleKTilesPerStage = remainingL1Bytes / minScaleTotalBytes;
    const uint64_t maxGmmKTileCount = maxGmmK == 0 ? 1 : (maxGmmK + KTile - 1) / KTile;
    scaleKTilesPerStage = scaleKTilesPerStage < maxGmmKTileCount ? scaleKTilesPerStage : maxGmmKTileCount;
    MegaMoeMxL1Allocation result;
    result.aBufferNum = aBufferNum;
    result.bBufferNum = bBufferNum;
    result.tileK = KTile;
    result.scaleK = scaleKTilesPerStage * KTile;
    result.aSlotBytes = aBufferBytes;
    result.bSlotBytes = bBufferBytes;
    result.scaleASlotBytes = AlignL1AllocationSize(
        tileM * (result.scaleK / K_ELEMENTS_PER_SCALE_BYTE), SLOT_BUDGET_ALIGNMENT_BYTES);
    result.scaleBSlotBytes = AlignL1AllocationSize(
        tileN * (result.scaleK / K_ELEMENTS_PER_SCALE_BYTE), SLOT_BUDGET_ALIGNMENT_BYTES);

    // 步骤 4：A 连续排列；B 的逻辑槽交错分布；末尾放置两对 scaleA/scaleB。
    uint64_t nextL1Offset = 0;
    for (uint64_t slot = 0; slot < aBufferNum; ++slot) {
        result.aOffsets[slot] = nextL1Offset;
        nextL1Offset += aBufferBytes;
    }
    for (uint64_t slot = 0; slot < bBufferNum; ++slot) {
        const uint64_t physicalBSlot = (slot & 1) * ((bBufferNum + 1) / 2) + slot / 2;
        result.bOffsets[slot] = nextL1Offset + physicalBSlot * bBufferBytes;
    }
    nextL1Offset += bBufferNum * bBufferBytes;
    for (uint64_t slot = 0; slot < SCALE_STAGE_COUNT; ++slot) {
        result.scaleAOffsets[slot] = nextL1Offset;
        nextL1Offset += result.scaleASlotBytes;
        result.scaleBOffsets[slot] = nextL1Offset;
        nextL1Offset += result.scaleBSlotBytes;
    }
    return result;
}

} // namespace Gemm
} // namespace Blaze
