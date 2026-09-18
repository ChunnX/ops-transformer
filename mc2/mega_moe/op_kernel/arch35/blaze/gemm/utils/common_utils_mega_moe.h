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

// Shared by the AIC consumer and AIV0 producer. Slots stay fixed across all
// persistent GMM calls, including tails; only the tensor's valid shape changes.
struct MegaMoeMxL1Layout {
    static constexpr uint64_t KB = 1024;
    static constexpr uint64_t A_STAGES = 2;
    static constexpr uint64_t B_STAGES = 4;
    static constexpr uint64_t A_SLOT_BYTES = 64 * KB;
    static constexpr uint64_t B_SLOT_BYTES = 64 * KB;
    static constexpr uint64_t SCALE_SLOT_BYTES = 32 * KB;
    static constexpr uint64_t HALF_L1_BYTES = 256 * KB;
    static constexpr uint64_t A_OFFSET = 128 * KB;
    static constexpr uint64_t SCALE_A_OFFSET = 192 * KB;
    static constexpr uint64_t SCALE_B_OFFSET = 224 * KB;

    // B4 map, in KiB:
    // [0,64) B0, [64,128) B2, [128,192) A0, [192,224) scA0, [224,256) scB0
    // [256,320) B1, [320,384) B3, [384,448) A1, [448,480) scA1, [480,512) scB1
    // Both AIC and AIV0 use this mapping for the B consumer/producer ring.
    __aicore__ static inline uint64_t BOffset(uint64_t slot)
    {
        return (slot & 1U) * HALF_L1_BYTES + (slot >> 1U) * B_SLOT_BYTES;
    }
};

// 静态和动态布局共用的数据描述；不包含流水策略和计算逻辑。
struct MegaMoeMxL1Allocation {
    static constexpr uint64_t MAX_A_STAGES = 6;
    static constexpr uint64_t MAX_B_STAGES = 8;
    uint64_t aStages = 0;
    uint64_t bStages = 0;
    uint64_t tileK = 0;
    uint64_t scaleK = 0;
    uint64_t aSlotBytes = 0;
    uint64_t bSlotBytes = 0;
    uint64_t scaleASlotBytes = 0;
    uint64_t scaleBSlotBytes = 0;
    uint64_t aOffsets[MAX_A_STAGES] = {};
    uint64_t bOffsets[MAX_B_STAGES] = {};
    uint64_t scaleAOffsets[2] = {};
    uint64_t scaleBOffsets[2] = {};
};

__aicore__ constexpr MegaMoeMxL1Allocation GetStaticMegaMoeMxL1Allocation()
{
    using Layout = MegaMoeMxL1Layout;
    MegaMoeMxL1Allocation result;
    result.aStages = Layout::A_STAGES;
    result.bStages = Layout::B_STAGES;
    result.tileK = MX_FP8FP4_L1_K_CONFIG_256;
    result.scaleK = MX_FP8FP4_SCALE_K_L1_SIZE;
    result.aSlotBytes = Layout::A_SLOT_BYTES;
    result.bSlotBytes = Layout::B_SLOT_BYTES;
    result.scaleASlotBytes = Layout::SCALE_SLOT_BYTES;
    result.scaleBSlotBytes = Layout::SCALE_SLOT_BYTES;
    for (uint64_t slot = 0; slot < Layout::A_STAGES; ++slot) {
        result.aOffsets[slot] = Layout::A_OFFSET + slot * Layout::HALF_L1_BYTES;
        result.scaleAOffsets[slot] = Layout::SCALE_A_OFFSET + slot * Layout::HALF_L1_BYTES;
        result.scaleBOffsets[slot] = Layout::SCALE_B_OFFSET + slot * Layout::HALF_L1_BYTES;
    }
    for (uint64_t slot = 0; slot < Layout::B_STAGES; ++slot) {
        result.bOffsets[slot] = (slot & 1U) * Layout::HALF_L1_BYTES + (slot >> 1U) * Layout::B_SLOT_BYTES;
    }
    return result;
}

// 静态对象只引用编译期配置，不在每次局部实例化时初始化或复制配置数组。
// 动态对象拥有外层计算好的配置，供整个专家循环复用。
template <bool UseDynamicL1>
struct MegaMoeMxL1Storage {
    MegaMoeMxL1Allocation allocation_;
    __aicore__ MegaMoeMxL1Storage() = default;
    __aicore__ MegaMoeMxL1Storage(const MegaMoeMxL1Allocation &allocation) : allocation_(allocation) {}
};

template <>
struct MegaMoeMxL1Storage<false> {
    static constexpr MegaMoeMxL1Allocation allocation_ = GetStaticMegaMoeMxL1Allocation();
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
    const uint64_t aStageBytes = AlignL1AllocationSize(tileM * KTile, SLOT_BUDGET_ALIGNMENT_BYTES);
    const uint64_t bStageBytes = AlignL1AllocationSize(tileN * KTile, SLOT_BUDGET_ALIGNMENT_BYTES);

    // 步骤 1：按一份完整 K 的 scaleA+scaleB 预算，估算 A/B 同槽数时可用的 A 槽数。
    const uint64_t maxScaleBytesPerRow = AlignL1AllocationSize(maxGmmK, SCALE_K_ALIGNMENT) / K_ELEMENTS_PER_SCALE_BYTE;
    const uint64_t fullKScalePairBytes =
        AlignL1AllocationSize(tileM * maxScaleBytesPerRow, SLOT_BUDGET_ALIGNMENT_BYTES) +
        AlignL1AllocationSize(tileN * maxScaleBytesPerRow, SLOT_BUDGET_ALIGNMENT_BYTES);
    uint64_t aStageCount = fullKScalePairBytes < L1_CAPACITY_BYTES ?
        (L1_CAPACITY_BYTES - fullKScalePairBytes) / (aStageBytes + bStageBytes) : 0;
    aStageCount = aStageCount == 0 ? 1 : aStageCount;
    aStageCount = aStageCount < MegaMoeMxL1Allocation::MAX_A_STAGES ? aStageCount : MegaMoeMxL1Allocation::MAX_A_STAGES;

    // 步骤 2：每个 scale 槽先仅覆盖一个 KTile；固定 A 后，尽可能将空间分给 B。
    const uint64_t minScaleAStageBytes = AlignL1AllocationSize(
        tileM * (KTile / K_ELEMENTS_PER_SCALE_BYTE), SLOT_BUDGET_ALIGNMENT_BYTES);
    const uint64_t minScaleBStageBytes = AlignL1AllocationSize(
        tileN * (KTile / K_ELEMENTS_PER_SCALE_BYTE), SLOT_BUDGET_ALIGNMENT_BYTES);
    const uint64_t minScaleTotalBytes = SCALE_STAGE_COUNT * (minScaleAStageBytes + minScaleBStageBytes);
    uint64_t bStageCount = (L1_CAPACITY_BYTES - aStageCount * aStageBytes - minScaleTotalBytes) / bStageBytes;
    bStageCount = bStageCount < MegaMoeMxL1Allocation::MAX_B_STAGES ? bStageCount : MegaMoeMxL1Allocation::MAX_B_STAGES;

    // 步骤 3：将余量分给两对 scale 槽，覆盖范围按 KTile 对齐且不超过任务所需的 K tile 数。
    const uint64_t remainingL1Bytes = L1_CAPACITY_BYTES - aStageCount * aStageBytes - bStageCount * bStageBytes;
    uint64_t scaleKTilesPerStage = remainingL1Bytes / minScaleTotalBytes;
    const uint64_t maxGmmKTileCount = maxGmmK == 0 ? 1 : (maxGmmK + KTile - 1) / KTile;
    scaleKTilesPerStage = scaleKTilesPerStage < maxGmmKTileCount ? scaleKTilesPerStage : maxGmmKTileCount;
    MegaMoeMxL1Allocation result;
    result.aStages = aStageCount;
    result.bStages = bStageCount;
    result.tileK = KTile;
    result.scaleK = scaleKTilesPerStage * KTile;
    result.aSlotBytes = aStageBytes;
    result.bSlotBytes = bStageBytes;
    result.scaleASlotBytes = AlignL1AllocationSize(
        tileM * (result.scaleK / K_ELEMENTS_PER_SCALE_BYTE), SLOT_BUDGET_ALIGNMENT_BYTES);
    result.scaleBSlotBytes = AlignL1AllocationSize(
        tileN * (result.scaleK / K_ELEMENTS_PER_SCALE_BYTE), SLOT_BUDGET_ALIGNMENT_BYTES);

    // 步骤 4：A 连续排列；B 的逻辑槽交错分布；末尾放置两对 scaleA/scaleB。
    uint64_t nextL1Offset = 0;
    for (uint64_t slot = 0; slot < aStageCount; ++slot) {
        result.aOffsets[slot] = nextL1Offset;
        nextL1Offset += aStageBytes;
    }
    for (uint64_t slot = 0; slot < bStageCount; ++slot) {
        const uint64_t physicalBSlot = (slot & 1) * ((bStageCount + 1) / 2) + slot / 2;
        result.bOffsets[slot] = nextL1Offset + physicalBSlot * bStageBytes;
    }
    nextL1Offset += bStageCount * bStageBytes;
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
