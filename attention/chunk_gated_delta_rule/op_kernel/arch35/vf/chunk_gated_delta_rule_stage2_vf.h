/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHUNK_GATED_DELTA_RULE_STAGE2_VF_H
#define CHUNK_GATED_DELTA_RULE_STAGE2_VF_H

#include "kernel_tensor.h"

namespace ChunkGatedDeltaRule {
using namespace AscendC;
using namespace Reg;

/*
 * FP32 状态缩放：scale = exp(gCumLast)，state = state * scale。
 * exp 结果保留在寄存器中，整条计算链只读写一次 state UB。
 */
__simd_vf__ inline void ScaleFp32StateByExpVF(__ubuf__ float *stateAddr, float gCumLast, uint32_t count)
{
    constexpr uint32_t elementsPerRepeat = VECTOR_REG_WIDTH / sizeof(float);
    RegTensor<float> scaleReg;
    RegTensor<float> stateReg;
    MaskReg allMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg mask;

    Duplicate(scaleReg, gCumLast);
    Exp<float, MaskMergeMode::ZEROING>(scaleReg, scaleReg, allMask);

    uint32_t remaining = count;
    uint16_t repeatTimes = static_cast<uint16_t>((count + elementsPerRepeat - 1) / elementsPerRepeat);
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = UpdateMask<float>(remaining);
        uint32_t offset = static_cast<uint32_t>(i) * elementsPerRepeat;
        LoadAlign(stateReg, stateAddr + offset);
        Mul(stateReg, stateReg, scaleReg, mask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(stateAddr + offset, stateReg, mask);
    }
}

/*
 * BF16 gated 状态缩放：scale = exp(gCumLast)，dst = BF16(FP32(src) * scale)。
 * Cast、Exp、Mul、Cast 在同一个 VF 中完成，FP32 中间结果不再写回临时 UB。
 */
__simd_vf__ inline void ScaleBf16StateByExpVF(__ubuf__ bfloat16_t *dstAddr, __ubuf__ bfloat16_t *srcAddr,
                                              float gCumLast, uint32_t count)
{
    constexpr uint32_t elementsPerRepeat = VECTOR_REG_WIDTH / sizeof(float);
    static constexpr CastTrait castBf16ToFp32 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                 RoundMode::UNKNOWN};
    static constexpr CastTrait castFp32ToBf16 = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING,
                                                 RoundMode::CAST_RINT};
    RegTensor<bfloat16_t> srcBf16Reg;
    RegTensor<bfloat16_t> dstBf16Reg;
    RegTensor<float> stateReg;
    RegTensor<float> scaleReg;
    MaskReg allMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg mask;

    Duplicate(scaleReg, gCumLast);
    Exp<float, MaskMergeMode::ZEROING>(scaleReg, scaleReg, allMask);

    uint32_t remaining = count;
    uint16_t repeatTimes = static_cast<uint16_t>((count + elementsPerRepeat - 1) / elementsPerRepeat);
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = UpdateMask<float>(remaining);
        uint32_t offset = static_cast<uint32_t>(i) * elementsPerRepeat;
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(srcBf16Reg, srcAddr + offset);
        Cast<float, bfloat16_t, castBf16ToFp32>(stateReg, srcBf16Reg, mask);
        Mul(stateReg, stateReg, scaleReg, mask);
        Cast<bfloat16_t, float, castFp32ToBf16>(dstBf16Reg, stateReg, mask);
        StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(dstAddr + offset, dstBf16Reg, mask);
    }
}

/* 无门控路径保持原 Cast -> Muls(1.0) -> Cast 顺序，并在寄存器中完成。 */
__simd_vf__ inline void ScaleBf16StateVF(__ubuf__ bfloat16_t *dstAddr, __ubuf__ bfloat16_t *srcAddr, uint32_t count)
{
    constexpr uint32_t elementsPerRepeat = VECTOR_REG_WIDTH / sizeof(float);
    static constexpr CastTrait castBf16ToFp32 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                 RoundMode::UNKNOWN};
    static constexpr CastTrait castFp32ToBf16 = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING,
                                                 RoundMode::CAST_RINT};
    RegTensor<bfloat16_t> srcBf16Reg;
    RegTensor<bfloat16_t> dstBf16Reg;
    RegTensor<float> stateReg;
    RegTensor<float> oneReg;
    MaskReg mask;

    Duplicate(oneReg, static_cast<float>(1.0f));
    uint32_t remaining = count;
    uint16_t repeatTimes = static_cast<uint16_t>((count + elementsPerRepeat - 1) / elementsPerRepeat);
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = UpdateMask<float>(remaining);
        uint32_t offset = static_cast<uint32_t>(i) * elementsPerRepeat;
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(srcBf16Reg, srcAddr + offset);
        Cast<float, bfloat16_t, castBf16ToFp32>(stateReg, srcBf16Reg, mask);
        Mul(stateReg, stateReg, oneReg, mask);
        Cast<bfloat16_t, float, castFp32ToBf16>(dstBf16Reg, stateReg, mask);
        StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(dstAddr + offset, dstBf16Reg, mask);
    }
}

/*
 * BF16 chunk 状态：把旧状态和 Cube 已转成 BF16 的增量升到 FP32，计算 exp(G_last)*S+update，
 * 最后一次舍入写回 BF16 状态。此处融合衰减与加法，避免在两者之间再落地一个 BF16 中间状态。
 */
__simd_vf__ inline void UpdateBf16StateByExpVF(__ubuf__ bfloat16_t *stateAddr, __ubuf__ bfloat16_t *updateAddr,
                                               __ubuf__ float *gCumLastAddr, uint32_t count)
{
    constexpr uint32_t elementsPerRepeat = VECTOR_REG_WIDTH / sizeof(float);
    static constexpr CastTrait castBf16ToFp32 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                 RoundMode::UNKNOWN};
    static constexpr CastTrait castFp32ToBf16 = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING,
                                                 RoundMode::CAST_RINT};
    RegTensor<bfloat16_t> stateBf16Reg;
    RegTensor<bfloat16_t> updateBf16Reg;
    RegTensor<bfloat16_t> roundedBf16Reg;
    RegTensor<float> stateReg;
    RegTensor<float> updateReg;
    RegTensor<float> scaleReg;
    MaskReg allMask = CreateMask<float, MaskPattern::ALL>();
    DataCopy<float, LoadDist::DIST_BRC_B32>(scaleReg, gCumLastAddr);
    Exp<float, MaskMergeMode::ZEROING>(scaleReg, scaleReg, allMask);

    uint32_t remaining = count;
    const uint16_t repeats = static_cast<uint16_t>((count + elementsPerRepeat - 1) / elementsPerRepeat);
    for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
        MaskReg mask = UpdateMask<float>(remaining);
        const uint32_t offset = static_cast<uint32_t>(repeat) * elementsPerRepeat;
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(stateBf16Reg, stateAddr + offset);
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(updateBf16Reg, updateAddr + offset);
        Cast<float, bfloat16_t, castBf16ToFp32>(stateReg, stateBf16Reg, mask);
        Mul(stateReg, stateReg, scaleReg, mask);
        Cast<float, bfloat16_t, castBf16ToFp32>(updateReg, updateBf16Reg, mask);
        Add(stateReg, stateReg, updateReg, mask);
        Cast<bfloat16_t, float, castFp32ToBf16>(roundedBf16Reg, stateReg, mask);
        StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(stateAddr + offset, roundedBf16Reg, mask);
    }
}

/*
 * FP32 chunk 主状态常驻行主序 UB；Cube 增量通过 Fixpipe 以 FP32 直接送达。
 * 保持先乘后加的递推顺序，中间不做 BF16 舍入；但 Cube 的矩阵输入仍是 BF16，并非全链路 FP32。
 */
__simd_vf__ inline void UpdateFp32StateByExpVF(__ubuf__ float *stateAddr, __ubuf__ float *updateAddr,
                                               __ubuf__ float *gCumLastAddr, uint32_t count)
{
    constexpr uint32_t elementsPerRepeat = VECTOR_REG_WIDTH / sizeof(float);
    RegTensor<float> stateReg;
    RegTensor<float> updateReg;
    RegTensor<float> scaleReg;
    MaskReg allMask = CreateMask<float, MaskPattern::ALL>();
    DataCopy<float, LoadDist::DIST_BRC_B32>(scaleReg, gCumLastAddr);
    Exp<float, MaskMergeMode::ZEROING>(scaleReg, scaleReg, allMask);

    uint32_t remaining = count;
    const uint16_t repeats = static_cast<uint16_t>((count + elementsPerRepeat - 1) / elementsPerRepeat);
    for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
        MaskReg mask = UpdateMask<float>(remaining);
        const uint32_t offset = static_cast<uint32_t>(repeat) * elementsPerRepeat;
        LoadAlign(stateReg, stateAddr + offset);
        LoadAlign(updateReg, updateAddr + offset);
        Mul(stateReg, stateReg, scaleReg, mask);
        Add(stateReg, stateReg, updateReg, mask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(stateAddr + offset, stateReg, mask);
    }
}

/*
 * 将行主序 FP32 主状态转成 compact zN 的 BF16 副本供 AIC 读取，舍入方式为 CAST_RINT。
 * 转换不覆盖 FP32 主状态，后续跨 chunk 的衰减和累加继续使用原 FP32 值。
 */
__simd_vf__ inline void RowMajorFp32ToCompactZnBf16VF(__ubuf__ float *srcAddr, __ubuf__ bfloat16_t *dstZnAddr,
                                                      uint16_t rowCount, uint16_t colCount, uint16_t srcRowStride,
                                                      uint16_t dstFullRows)
{
    constexpr uint32_t elementsPerRepeat = VECTOR_REG_WIDTH / sizeof(float);
    constexpr uint32_t elementsPerBlock = 32 / sizeof(bfloat16_t);
    static constexpr CastTrait castFp32ToBf16 = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING,
                                                 RoundMode::CAST_RINT};
    RegTensor<float> srcReg;
    RegTensor<bfloat16_t> bf16Reg;
    RegTensor<uint16_t> packedBf16Reg;

    const uint16_t repeats = static_cast<uint16_t>((colCount + elementsPerRepeat - 1) / elementsPerRepeat);
    for (uint16_t row = 0; row < rowCount; ++row) {
        for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
            const uint32_t featureBegin = static_cast<uint32_t>(repeat) * elementsPerRepeat;
            const uint32_t remaining = colCount - featureBegin;
            uint32_t validCount = remaining < elementsPerRepeat ? remaining : elementsPerRepeat;
            uint32_t packedCount = validCount;
            MaskReg validMask = UpdateMask<float>(validCount);
            MaskReg packedMask = UpdateMask<uint16_t>(packedCount);
            const uint32_t srcOffset = static_cast<uint32_t>(row) * srcRowStride + featureBegin;
            const uint32_t dstOffset = static_cast<uint32_t>(repeat) * (elementsPerRepeat / elementsPerBlock) *
                                           dstFullRows * elementsPerBlock +
                                       static_cast<uint32_t>(row) * elementsPerBlock;
            LoadAlign(srcReg, srcAddr + srcOffset);
            Cast<bfloat16_t, float, castFp32ToBf16>(bf16Reg, srcReg, validMask);
            Pack<uint16_t, uint32_t, HighLowPart::LOWEST>(packedBf16Reg,
                                                          reinterpret_cast<RegTensor<uint32_t> &>(bf16Reg));
            StoreAlign<uint16_t, DataCopyMode::DATA_BLOCK_COPY>(
                reinterpret_cast<__ubuf__ uint16_t *>(dstZnAddr) + dstOffset, packedBf16Reg, dstFullRows, packedMask);
        }
    }
}

// 初始 BF16 状态从行主序转为 compact zN；之后状态更新和 UB->L1 发布一直沿用该布局。
__simd_vf__ inline void RowMajorToCompactZnBf16VF(__ubuf__ bfloat16_t *srcAddr, __ubuf__ bfloat16_t *dstZnAddr,
                                                  uint16_t rowCount, uint16_t colCount, uint16_t srcRowStride,
                                                  uint16_t dstFullRows)
{
    constexpr uint32_t elementsPerRepeat = VECTOR_REG_WIDTH / sizeof(float);
    constexpr uint32_t elementsPerBlock = 32 / sizeof(bfloat16_t);
    RegTensor<bfloat16_t> srcBf16Reg;
    RegTensor<uint16_t> packedBf16Reg;

    const uint16_t repeats = static_cast<uint16_t>((colCount + elementsPerRepeat - 1) / elementsPerRepeat);
    for (uint16_t row = 0; row < rowCount; ++row) {
        for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
            const uint32_t srcOffset =
                static_cast<uint32_t>(row) * srcRowStride + static_cast<uint32_t>(repeat) * elementsPerRepeat;
            const uint32_t dstOffset = static_cast<uint32_t>(repeat) * (elementsPerRepeat / elementsPerBlock) *
                                           dstFullRows * elementsPerBlock +
                                       static_cast<uint32_t>(row) * elementsPerBlock;
            const uint32_t remaining = colCount - static_cast<uint32_t>(repeat) * elementsPerRepeat;
            uint32_t packedCount = remaining < elementsPerRepeat ? remaining : elementsPerRepeat;
            MaskReg packedMask = UpdateMask<uint16_t>(packedCount);
            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(srcBf16Reg, srcAddr + srcOffset);
            Pack<uint16_t, uint32_t, HighLowPart::LOWEST>(packedBf16Reg,
                                                          reinterpret_cast<RegTensor<uint32_t> &>(srcBf16Reg));
            StoreAlign<uint16_t, DataCopyMode::DATA_BLOCK_COPY>(
                reinterpret_cast<__ubuf__ uint16_t *>(dstZnAddr) + dstOffset, packedBf16Reg, dstFullRows, packedMask);
        }
    }
}

// 优化方案（Dk<=8）：一个 32 字节 FP32 数据块容纳一行，将 8 个独立 Dv 行打包进一个寄存器。
// state:[rowCount,128]；有效子矩阵 [rowCount,Dk] 打包成寄存器 [8行,8列]，多余 lane 置零。
// 每行执行 r=beta*(v-exp(g)*dot(S,k))，S=exp(g)*S+r*k，o=dot(S,scale*q)。
// ReduceDataBlock 按 8 元素分组归约，padding lane 必须排除在乘法和归约之外。
// 每批 token 内状态保留在寄存器，仅在批首 Gather、批尾 Scatter；搬运步长与通用批处理一致。
template <bool hasGamma>
__simd_vf__ inline void StreamingExactPackedRowsVF(__ubuf__ float *stateAddr, __ubuf__ bfloat16_t *queryAddr,
                                                   __ubuf__ bfloat16_t *keyAddr, __ubuf__ float *valueAddr,
                                                   __ubuf__ float *betaAddr, __ubuf__ float *gAddr,
                                                   __ubuf__ float *outAddr, float scale, uint16_t rowCount, uint16_t dk,
                                                   uint16_t stateRowStride, uint16_t tokenCount)
{
    static constexpr CastTrait castBf16ToFp32 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                 RoundMode::UNKNOWN};
    RegTensor<int32_t> lane, rowIndex, featureIndex, offset, tempIndex;
    RegTensor<float> state, key, query, source, product, reduced, total;
    RegTensor<float> alpha, beta, value, delta, deltaBroadcast;
    RegTensor<bfloat16_t> bf16;
    MaskReg allMask = CreateMask<float, MaskPattern::ALL>();
    Arange(lane, static_cast<int32_t>(0));
    ShiftRights(rowIndex, lane, static_cast<int16_t>(3), allMask);
    Muls(tempIndex, rowIndex, static_cast<int32_t>(8), allMask);
    Sub(featureIndex, lane, tempIndex, allMask);
    // Padding 只占存储空间；乘法及两次分段归约都必须使用排除 padding 的掩码。
    MaskReg featureMask;
    Compares<int32_t, CMPMODE::LT>(featureMask, featureIndex, static_cast<int32_t>(dk), allMask);
    Muls(offset, rowIndex, static_cast<int32_t>(stateRowStride), allMask);
    Add(offset, offset, featureIndex, allMask);
    // UB Gather/Scatter 索引以对应类型的元素数计，不能乘 sizeof(float) 当作字节偏移。

    for (uint16_t rowBegin = 0; rowBegin < rowCount; rowBegin += 8) {
        uint32_t validRows = rowCount - rowBegin < 8 ? rowCount - rowBegin : 8;
        uint32_t packedCount = validRows * 8;
        MaskReg packedMask = UpdateMask<float>(packedCount);
        And(packedMask, packedMask, featureMask, allMask);
        MaskReg rowMask = UpdateMask<float>(validRows);
        auto stateBase = stateAddr + static_cast<uint32_t>(rowBegin) * stateRowStride;
        Gather(state, stateBase, reinterpret_cast<RegTensor<uint32_t> &>(offset), packedMask);
        for (uint16_t token = 0; token < tokenCount; ++token) {
            const uint32_t inputOffset = static_cast<uint32_t>(token) * stateRowStride;
            if constexpr (hasGamma) {
                LoadAlign<float, LoadDist::DIST_BRC_B32>(alpha, gAddr + token * 8);
                Exp<float, MaskMergeMode::ZEROING>(alpha, alpha, allMask);
            } else {
                Duplicate(alpha, static_cast<float>(1.0f));
            }
            LoadAlign<float, LoadDist::DIST_BRC_B32>(beta, betaAddr + token * 16);
            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(bf16, keyAddr + inputOffset);
            Cast<float, bfloat16_t, castBf16ToFp32>(source, bf16, allMask);
            Gather<float, uint32_t>(key, source, reinterpret_cast<RegTensor<uint32_t> &>(featureIndex));
            Mul(product, state, key, packedMask);
            ReduceDataBlock<ReduceType::SUM>(reduced, product, packedMask);
            Duplicate(total, static_cast<float>(0.0f));
            Add(total, total, reduced, rowMask);
            LoadAlign(value, valueAddr + inputOffset + rowBegin);
            Mul(product, alpha, total, rowMask);
            Sub(delta, value, product, rowMask);
            Mul(delta, delta, beta, rowMask);
            Gather<float, uint32_t>(deltaBroadcast, delta, reinterpret_cast<RegTensor<uint32_t> &>(rowIndex));
            Mul(state, state, alpha, packedMask);
            Mul(product, key, deltaBroadcast, packedMask);
            Add(state, state, product, packedMask);
            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(bf16, queryAddr + inputOffset);
            Cast<float, bfloat16_t, castBf16ToFp32>(source, bf16, allMask);
            Gather<float, uint32_t>(query, source, reinterpret_cast<RegTensor<uint32_t> &>(featureIndex));
            Muls(query, query, scale, packedMask);
            Mul(product, state, query, packedMask);
            ReduceDataBlock<ReduceType::SUM>(reduced, product, packedMask);
            Duplicate(total, static_cast<float>(0.0f));
            Add(total, total, reduced, rowMask);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(outAddr + static_cast<uint32_t>(token) * 64 + rowBegin, total,
                                                        rowMask);
        }
        Scatter(stateBase, state, reinterpret_cast<RegTensor<uint32_t> &>(offset), packedMask);
    }
}

/*
 * 通用 Streaming 每次 VF 处理一批 token 和一个 AIV 独立拥有的 Dv 行切片。
 * UB 物理 shape：S:[rowCount,128]，Q/K/V:[tokenCount,128]，out:[tokenCount,64]；
 * beta:[tokenCount,16]、g:[tokenCount,8] 为 DMA 对齐布局，每 token 只用首元素。
 * 优化方案：状态行之间无依赖，先遍历行，再在寄存器中沿 token 递推，减少 UB 状态读写：
 * delta_r=beta*(v_r-alpha*dot(S_r,k))，S_r=alpha*S_r+delta_r*k，out_r=dot(S_r,scale*q)。
 * 两种接口状态类型都以 FP32 持续递推，仅在调用方写出 BF16 finalState 时量化状态。
 * Chunk 路径还会量化 Cube 输入和 BF16 接口的 delta，因此 Exact 不表示两条路径逐位相同。
 */
template <bool hasGamma>
__simd_vf__ inline void StreamingExactBatchVF(__ubuf__ float *stateAddr, __ubuf__ bfloat16_t *queryAddr,
                                              __ubuf__ bfloat16_t *keyAddr, __ubuf__ float *valueAddr,
                                              __ubuf__ float *betaAddr, __ubuf__ float *gAddr, __ubuf__ float *outAddr,
                                              float scale, uint16_t rowCount, uint16_t dk, uint16_t stateRowStride,
                                              uint16_t tokenCount)
{
    constexpr uint32_t elementsPerRepeat = VECTOR_REG_WIDTH / sizeof(float);
    static constexpr CastTrait castBf16ToFp32 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                 RoundMode::UNKNOWN};

    RegTensor<bfloat16_t> inputBf16Reg;
    RegTensor<float> stateLowReg;
    RegTensor<float> stateHighReg;
    RegTensor<float> keyLowReg;
    RegTensor<float> keyHighReg;
    RegTensor<float> queryReg;
    RegTensor<float> productReg;
    RegTensor<float> reducedReg;
    RegTensor<float> totalReg;
    RegTensor<float> alphaReg;
    RegTensor<float> betaReg;
    RegTensor<float> valueReg;
    RegTensor<float> deltaReg;
    RegTensor<float> deltaBroadcastReg;
    MaskReg allMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg oneMask = CreateMask<float, MaskPattern::VL1>();

    uint32_t lowCount = dk < elementsPerRepeat ? dk : elementsPerRepeat;
    uint32_t highCount = dk > elementsPerRepeat ? dk - elementsPerRepeat : 0;
    MaskReg lowMask = UpdateMask<float>(lowCount);
    MaskReg highMask = UpdateMask<float>(highCount);
    // 状态行相互独立；一行的前后两个特征片段在整批 token 内驻留寄存器。
    // Dk>64 时保持两次 ReduceSum/Add 的原有顺序，避免额外改变浮点累加顺序。
    for (uint16_t row = 0; row < rowCount; ++row) {
        const uint32_t stateOffset = static_cast<uint32_t>(row) * stateRowStride;
        LoadAlign(stateLowReg, stateAddr + stateOffset);
        if (dk > elementsPerRepeat) {
            LoadAlign(stateHighReg, stateAddr + stateOffset + elementsPerRepeat);
        }
        for (uint16_t token = 0; token < tokenCount; ++token) {
            const uint32_t inputOffset = static_cast<uint32_t>(token) * stateRowStride;
            if constexpr (hasGamma) {
                LoadAlign<float, LoadDist::DIST_BRC_B32>(alphaReg, gAddr + token * 8);
                Exp<float, MaskMergeMode::ZEROING>(alphaReg, alphaReg, allMask);
            } else {
                Duplicate(alphaReg, static_cast<float>(1.0f));
            }
            LoadAlign<float, LoadDist::DIST_BRC_B32>(betaReg, betaAddr + token * 16);
            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(inputBf16Reg, keyAddr + inputOffset);
            Cast<float, bfloat16_t, castBf16ToFp32>(keyLowReg, inputBf16Reg, lowMask);
            Mul(productReg, stateLowReg, keyLowReg, lowMask);
            ReduceSum(reducedReg, productReg, lowMask);
            Duplicate(totalReg, static_cast<float>(0.0f));
            Add(totalReg, totalReg, reducedReg, oneMask);
            if (dk > elementsPerRepeat) {
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(inputBf16Reg,
                                                                 keyAddr + inputOffset + elementsPerRepeat);
                Cast<float, bfloat16_t, castBf16ToFp32>(keyHighReg, inputBf16Reg, highMask);
                Mul(productReg, stateHighReg, keyHighReg, highMask);
                ReduceSum(reducedReg, productReg, highMask);
                Add(totalReg, totalReg, reducedReg, oneMask);
            }

            LoadAlign<float, LoadDist::DIST_BRC_B32>(valueReg, valueAddr + inputOffset + row);
            Mul(productReg, alphaReg, totalReg, oneMask);
            Sub(deltaReg, valueReg, productReg, oneMask);
            Mul(deltaReg, deltaReg, betaReg, oneMask);
            // 标量递推仅定义 lane 0；外积更新前须广播到全部有效 Dk lane。
            Duplicate(deltaBroadcastReg, deltaReg, allMask);

            Mul(stateLowReg, stateLowReg, alphaReg, lowMask);
            Mul(productReg, keyLowReg, deltaBroadcastReg, lowMask);
            Add(stateLowReg, stateLowReg, productReg, lowMask);
            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(inputBf16Reg, queryAddr + inputOffset);
            Cast<float, bfloat16_t, castBf16ToFp32>(queryReg, inputBf16Reg, lowMask);
            Muls(queryReg, queryReg, scale, lowMask);
            Mul(productReg, stateLowReg, queryReg, lowMask);
            ReduceSum(reducedReg, productReg, lowMask);
            Duplicate(totalReg, static_cast<float>(0.0f));
            Add(totalReg, totalReg, reducedReg, oneMask);
            if (dk > elementsPerRepeat) {
                Mul(stateHighReg, stateHighReg, alphaReg, highMask);
                Mul(productReg, keyHighReg, deltaBroadcastReg, highMask);
                Add(stateHighReg, stateHighReg, productReg, highMask);
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(inputBf16Reg,
                                                                 queryAddr + inputOffset + elementsPerRepeat);
                Cast<float, bfloat16_t, castBf16ToFp32>(queryReg, inputBf16Reg, highMask);
                Muls(queryReg, queryReg, scale, highMask);
                Mul(productReg, stateHighReg, queryReg, highMask);
                ReduceSum(reducedReg, productReg, highMask);
                Add(totalReg, totalReg, reducedReg, oneMask);
            }
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                outAddr + static_cast<uint32_t>(token) * elementsPerRepeat + row, totalReg, oneMask);
        }
        StoreAlign<float, StoreDist::DIST_NORM_B32>(stateAddr + stateOffset, stateLowReg, lowMask);
        if (dk > elementsPerRepeat) {
            StoreAlign<float, StoreDist::DIST_NORM_B32>(stateAddr + stateOffset + elementsPerRepeat, stateHighReg,
                                                        highMask);
        }
    }
}

} // namespace ChunkGatedDeltaRule

#endif // CHUNK_GATED_DELTA_RULE_STAGE2_VF_H
