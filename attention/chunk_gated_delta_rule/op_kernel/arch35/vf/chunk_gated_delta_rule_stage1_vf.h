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
using namespace Reg;

/*
 * 按传入的 chunkSize 计算 FP32 inclusive cumsum、exp 和 Gamma：
 *   gCum[i] = sum(g[0:i + 1])
 *   gCumExp[i] = exp(gCum[i])
 *   gamma[i, j] = j < i ? exp(gCum[i] - gCum[j]) : 0
 * gCum 计算完后保留在寄存器中直接生成 Gamma，避免再次从 UB 加载以及两次 Broadcast。
 */
__simd_vf__ inline void CumSumExpVF(__ubuf__ float *gAddr, __ubuf__ float *gCumAddr, __ubuf__ float *gCumExpAddr,
                                    __ubuf__ float *gammaAddr, uint32_t chunkSize)
{
    // chunkSize <= 单个 FP32 Vector 寄存器的元素数，因此一个寄存器可保存整个 chunk 的 g/gCum
    RegTensor<float> gCumReg;
    RegTensor<float> shiftedReg;
    RegTensor<float> gCumExpReg;
    RegTensor<int32_t> laneIndexReg;
    RegTensor<int32_t> gatherIndexReg;
    // validMask 只使能当前 chunk 的有效 lane，避免寄存器尾部数据参与计算和写回
    MaskReg allMask = CreateMask<int32_t, MaskPattern::ALL>();
    uint32_t validCount = chunkSize;
    MaskReg validMask = UpdateMask<float>(validCount);
    MaskReg addMask;
    MaskReg gammaMask;

    // gCumReg 初始为原始 g；laneIndexReg 保存 [0, 1, ..., chunkSize - 1]，用于构造移位索引
    LoadAlign<float, LoadDist::DIST_NORM>(gCumReg, gAddr);
    Arange(laneIndexReg, static_cast<int32_t>(0));

    // 使用 Hillis-Steele inclusive scan 计算前缀和。每轮 stride 翻倍，分别合并相邻的
    // 1、2、4、... 个元素，经过 ceil(log2(chunkSize)) 轮后得到：
    //   gCumReg[i] = g[0] + g[1] + ... + g[i]。
    for (int32_t stride = 1; stride < static_cast<int32_t>(chunkSize); stride <<= 1) {
        // shiftedReg[i] 读取上一轮的 gCumReg[i - stride]。索引小于 0 的 lane 保持为0
        // 再通过 addMask 屏蔽，因此这些 lane 保持原值，不会重复累加 gCumReg[0]
        Adds<int32_t>(gatherIndexReg, laneIndexReg, -stride, allMask);
        Maxs<int32_t>(gatherIndexReg, gatherIndexReg, static_cast<int32_t>(0), allMask);
        Gather<float, uint32_t>(shiftedReg, gCumReg, reinterpret_cast<RegTensor<uint32_t> &>(gatherIndexReg));
        Compares<int32_t, CMPMODE::GE>(addMask, laneIndexReg, stride, validMask);
        Add<float, MaskMergeMode::MERGING>(gCumReg, gCumReg, shiftedReg, addMask);
    }

    Exp<float, MaskMergeMode::ZEROING>(gCumExpReg, gCumReg, validMask);
    StoreAlign<float, StoreDist::DIST_NORM>(gCumAddr, gCumReg, validMask);
    StoreAlign<float, StoreDist::DIST_NORM>(gCumExpAddr, gCumExpReg, validMask);

    // stageOneMask 是严格下三角矩阵。直接用 laneIndexReg < row 生成行 mask，
    // 在 Gamma 矩阵中写出对角线及上三角为 0，不再从 GM 搬运 mask 或执行两次 Mul。
    Duplicate(gCumExpReg, static_cast<float>(0.0));
    StoreAlign<float, StoreDist::DIST_NORM_B32>(gammaAddr, gCumExpReg, validMask);
    for (uint16_t row = 1; row < static_cast<uint16_t>(chunkSize); ++row) {
        uint32_t rowOffset = static_cast<uint32_t>(row) * chunkSize;
        Duplicate(reinterpret_cast<RegTensor<uint32_t> &>(gatherIndexReg), static_cast<uint32_t>(row));
        Gather<float, uint32_t>(shiftedReg, gCumReg, reinterpret_cast<RegTensor<uint32_t> &>(gatherIndexReg));
        Compares<int32_t, CMPMODE::LT>(gammaMask, laneIndexReg, static_cast<int32_t>(row), validMask);
        Sub<float, MaskMergeMode::ZEROING>(gCumExpReg, shiftedReg, gCumReg, gammaMask);
        Exp<float, MaskMergeMode::ZEROING>(gCumExpReg, gCumExpReg, gammaMask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(gammaAddr + rowOffset, gCumExpReg, validMask);
    }
}

/* Exact arch35 fast path: both AIVs retain the complete 64-element g prefix. */
__simd_vf__ inline void CumSumExpExactVF(__ubuf__ float *gAddr, __ubuf__ float *gCumAddr, __ubuf__ float *gCumExpAddr,
                                         uint16_t chunkSize)
{
    RegTensor<float> gCumReg;
    RegTensor<float> shiftedReg;
    RegTensor<float> resultReg;
    RegTensor<int32_t> laneIndexReg;
    RegTensor<int32_t> gatherIndexReg;
    MaskReg allMask = CreateMask<int32_t, MaskPattern::ALL>();
    uint32_t validCount = chunkSize;
    MaskReg validMask = UpdateMask<float>(validCount);
    MaskReg addMask;

    LoadAlign<float, LoadDist::DIST_NORM>(gCumReg, gAddr);
    Arange(laneIndexReg, static_cast<int32_t>(0));
    for (int32_t stride = 1; stride < static_cast<int32_t>(chunkSize); stride <<= 1) {
        Adds<int32_t>(gatherIndexReg, laneIndexReg, -stride, allMask);
        Maxs<int32_t>(gatherIndexReg, gatherIndexReg, static_cast<int32_t>(0), allMask);
        Gather<float, uint32_t>(shiftedReg, gCumReg, reinterpret_cast<RegTensor<uint32_t> &>(gatherIndexReg));
        Compares<int32_t, CMPMODE::GE>(addMask, laneIndexReg, stride, validMask);
        Add<float, MaskMergeMode::MERGING>(gCumReg, gCumReg, shiftedReg, addMask);
    }

    Exp<float, MaskMergeMode::ZEROING>(resultReg, gCumReg, validMask);
    StoreAlign<float, StoreDist::DIST_NORM>(gCumAddr, gCumReg, validMask);
    StoreAlign<float, StoreDist::DIST_NORM>(gCumExpAddr, resultReg, validMask);
}

/*
 * Full-width specialization of the Stage1 transform.  Keeping Q/K/V in one
 * constant-width loop preserves the proven arch35 VF schedule for dk=dv=128;
 * the generic exact path below remains the source of truth for smaller widths.
 */
__simd_vf__ inline void PrepareStage1FullDimSliceVF(__ubuf__ bfloat16_t *qAddr, __ubuf__ bfloat16_t *kAddr,
                                                    __ubuf__ bfloat16_t *vAddr, __ubuf__ bfloat16_t *rawKcdZnAddr,
                                                    __ubuf__ bfloat16_t *vBetaZnAddr, __ubuf__ bfloat16_t *qPrimeZnAddr,
                                                    __ubuf__ bfloat16_t *kgZnAddr, __ubuf__ float *betaAddr,
                                                    __ubuf__ float *gCumAddr, __ubuf__ float *gCumExpAddr, float scale,
                                                    uint16_t rowStart, uint16_t rowCount, uint16_t dim,
                                                    uint16_t dstRowStart, uint16_t dstFullRows)
{
    constexpr uint32_t elementsPerRepeat = VECTOR_REG_WIDTH / sizeof(float);
    constexpr uint32_t elementsPerBlock = 32 / sizeof(bfloat16_t);
    static constexpr CastTrait castBf16ToFp32 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                 RoundMode::UNKNOWN};
    static constexpr CastTrait castFp32ToBf16 = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING,
                                                 RoundMode::CAST_RINT};
    RegTensor<bfloat16_t> qBf16Reg;
    RegTensor<bfloat16_t> kBf16Reg;
    RegTensor<bfloat16_t> vBf16Reg;
    RegTensor<bfloat16_t> outBf16Reg;
    RegTensor<uint16_t> packedBf16Reg;
    RegTensor<float> qReg;
    RegTensor<float> kReg;
    RegTensor<float> vReg;
    RegTensor<float> betaReg;
    RegTensor<float> gExpReg;
    RegTensor<float> decayReg;
    RegTensor<float> tmpReg;
    RegTensor<float> rawKcdReg;
    MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
    uint32_t packedCount = elementsPerRepeat;
    MaskReg packedMask = UpdateMask<uint16_t>(packedCount);

    DataCopy<float, LoadDist::DIST_BRC_B32>(decayReg, gCumAddr + 63);
    const uint16_t repeats = static_cast<uint16_t>((dim + elementsPerRepeat - 1) / elementsPerRepeat);
    for (uint16_t row = 0; row < rowCount; ++row) {
        DataCopy<float, LoadDist::DIST_BRC_B32>(betaReg, betaAddr + row);
        DataCopy<float, LoadDist::DIST_BRC_B32>(gExpReg, gCumExpAddr + rowStart + row);
        DataCopy<float, LoadDist::DIST_BRC_B32>(tmpReg, gCumAddr + rowStart + row);
        Sub(tmpReg, decayReg, tmpReg, fullMask);
        Exp<float, MaskMergeMode::ZEROING>(tmpReg, tmpReg, fullMask);

        for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
            const uint32_t offset = static_cast<uint32_t>(row) * dim + repeat * elementsPerRepeat;
            const uint32_t znOffset = static_cast<uint32_t>(repeat) * (elementsPerRepeat / elementsPerBlock) *
                                          dstFullRows * elementsPerBlock +
                                      static_cast<uint32_t>(dstRowStart + row) * elementsPerBlock;
            uint32_t remaining = dim - static_cast<uint32_t>(repeat) * elementsPerRepeat;
            MaskReg mask = UpdateMask<float>(remaining);
            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(qBf16Reg, qAddr + offset);
            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(kBf16Reg, kAddr + offset);
            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(vBf16Reg, vAddr + offset);
            Cast<float, bfloat16_t, castBf16ToFp32>(qReg, qBf16Reg, mask);
            Cast<float, bfloat16_t, castBf16ToFp32>(kReg, kBf16Reg, mask);
            Cast<float, bfloat16_t, castBf16ToFp32>(vReg, vBf16Reg, mask);

            Mul(rawKcdReg, betaReg, gExpReg, mask);
            Muls(rawKcdReg, rawKcdReg, static_cast<float>(-1.0f), mask);
            Mul(rawKcdReg, kReg, rawKcdReg, mask);
            Cast<bfloat16_t, float, castFp32ToBf16>(outBf16Reg, rawKcdReg, mask);
            Pack<uint16_t, uint32_t, HighLowPart::LOWEST>(packedBf16Reg,
                                                          reinterpret_cast<RegTensor<uint32_t> &>(outBf16Reg));
            StoreAlign<uint16_t, DataCopyMode::DATA_BLOCK_COPY>(
                reinterpret_cast<__ubuf__ uint16_t *>(rawKcdZnAddr) + znOffset, packedBf16Reg, dstFullRows, packedMask);

            Mul(kReg, kReg, tmpReg, mask);
            Cast<bfloat16_t, float, castFp32ToBf16>(outBf16Reg, kReg, mask);
            Pack<uint16_t, uint32_t, HighLowPart::LOWEST>(packedBf16Reg,
                                                          reinterpret_cast<RegTensor<uint32_t> &>(outBf16Reg));
            StoreAlign<uint16_t, DataCopyMode::DATA_BLOCK_COPY>(
                reinterpret_cast<__ubuf__ uint16_t *>(kgZnAddr) + znOffset, packedBf16Reg, dstFullRows, packedMask);

            Muls(qReg, qReg, scale, mask);
            Mul(qReg, qReg, gExpReg, mask);
            Cast<bfloat16_t, float, castFp32ToBf16>(outBf16Reg, qReg, mask);
            Pack<uint16_t, uint32_t, HighLowPart::LOWEST>(packedBf16Reg,
                                                          reinterpret_cast<RegTensor<uint32_t> &>(outBf16Reg));
            StoreAlign<uint16_t, DataCopyMode::DATA_BLOCK_COPY>(
                reinterpret_cast<__ubuf__ uint16_t *>(qPrimeZnAddr) + znOffset, packedBf16Reg, dstFullRows, packedMask);

            Mul(vReg, vReg, betaReg, mask);
            Cast<bfloat16_t, float, castFp32ToBf16>(outBf16Reg, vReg, mask);
            Pack<uint16_t, uint32_t, HighLowPart::LOWEST>(packedBf16Reg,
                                                          reinterpret_cast<RegTensor<uint32_t> &>(outBf16Reg));
            StoreAlign<uint16_t, DataCopyMode::DATA_BLOCK_COPY>(
                reinterpret_cast<__ubuf__ uint16_t *>(vBetaZnAddr) + znOffset, packedBf16Reg, dstFullRows, packedMask);
        }
    }
}

/*
 * Build a row range of the four BF16 Stage1 right-hand sides.  The sources are
 * row-major while the destinations retain the full-slice compact zN stride, so
 * disjoint input batches can fill one tile before its strided publication to L1.
 */
__simd_vf__ inline void PrepareStage1SliceVF(__ubuf__ bfloat16_t *qAddr, __ubuf__ bfloat16_t *kAddr,
                                             __ubuf__ bfloat16_t *vAddr, __ubuf__ bfloat16_t *rawKcdZnAddr,
                                             __ubuf__ bfloat16_t *vBetaZnAddr, __ubuf__ bfloat16_t *qPrimeZnAddr,
                                             __ubuf__ bfloat16_t *kgZnAddr, __ubuf__ float *betaAddr,
                                             __ubuf__ float *gCumAddr, __ubuf__ float *gCumExpAddr, float scale,
                                             uint16_t rowStart, uint16_t rowCount, uint16_t dk, uint16_t dv,
                                             uint16_t dkRowStride, uint16_t dvRowStride, uint16_t dstRowStart,
                                             uint16_t dstFullRows)
{
    constexpr uint32_t elementsPerRepeat = VECTOR_REG_WIDTH / sizeof(float);
    constexpr uint32_t elementsPerBlock = 32 / sizeof(bfloat16_t);
    static constexpr CastTrait castBf16ToFp32 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                 RoundMode::UNKNOWN};
    static constexpr CastTrait castFp32ToBf16 = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING,
                                                 RoundMode::CAST_RINT};
    RegTensor<bfloat16_t> qBf16Reg;
    RegTensor<bfloat16_t> kBf16Reg;
    RegTensor<bfloat16_t> vBf16Reg;
    RegTensor<bfloat16_t> outBf16Reg;
    RegTensor<uint16_t> packedBf16Reg;
    RegTensor<float> qReg;
    RegTensor<float> kReg;
    RegTensor<float> vReg;
    RegTensor<float> betaReg;
    RegTensor<float> gExpReg;
    RegTensor<float> decayReg;
    RegTensor<float> tmpReg;
    RegTensor<float> rawKcdReg;
    MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
    uint32_t packedCount = elementsPerRepeat;
    MaskReg packedMask = UpdateMask<uint16_t>(packedCount);

    DataCopy<float, LoadDist::DIST_BRC_B32>(decayReg, gCumAddr + 63);
    const uint16_t maxDim = dk > dv ? dk : dv;
    const uint16_t repeats = static_cast<uint16_t>((maxDim + elementsPerRepeat - 1) / elementsPerRepeat);
    for (uint16_t row = 0; row < rowCount; ++row) {
        DataCopy<float, LoadDist::DIST_BRC_B32>(betaReg, betaAddr + row);
        DataCopy<float, LoadDist::DIST_BRC_B32>(gExpReg, gCumExpAddr + rowStart + row);
        DataCopy<float, LoadDist::DIST_BRC_B32>(tmpReg, gCumAddr + rowStart + row);
        Sub(tmpReg, decayReg, tmpReg, fullMask);
        Exp<float, MaskMergeMode::ZEROING>(tmpReg, tmpReg, fullMask);

        for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
            const uint32_t featureBegin = static_cast<uint32_t>(repeat) * elementsPerRepeat;
            const uint32_t znOffset = static_cast<uint32_t>(repeat) * (elementsPerRepeat / elementsPerBlock) *
                                          dstFullRows * elementsPerBlock +
                                      static_cast<uint32_t>(dstRowStart + row) * elementsPerBlock;
            if (featureBegin < dk) {
                const uint32_t qkOffset = static_cast<uint32_t>(row) * dkRowStride + featureBegin;
                uint32_t qkRemaining = dk - featureBegin;
                MaskReg qkMask = UpdateMask<float>(qkRemaining);
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(qBf16Reg, qAddr + qkOffset);
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(kBf16Reg, kAddr + qkOffset);
                Cast<float, bfloat16_t, castBf16ToFp32>(qReg, qBf16Reg, qkMask);
                Cast<float, bfloat16_t, castBf16ToFp32>(kReg, kBf16Reg, qkMask);

                Mul(rawKcdReg, betaReg, gExpReg, qkMask);
                Muls(rawKcdReg, rawKcdReg, static_cast<float>(-1.0f), qkMask);
                Mul(rawKcdReg, kReg, rawKcdReg, qkMask);
                Cast<bfloat16_t, float, castFp32ToBf16>(outBf16Reg, rawKcdReg, qkMask);
                Pack<uint16_t, uint32_t, HighLowPart::LOWEST>(packedBf16Reg,
                                                              reinterpret_cast<RegTensor<uint32_t> &>(outBf16Reg));
                StoreAlign<uint16_t, DataCopyMode::DATA_BLOCK_COPY>(
                    reinterpret_cast<__ubuf__ uint16_t *>(rawKcdZnAddr) + znOffset, packedBf16Reg, dstFullRows,
                    packedMask);

                Mul(kReg, kReg, tmpReg, qkMask);
                Cast<bfloat16_t, float, castFp32ToBf16>(outBf16Reg, kReg, qkMask);
                Pack<uint16_t, uint32_t, HighLowPart::LOWEST>(packedBf16Reg,
                                                              reinterpret_cast<RegTensor<uint32_t> &>(outBf16Reg));
                StoreAlign<uint16_t, DataCopyMode::DATA_BLOCK_COPY>(
                    reinterpret_cast<__ubuf__ uint16_t *>(kgZnAddr) + znOffset, packedBf16Reg, dstFullRows, packedMask);

                Muls(qReg, qReg, scale, qkMask);
                Mul(qReg, qReg, gExpReg, qkMask);
                Cast<bfloat16_t, float, castFp32ToBf16>(outBf16Reg, qReg, qkMask);
                Pack<uint16_t, uint32_t, HighLowPart::LOWEST>(packedBf16Reg,
                                                              reinterpret_cast<RegTensor<uint32_t> &>(outBf16Reg));
                StoreAlign<uint16_t, DataCopyMode::DATA_BLOCK_COPY>(
                    reinterpret_cast<__ubuf__ uint16_t *>(qPrimeZnAddr) + znOffset, packedBf16Reg, dstFullRows,
                    packedMask);
            }

            if (featureBegin < dv) {
                const uint32_t vOffset = static_cast<uint32_t>(row) * dvRowStride + featureBegin;
                uint32_t vRemaining = dv - featureBegin;
                MaskReg vMask = UpdateMask<float>(vRemaining);
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(vBf16Reg, vAddr + vOffset);
                Cast<float, bfloat16_t, castBf16ToFp32>(vReg, vBf16Reg, vMask);
                Mul(vReg, vReg, betaReg, vMask);
                Cast<bfloat16_t, float, castFp32ToBf16>(outBf16Reg, vReg, vMask);
                Pack<uint16_t, uint32_t, HighLowPart::LOWEST>(packedBf16Reg,
                                                              reinterpret_cast<RegTensor<uint32_t> &>(outBf16Reg));
                StoreAlign<uint16_t, DataCopyMode::DATA_BLOCK_COPY>(
                    reinterpret_cast<__ubuf__ uint16_t *>(vBetaZnAddr) + znOffset, packedBf16Reg, dstFullRows,
                    packedMask);
            }
        }
    }
}

/*
 * Exact arch35 prefix postprocess.  One stable pairwise decay serves both
 * strict-lower KKT (inverse input) and lower-inclusive QKT (output attention).
 */
__simd_vf__ inline void BuildAttnMaskedQktExactSliceVF(__ubuf__ float *attnAddr, __ubuf__ float *inverseAddr,
                                                       __ubuf__ bfloat16_t *maskedQktZnAddr,
                                                       __ubuf__ bfloat16_t *kktAddr, __ubuf__ bfloat16_t *qktAddr,
                                                       __ubuf__ float *gCumAddr, __ubuf__ float *betaAddr, float scale,
                                                       uint16_t rowStart, uint16_t rowCount, uint16_t rowStride)
{
    static constexpr CastTrait castBf16ToFp32 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                 RoundMode::UNKNOWN};
    static constexpr CastTrait castFp32ToBf16 = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING,
                                                 RoundMode::CAST_RINT};
    RegTensor<int32_t> laneIndexReg;
    RegTensor<float> gColReg;
    RegTensor<float> gRowReg;
    RegTensor<float> decayReg;
    RegTensor<float> betaReg;
    RegTensor<float> kktFp32Reg;
    RegTensor<float> qktFp32Reg;
    RegTensor<float> attnReg;
    RegTensor<float> inverseInitReg;
    RegTensor<bfloat16_t> kktBf16Reg;
    RegTensor<bfloat16_t> qktBf16Reg;
    RegTensor<bfloat16_t> maskedQktBf16Reg;
    RegTensor<uint16_t> packedBf16Reg;
    uint32_t validCount = rowStride;
    uint32_t packedCount = rowStride;
    MaskReg validMask = UpdateMask<float>(validCount);
    MaskReg packedMask = UpdateMask<uint16_t>(packedCount);
    MaskReg strictLowerMask;
    MaskReg lowerInclusiveMask;

    LoadAlign(gColReg, gCumAddr);
    Arange(laneIndexReg, static_cast<int32_t>(0));
    for (uint16_t localRow = 0; localRow < rowCount; ++localRow) {
        const uint16_t row = rowStart + localRow;
        const uint32_t offset = static_cast<uint32_t>(localRow) * rowStride;
        Compares<int32_t, CMPMODE::LT>(strictLowerMask, laneIndexReg, static_cast<int32_t>(row), validMask);
        Compares<int32_t, CMPMODE::LT>(lowerInclusiveMask, laneIndexReg, static_cast<int32_t>(row + 1), validMask);
        DataCopy<float, LoadDist::DIST_BRC_B32>(gRowReg, gCumAddr + row);
        Sub<float, MaskMergeMode::ZEROING>(decayReg, gRowReg, gColReg, lowerInclusiveMask);
        Exp<float, MaskMergeMode::ZEROING>(decayReg, decayReg, validMask);

        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(kktBf16Reg, kktAddr + offset);
        Cast<float, bfloat16_t, castBf16ToFp32>(kktFp32Reg, kktBf16Reg, validMask);
        Mul<float, MaskMergeMode::ZEROING>(attnReg, kktFp32Reg, decayReg, strictLowerMask);
        DataCopy<float, LoadDist::DIST_BRC_B32>(betaReg, betaAddr + localRow);
        Mul(attnReg, attnReg, betaReg, validMask);
        Muls(inverseInitReg, attnReg, static_cast<float>(-1.0f), validMask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(attnAddr + offset, attnReg, validMask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(inverseAddr + offset, inverseInitReg, validMask);

        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(qktBf16Reg, qktAddr + offset);
        Cast<float, bfloat16_t, castBf16ToFp32>(qktFp32Reg, qktBf16Reg, validMask);
        Muls(qktFp32Reg, qktFp32Reg, scale, validMask);
        Mul<float, MaskMergeMode::ZEROING>(qktFp32Reg, qktFp32Reg, decayReg, lowerInclusiveMask);
        Cast<bfloat16_t, float, castFp32ToBf16>(maskedQktBf16Reg, qktFp32Reg, validMask);
        Pack<uint16_t, uint32_t, HighLowPart::LOWEST>(packedBf16Reg,
                                                      reinterpret_cast<RegTensor<uint32_t> &>(maskedQktBf16Reg));
        StoreAlign<uint16_t, DataCopyMode::DATA_BLOCK_COPY>(
            reinterpret_cast<__ubuf__ uint16_t *>(maskedQktZnAddr) + static_cast<uint32_t>(localRow) * 16,
            packedBf16Reg, rowCount, packedMask);
    }
}

/* Convert one row-major FP32 tile to BF16 while storing it directly as compact zN. */
__simd_vf__ inline void CastFp32ToCompactZnBf16VF(__ubuf__ float *srcAddr, __ubuf__ bfloat16_t *dstZnAddr,
                                                  uint16_t rowCount, uint16_t colCount)
{
    static constexpr CastTrait castFp32ToBf16 = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING,
                                                 RoundMode::CAST_RINT};
    RegTensor<float> srcReg;
    RegTensor<bfloat16_t> bf16Reg;
    RegTensor<uint16_t> packedBf16Reg;
    uint32_t validCount = colCount;
    uint32_t packedCount = colCount;
    MaskReg validMask = UpdateMask<float>(validCount);
    MaskReg packedMask = UpdateMask<uint16_t>(packedCount);

    for (uint16_t row = 0; row < rowCount; ++row) {
        LoadAlign(srcReg, srcAddr + static_cast<uint32_t>(row) * colCount);
        Cast<bfloat16_t, float, castFp32ToBf16>(bf16Reg, srcReg, validMask);
        Pack<uint16_t, uint32_t, HighLowPart::LOWEST>(packedBf16Reg, reinterpret_cast<RegTensor<uint32_t> &>(bf16Reg));
        StoreAlign<uint16_t, DataCopyMode::DATA_BLOCK_COPY>(
            reinterpret_cast<__ubuf__ uint16_t *>(dstZnAddr) + static_cast<uint32_t>(row) * 16, packedBf16Reg, rowCount,
            packedMask);
    }
}

/*!
 * InverseAIVVF: forward-substitution inverse of the N x N lower-triangular (unit-diagonal) attn block.
 *   inv[0] = e_0; for i=1..N-1: inv[i] = e_i - sum_{j<i} attn[i,j] * inv[j]
 * attnUb/invResUb layout: [halfChunkSize rows, chunkSize cols]; subBlock uses column offset = `offset`,
 *   i.e. element (i, col) is at base + offset + i*chunkSize + col. Only first N columns per row are valid.
 * eiUb layout: identity matrix, row i = e_i, row stride = chunkSize (no column offset).
 *   eiUb must be pre-filled by the caller (unit-diagonal identity in the first N rows).
 */
template <uint32_t N>
__simd_vf__ inline void InverseAIVVFImpl(__ubuf__ float *attnUb, __ubuf__ float *invResUb, __ubuf__ float *eiUb,
                                         uint32_t offset, uint32_t chunkSize)
{
    // N<=64, 单寄存器容纳一行; UpdateMask 生成前 N 个 lane 有效的掩码
    uint32_t maskLen = N;
    MaskReg maskN = UpdateMask<float>(maskLen);

    // inv[0] = e_0: 加载单位阵首行, 写入结果首行(offset 列偏移)
    RegTensor<float> inv0;
    LoadAlign(inv0, eiUb);
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
        // inv[i] = e_i - acc, 写回结果第 i 行
        RegTensor<float> ei_i;
        LoadAlign(ei_i, eiUb + i * chunkSize);
        RegTensor<float> invi;
        Sub(invi, ei_i, acc, maskN);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(invResUb + offset + i * chunkSize, invi, maskN);
        // store->load 屏障: 保证本行写入对后续行加载可见(下一轮 i 需读取 inv[i])
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
}

template <uint32_t N>
__aicore__ inline void InverseAIVVF(const LocalTensor<float> &attnUb, const LocalTensor<float> &invResUb,
                                    const LocalTensor<float> &eiUb, uint32_t offset, uint32_t chunkSize)
{
    __ubuf__ float *attn = reinterpret_cast<__ubuf__ float *>(attnUb.GetPhyAddr());
    __ubuf__ float *inv = reinterpret_cast<__ubuf__ float *>(invResUb.GetPhyAddr());
    __ubuf__ float *ei = reinterpret_cast<__ubuf__ float *>(eiUb.GetPhyAddr());
    InverseAIVVFImpl<N>(attn, inv, ei, offset, chunkSize);
}

/*
 * Fixed 32x32 exact-path inverse in eight-row panels. Compared with the four-row panel,
 * each previously solved inverse row is reused by twice as many accumulators before it
 * leaves the register file. The contribution order for every output row remains j=0..i-1.
 */
__simd_vf__ inline void InverseAIVPanel8VFImpl(__ubuf__ float *attnUb, __ubuf__ float *invResUb, __ubuf__ float *eiUb,
                                               uint32_t offset, uint32_t chunkSize)
{
    constexpr uint16_t inverseSize = 32;
    constexpr uint16_t panelRows = 8;
    uint32_t maskLen = inverseSize;
    MaskReg mask = UpdateMask<float>(maskLen);
    RegTensor<float> attnRow[panelRows];
    RegTensor<float> acc[panelRows];
    RegTensor<float> invRow[panelRows];
    RegTensor<float> previousInv;
    RegTensor<float> coefficient;
    RegTensor<uint32_t> index;

    for (uint16_t panel = 0; panel < inverseSize; panel += panelRows) {
        LoadAlign(attnRow[0], attnUb + offset + static_cast<uint32_t>(panel) * chunkSize);
        LoadAlign(attnRow[1], attnUb + offset + static_cast<uint32_t>(panel + 1) * chunkSize);
        LoadAlign(attnRow[2], attnUb + offset + static_cast<uint32_t>(panel + 2) * chunkSize);
        LoadAlign(attnRow[3], attnUb + offset + static_cast<uint32_t>(panel + 3) * chunkSize);
        LoadAlign(attnRow[4], attnUb + offset + static_cast<uint32_t>(panel + 4) * chunkSize);
        LoadAlign(attnRow[5], attnUb + offset + static_cast<uint32_t>(panel + 5) * chunkSize);
        LoadAlign(attnRow[6], attnUb + offset + static_cast<uint32_t>(panel + 6) * chunkSize);
        LoadAlign(attnRow[7], attnUb + offset + static_cast<uint32_t>(panel + 7) * chunkSize);
        Duplicate(acc[0], static_cast<float>(0.0f));
        Duplicate(acc[1], static_cast<float>(0.0f));
        Duplicate(acc[2], static_cast<float>(0.0f));
        Duplicate(acc[3], static_cast<float>(0.0f));
        Duplicate(acc[4], static_cast<float>(0.0f));
        Duplicate(acc[5], static_cast<float>(0.0f));
        Duplicate(acc[6], static_cast<float>(0.0f));
        Duplicate(acc[7], static_cast<float>(0.0f));

        for (uint16_t j = 0; j < panel; ++j) {
            LoadAlign(previousInv, invResUb + offset + static_cast<uint32_t>(j) * chunkSize);
            Duplicate(index, static_cast<uint32_t>(j));
            Gather(coefficient, attnRow[0], index);
            MulAddDst(acc[0], previousInv, coefficient, mask);
            Gather(coefficient, attnRow[1], index);
            MulAddDst(acc[1], previousInv, coefficient, mask);
            Gather(coefficient, attnRow[2], index);
            MulAddDst(acc[2], previousInv, coefficient, mask);
            Gather(coefficient, attnRow[3], index);
            MulAddDst(acc[3], previousInv, coefficient, mask);
            Gather(coefficient, attnRow[4], index);
            MulAddDst(acc[4], previousInv, coefficient, mask);
            Gather(coefficient, attnRow[5], index);
            MulAddDst(acc[5], previousInv, coefficient, mask);
            Gather(coefficient, attnRow[6], index);
            MulAddDst(acc[6], previousInv, coefficient, mask);
            Gather(coefficient, attnRow[7], index);
            MulAddDst(acc[7], previousInv, coefficient, mask);
        }

        LoadAlign(invRow[0], eiUb + static_cast<uint32_t>(panel) * chunkSize);
        Sub(invRow[0], invRow[0], acc[0], mask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(invResUb + offset + static_cast<uint32_t>(panel) * chunkSize,
                                                    invRow[0], mask);

        Duplicate(index, static_cast<uint32_t>(panel));
        Gather(coefficient, attnRow[1], index);
        MulAddDst(acc[1], invRow[0], coefficient, mask);
        LoadAlign(invRow[1], eiUb + static_cast<uint32_t>(panel + 1) * chunkSize);
        Sub(invRow[1], invRow[1], acc[1], mask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(invResUb + offset + static_cast<uint32_t>(panel + 1) * chunkSize,
                                                    invRow[1], mask);

        Duplicate(index, static_cast<uint32_t>(panel));
        Gather(coefficient, attnRow[2], index);
        MulAddDst(acc[2], invRow[0], coefficient, mask);
        Duplicate(index, static_cast<uint32_t>(panel + 1));
        Gather(coefficient, attnRow[2], index);
        MulAddDst(acc[2], invRow[1], coefficient, mask);
        LoadAlign(invRow[2], eiUb + static_cast<uint32_t>(panel + 2) * chunkSize);
        Sub(invRow[2], invRow[2], acc[2], mask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(invResUb + offset + static_cast<uint32_t>(panel + 2) * chunkSize,
                                                    invRow[2], mask);

        Duplicate(index, static_cast<uint32_t>(panel));
        Gather(coefficient, attnRow[3], index);
        MulAddDst(acc[3], invRow[0], coefficient, mask);
        Duplicate(index, static_cast<uint32_t>(panel + 1));
        Gather(coefficient, attnRow[3], index);
        MulAddDst(acc[3], invRow[1], coefficient, mask);
        Duplicate(index, static_cast<uint32_t>(panel + 2));
        Gather(coefficient, attnRow[3], index);
        MulAddDst(acc[3], invRow[2], coefficient, mask);
        LoadAlign(invRow[3], eiUb + static_cast<uint32_t>(panel + 3) * chunkSize);
        Sub(invRow[3], invRow[3], acc[3], mask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(invResUb + offset + static_cast<uint32_t>(panel + 3) * chunkSize,
                                                    invRow[3], mask);

        Duplicate(index, static_cast<uint32_t>(panel));
        Gather(coefficient, attnRow[4], index);
        MulAddDst(acc[4], invRow[0], coefficient, mask);
        Duplicate(index, static_cast<uint32_t>(panel + 1));
        Gather(coefficient, attnRow[4], index);
        MulAddDst(acc[4], invRow[1], coefficient, mask);
        Duplicate(index, static_cast<uint32_t>(panel + 2));
        Gather(coefficient, attnRow[4], index);
        MulAddDst(acc[4], invRow[2], coefficient, mask);
        Duplicate(index, static_cast<uint32_t>(panel + 3));
        Gather(coefficient, attnRow[4], index);
        MulAddDst(acc[4], invRow[3], coefficient, mask);
        LoadAlign(invRow[4], eiUb + static_cast<uint32_t>(panel + 4) * chunkSize);
        Sub(invRow[4], invRow[4], acc[4], mask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(invResUb + offset + static_cast<uint32_t>(panel + 4) * chunkSize,
                                                    invRow[4], mask);

        Duplicate(index, static_cast<uint32_t>(panel));
        Gather(coefficient, attnRow[5], index);
        MulAddDst(acc[5], invRow[0], coefficient, mask);
        Duplicate(index, static_cast<uint32_t>(panel + 1));
        Gather(coefficient, attnRow[5], index);
        MulAddDst(acc[5], invRow[1], coefficient, mask);
        Duplicate(index, static_cast<uint32_t>(panel + 2));
        Gather(coefficient, attnRow[5], index);
        MulAddDst(acc[5], invRow[2], coefficient, mask);
        Duplicate(index, static_cast<uint32_t>(panel + 3));
        Gather(coefficient, attnRow[5], index);
        MulAddDst(acc[5], invRow[3], coefficient, mask);
        Duplicate(index, static_cast<uint32_t>(panel + 4));
        Gather(coefficient, attnRow[5], index);
        MulAddDst(acc[5], invRow[4], coefficient, mask);
        LoadAlign(invRow[5], eiUb + static_cast<uint32_t>(panel + 5) * chunkSize);
        Sub(invRow[5], invRow[5], acc[5], mask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(invResUb + offset + static_cast<uint32_t>(panel + 5) * chunkSize,
                                                    invRow[5], mask);

        Duplicate(index, static_cast<uint32_t>(panel));
        Gather(coefficient, attnRow[6], index);
        MulAddDst(acc[6], invRow[0], coefficient, mask);
        Duplicate(index, static_cast<uint32_t>(panel + 1));
        Gather(coefficient, attnRow[6], index);
        MulAddDst(acc[6], invRow[1], coefficient, mask);
        Duplicate(index, static_cast<uint32_t>(panel + 2));
        Gather(coefficient, attnRow[6], index);
        MulAddDst(acc[6], invRow[2], coefficient, mask);
        Duplicate(index, static_cast<uint32_t>(panel + 3));
        Gather(coefficient, attnRow[6], index);
        MulAddDst(acc[6], invRow[3], coefficient, mask);
        Duplicate(index, static_cast<uint32_t>(panel + 4));
        Gather(coefficient, attnRow[6], index);
        MulAddDst(acc[6], invRow[4], coefficient, mask);
        Duplicate(index, static_cast<uint32_t>(panel + 5));
        Gather(coefficient, attnRow[6], index);
        MulAddDst(acc[6], invRow[5], coefficient, mask);
        LoadAlign(invRow[6], eiUb + static_cast<uint32_t>(panel + 6) * chunkSize);
        Sub(invRow[6], invRow[6], acc[6], mask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(invResUb + offset + static_cast<uint32_t>(panel + 6) * chunkSize,
                                                    invRow[6], mask);

        Duplicate(index, static_cast<uint32_t>(panel));
        Gather(coefficient, attnRow[7], index);
        MulAddDst(acc[7], invRow[0], coefficient, mask);
        Duplicate(index, static_cast<uint32_t>(panel + 1));
        Gather(coefficient, attnRow[7], index);
        MulAddDst(acc[7], invRow[1], coefficient, mask);
        Duplicate(index, static_cast<uint32_t>(panel + 2));
        Gather(coefficient, attnRow[7], index);
        MulAddDst(acc[7], invRow[2], coefficient, mask);
        Duplicate(index, static_cast<uint32_t>(panel + 3));
        Gather(coefficient, attnRow[7], index);
        MulAddDst(acc[7], invRow[3], coefficient, mask);
        Duplicate(index, static_cast<uint32_t>(panel + 4));
        Gather(coefficient, attnRow[7], index);
        MulAddDst(acc[7], invRow[4], coefficient, mask);
        Duplicate(index, static_cast<uint32_t>(panel + 5));
        Gather(coefficient, attnRow[7], index);
        MulAddDst(acc[7], invRow[5], coefficient, mask);
        Duplicate(index, static_cast<uint32_t>(panel + 6));
        Gather(coefficient, attnRow[7], index);
        MulAddDst(acc[7], invRow[6], coefficient, mask);
        LoadAlign(invRow[7], eiUb + static_cast<uint32_t>(panel + 7) * chunkSize);
        Sub(invRow[7], invRow[7], acc[7], mask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(invResUb + offset + static_cast<uint32_t>(panel + 7) * chunkSize,
                                                    invRow[7], mask);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
}
#if 0
/*
 * Solve two adjacent rows in one 64-lane accumulator. The even lanes belong to
 * the first row and the odd lanes to the second row. This preserves each row's
 * j-order while sharing one vector FMA for every coefficient pair.
 */
__simd_vf__ inline void InverseAIVPanel8PairVFImpl(
    __ubuf__ float *attnUb, __ubuf__ float *invResUb, __ubuf__ float *eiUb,
    uint32_t offset, uint32_t chunkSize)
{
    constexpr uint16_t inverseSize = 32;
    constexpr uint16_t panelRows = 8;
    constexpr uint16_t pairCount = panelRows / 2;
    uint32_t rowMaskLen = inverseSize;
    uint32_t pairMaskLen = inverseSize * 2;
    MaskReg rowMask = UpdateMask<float>(rowMaskLen);
    MaskReg pairMask = UpdateMask<float>(pairMaskLen);
    RegTensor<float> pairAcc[pairCount];
    RegTensor<float> invRow[panelRows];
    RegTensor<float> previousInv;
    RegTensor<float> pairedInv;
    RegTensor<float> coefficient0;
    RegTensor<float> coefficient1;
    RegTensor<float> pairedCoefficient;
    RegTensor<float> interleaveScratch;

#define CGDR_INVERSE_PREPARE_PAIRED_ROW(inverseRow) Interleave(pairedInv, interleaveScratch, (inverseRow), (inverseRow))
#define CGDR_INVERSE_PAIR_FMA(accumulator, row0, row1, col) \
    DataCopy<float, LoadDist::DIST_BRC_B32>(coefficient0, \
                                            attnUb + offset + static_cast<uint32_t>(row0) * chunkSize + (col)); \
    DataCopy<float, LoadDist::DIST_BRC_B32>(coefficient1, \
                                            attnUb + offset + static_cast<uint32_t>(row1) * chunkSize + (col)); \
    Interleave(pairedCoefficient, interleaveScratch, coefficient0, coefficient1); \
    MulAddDst((accumulator), pairedInv, pairedCoefficient, pairMask)
#define CGDR_INVERSE_ODD_FMA(accumulator, row, col, inverseRow) \
    DataCopy<float, LoadDist::DIST_BRC_B32>(pairedCoefficient, \
                                            attnUb + offset + static_cast<uint32_t>(row) * chunkSize + (col)); \
    MulAddDst((accumulator), (inverseRow), pairedCoefficient, rowMask)

    for (uint16_t panel = 0; panel < inverseSize; panel += panelRows) {
        Duplicate(pairAcc[0], static_cast<float>(0.0f));
        Duplicate(pairAcc[1], static_cast<float>(0.0f));
        Duplicate(pairAcc[2], static_cast<float>(0.0f));
        Duplicate(pairAcc[3], static_cast<float>(0.0f));

        for (uint16_t j = 0; j < panel; ++j) {
            LoadAlign(previousInv, invResUb + offset + static_cast<uint32_t>(j) * chunkSize);
            CGDR_INVERSE_PREPARE_PAIRED_ROW(previousInv);
            CGDR_INVERSE_PAIR_FMA(pairAcc[0], panel, panel + 1, j);
            CGDR_INVERSE_PAIR_FMA(pairAcc[1], panel + 2, panel + 3, j);
            CGDR_INVERSE_PAIR_FMA(pairAcc[2], panel + 4, panel + 5, j);
            CGDR_INVERSE_PAIR_FMA(pairAcc[3], panel + 6, panel + 7, j);
        }

        DeInterleave(coefficient0, coefficient1, pairAcc[0], pairAcc[0]);
        LoadAlign(invRow[0], eiUb + static_cast<uint32_t>(panel) * chunkSize);
        Sub(invRow[0], invRow[0], coefficient0, rowMask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(
            invResUb + offset + static_cast<uint32_t>(panel) * chunkSize, invRow[0], rowMask);

        CGDR_INVERSE_ODD_FMA(coefficient1, panel + 1, panel, invRow[0]);
        LoadAlign(invRow[1], eiUb + static_cast<uint32_t>(panel + 1) * chunkSize);
        Sub(invRow[1], invRow[1], coefficient1, rowMask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(
            invResUb + offset + static_cast<uint32_t>(panel + 1) * chunkSize, invRow[1], rowMask);

        CGDR_INVERSE_PREPARE_PAIRED_ROW(invRow[0]);
        CGDR_INVERSE_PAIR_FMA(pairAcc[1], panel + 2, panel + 3, panel);
        CGDR_INVERSE_PAIR_FMA(pairAcc[2], panel + 4, panel + 5, panel);
        CGDR_INVERSE_PAIR_FMA(pairAcc[3], panel + 6, panel + 7, panel);
        CGDR_INVERSE_PREPARE_PAIRED_ROW(invRow[1]);
        CGDR_INVERSE_PAIR_FMA(pairAcc[1], panel + 2, panel + 3, panel + 1);
        CGDR_INVERSE_PAIR_FMA(pairAcc[2], panel + 4, panel + 5, panel + 1);
        CGDR_INVERSE_PAIR_FMA(pairAcc[3], panel + 6, panel + 7, panel + 1);

        DeInterleave(coefficient0, coefficient1, pairAcc[1], pairAcc[1]);
        LoadAlign(invRow[2], eiUb + static_cast<uint32_t>(panel + 2) * chunkSize);
        Sub(invRow[2], invRow[2], coefficient0, rowMask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(
            invResUb + offset + static_cast<uint32_t>(panel + 2) * chunkSize, invRow[2], rowMask);

        CGDR_INVERSE_ODD_FMA(coefficient1, panel + 3, panel + 2, invRow[2]);
        LoadAlign(invRow[3], eiUb + static_cast<uint32_t>(panel + 3) * chunkSize);
        Sub(invRow[3], invRow[3], coefficient1, rowMask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(
            invResUb + offset + static_cast<uint32_t>(panel + 3) * chunkSize, invRow[3], rowMask);

        CGDR_INVERSE_PREPARE_PAIRED_ROW(invRow[2]);
        CGDR_INVERSE_PAIR_FMA(pairAcc[2], panel + 4, panel + 5, panel + 2);
        CGDR_INVERSE_PAIR_FMA(pairAcc[3], panel + 6, panel + 7, panel + 2);
        CGDR_INVERSE_PREPARE_PAIRED_ROW(invRow[3]);
        CGDR_INVERSE_PAIR_FMA(pairAcc[2], panel + 4, panel + 5, panel + 3);
        CGDR_INVERSE_PAIR_FMA(pairAcc[3], panel + 6, panel + 7, panel + 3);

        DeInterleave(coefficient0, coefficient1, pairAcc[2], pairAcc[2]);
        LoadAlign(invRow[4], eiUb + static_cast<uint32_t>(panel + 4) * chunkSize);
        Sub(invRow[4], invRow[4], coefficient0, rowMask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(
            invResUb + offset + static_cast<uint32_t>(panel + 4) * chunkSize, invRow[4], rowMask);

        CGDR_INVERSE_ODD_FMA(coefficient1, panel + 5, panel + 4, invRow[4]);
        LoadAlign(invRow[5], eiUb + static_cast<uint32_t>(panel + 5) * chunkSize);
        Sub(invRow[5], invRow[5], coefficient1, rowMask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(
            invResUb + offset + static_cast<uint32_t>(panel + 5) * chunkSize, invRow[5], rowMask);

        CGDR_INVERSE_PREPARE_PAIRED_ROW(invRow[4]);
        CGDR_INVERSE_PAIR_FMA(pairAcc[3], panel + 6, panel + 7, panel + 4);
        CGDR_INVERSE_PREPARE_PAIRED_ROW(invRow[5]);
        CGDR_INVERSE_PAIR_FMA(pairAcc[3], panel + 6, panel + 7, panel + 5);

        DeInterleave(coefficient0, coefficient1, pairAcc[3], pairAcc[3]);
        LoadAlign(invRow[6], eiUb + static_cast<uint32_t>(panel + 6) * chunkSize);
        Sub(invRow[6], invRow[6], coefficient0, rowMask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(
            invResUb + offset + static_cast<uint32_t>(panel + 6) * chunkSize, invRow[6], rowMask);

        CGDR_INVERSE_ODD_FMA(coefficient1, panel + 7, panel + 6, invRow[6]);
        LoadAlign(invRow[7], eiUb + static_cast<uint32_t>(panel + 7) * chunkSize);
        Sub(invRow[7], invRow[7], coefficient1, rowMask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(
            invResUb + offset + static_cast<uint32_t>(panel + 7) * chunkSize, invRow[7], rowMask);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }

#undef CGDR_INVERSE_ODD_FMA
#undef CGDR_INVERSE_PAIR_FMA
#undef CGDR_INVERSE_PREPARE_PAIRED_ROW
}
#endif

/*
 * Solve two adjacent target rows in one 64-lane accumulator.  A fixed
 * [0,1,0,1,...] register-gather index broadcasts both coefficients from one
 * panel-resident pair row, so common contributions need one Gather and one FMA.
 * Each logical row still receives j=0..i-1 in the original FP32 order.
 */
__simd_vf__ inline void InverseAIVPanel8PairAoSoAVFImpl(__ubuf__ float *pairAttnUb, __ubuf__ float *invResUb,
                                                        __ubuf__ float *eiUb, uint32_t offset, uint32_t chunkSize)
{
    constexpr uint16_t inverseSize = 32;
    constexpr uint16_t panelRows = 8;
    constexpr uint16_t panelPairs = panelRows / 2;
    uint32_t rowMaskLen = inverseSize;
    uint32_t pairMaskLen = inverseSize * 2;
    MaskReg rowMask = UpdateMask<float>(rowMaskLen);
    MaskReg pairMask = UpdateMask<float>(pairMaskLen);
    RegTensor<float> pairAcc[panelPairs];
    RegTensor<float> previousInv;
    RegTensor<float> pairedInv;
    RegTensor<float> interleaveScratch;
    RegTensor<float> pairedCoefficient0;
    RegTensor<float> pairedCoefficient1;
    RegTensor<float> pairedCoefficient2;
    RegTensor<float> pairedCoefficient3;
    RegTensor<float> evenAcc;
    RegTensor<float> oddAcc;
    RegTensor<float> evenInv;
    RegTensor<float> oddInv;
    RegTensor<float> oddCoefficient;
    RegTensor<uint32_t> parityIndex;
    RegTensor<uint32_t> coefficientIndex;
    RegTensor<uint32_t> one;

    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    Duplicate(one, static_cast<uint32_t>(1));
    Arange(reinterpret_cast<RegTensor<int32_t> &>(parityIndex), static_cast<int32_t>(0));
    And(parityIndex, parityIndex, one, pairMask);

#define CGDR_INVERSE_PAIR_PREPARE_ROW(inverseRow) Interleave(pairedInv, interleaveScratch, (inverseRow), (inverseRow))
#define CGDR_INVERSE_PAIR_PREPARE_INDEX(col) \
    Adds(coefficientIndex, parityIndex, static_cast<uint32_t>(2 * (col)), pairMask)
#define CGDR_INVERSE_PAIR_GATHER(coefficient, pairIndex) \
    Gather((coefficient), pairAttnUb + static_cast<uint32_t>(pairIndex) * inverseSize * 2, coefficientIndex, pairMask)
#define CGDR_INVERSE_PAIR_FMA(accumulator, coefficient) MulAddDst((accumulator), pairedInv, (coefficient), pairMask)

    for (uint16_t panel = 0; panel < inverseSize; panel += panelRows) {
        Duplicate(pairAcc[0], static_cast<float>(0.0f));
        Duplicate(pairAcc[1], static_cast<float>(0.0f));
        Duplicate(pairAcc[2], static_cast<float>(0.0f));
        Duplicate(pairAcc[3], static_cast<float>(0.0f));

        for (uint16_t j = 0; j < panel; ++j) {
            LoadAlign(previousInv, invResUb + offset + static_cast<uint32_t>(j) * chunkSize);
            CGDR_INVERSE_PAIR_PREPARE_ROW(previousInv);
            CGDR_INVERSE_PAIR_PREPARE_INDEX(j);
            CGDR_INVERSE_PAIR_GATHER(pairedCoefficient0, panel / 2);
            CGDR_INVERSE_PAIR_GATHER(pairedCoefficient1, panel / 2 + 1);
            CGDR_INVERSE_PAIR_GATHER(pairedCoefficient2, panel / 2 + 2);
            CGDR_INVERSE_PAIR_GATHER(pairedCoefficient3, panel / 2 + 3);
            CGDR_INVERSE_PAIR_FMA(pairAcc[0], pairedCoefficient0);
            CGDR_INVERSE_PAIR_FMA(pairAcc[1], pairedCoefficient1);
            CGDR_INVERSE_PAIR_FMA(pairAcc[2], pairedCoefficient2);
            CGDR_INVERSE_PAIR_FMA(pairAcc[3], pairedCoefficient3);
        }

        DeInterleave(evenAcc, oddAcc, pairAcc[0], pairAcc[0]);
        LoadAlign(evenInv, eiUb + static_cast<uint32_t>(panel) * chunkSize);
        Sub(evenInv, evenInv, evenAcc, rowMask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(invResUb + offset + static_cast<uint32_t>(panel) * chunkSize,
                                                    evenInv, rowMask);
        DataCopy<float, LoadDist::DIST_BRC_B32>(
            oddCoefficient,
            pairAttnUb + static_cast<uint32_t>(panel / 2) * inverseSize * 2 + static_cast<uint32_t>(2 * panel + 1));
        MulAddDst(oddAcc, evenInv, oddCoefficient, rowMask);
        LoadAlign(oddInv, eiUb + static_cast<uint32_t>(panel + 1) * chunkSize);
        Sub(oddInv, oddInv, oddAcc, rowMask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(invResUb + offset + static_cast<uint32_t>(panel + 1) * chunkSize,
                                                    oddInv, rowMask);
        CGDR_INVERSE_PAIR_PREPARE_ROW(evenInv);
        CGDR_INVERSE_PAIR_PREPARE_INDEX(panel);
        CGDR_INVERSE_PAIR_GATHER(pairedCoefficient0, panel / 2 + 1);
        CGDR_INVERSE_PAIR_GATHER(pairedCoefficient1, panel / 2 + 2);
        CGDR_INVERSE_PAIR_GATHER(pairedCoefficient2, panel / 2 + 3);
        CGDR_INVERSE_PAIR_FMA(pairAcc[1], pairedCoefficient0);
        CGDR_INVERSE_PAIR_FMA(pairAcc[2], pairedCoefficient1);
        CGDR_INVERSE_PAIR_FMA(pairAcc[3], pairedCoefficient2);
        CGDR_INVERSE_PAIR_PREPARE_ROW(oddInv);
        CGDR_INVERSE_PAIR_PREPARE_INDEX(panel + 1);
        CGDR_INVERSE_PAIR_GATHER(pairedCoefficient0, panel / 2 + 1);
        CGDR_INVERSE_PAIR_GATHER(pairedCoefficient1, panel / 2 + 2);
        CGDR_INVERSE_PAIR_GATHER(pairedCoefficient2, panel / 2 + 3);
        CGDR_INVERSE_PAIR_FMA(pairAcc[1], pairedCoefficient0);
        CGDR_INVERSE_PAIR_FMA(pairAcc[2], pairedCoefficient1);
        CGDR_INVERSE_PAIR_FMA(pairAcc[3], pairedCoefficient2);

        DeInterleave(evenAcc, oddAcc, pairAcc[1], pairAcc[1]);
        LoadAlign(evenInv, eiUb + static_cast<uint32_t>(panel + 2) * chunkSize);
        Sub(evenInv, evenInv, evenAcc, rowMask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(invResUb + offset + static_cast<uint32_t>(panel + 2) * chunkSize,
                                                    evenInv, rowMask);
        DataCopy<float, LoadDist::DIST_BRC_B32>(oddCoefficient,
                                                pairAttnUb + static_cast<uint32_t>(panel / 2 + 1) * inverseSize * 2 +
                                                    static_cast<uint32_t>(2 * (panel + 2) + 1));
        MulAddDst(oddAcc, evenInv, oddCoefficient, rowMask);
        LoadAlign(oddInv, eiUb + static_cast<uint32_t>(panel + 3) * chunkSize);
        Sub(oddInv, oddInv, oddAcc, rowMask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(invResUb + offset + static_cast<uint32_t>(panel + 3) * chunkSize,
                                                    oddInv, rowMask);
        CGDR_INVERSE_PAIR_PREPARE_ROW(evenInv);
        CGDR_INVERSE_PAIR_PREPARE_INDEX(panel + 2);
        CGDR_INVERSE_PAIR_GATHER(pairedCoefficient0, panel / 2 + 2);
        CGDR_INVERSE_PAIR_GATHER(pairedCoefficient1, panel / 2 + 3);
        CGDR_INVERSE_PAIR_FMA(pairAcc[2], pairedCoefficient0);
        CGDR_INVERSE_PAIR_FMA(pairAcc[3], pairedCoefficient1);
        CGDR_INVERSE_PAIR_PREPARE_ROW(oddInv);
        CGDR_INVERSE_PAIR_PREPARE_INDEX(panel + 3);
        CGDR_INVERSE_PAIR_GATHER(pairedCoefficient0, panel / 2 + 2);
        CGDR_INVERSE_PAIR_GATHER(pairedCoefficient1, panel / 2 + 3);
        CGDR_INVERSE_PAIR_FMA(pairAcc[2], pairedCoefficient0);
        CGDR_INVERSE_PAIR_FMA(pairAcc[3], pairedCoefficient1);

        DeInterleave(evenAcc, oddAcc, pairAcc[2], pairAcc[2]);
        LoadAlign(evenInv, eiUb + static_cast<uint32_t>(panel + 4) * chunkSize);
        Sub(evenInv, evenInv, evenAcc, rowMask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(invResUb + offset + static_cast<uint32_t>(panel + 4) * chunkSize,
                                                    evenInv, rowMask);
        DataCopy<float, LoadDist::DIST_BRC_B32>(oddCoefficient,
                                                pairAttnUb + static_cast<uint32_t>(panel / 2 + 2) * inverseSize * 2 +
                                                    static_cast<uint32_t>(2 * (panel + 4) + 1));
        MulAddDst(oddAcc, evenInv, oddCoefficient, rowMask);
        LoadAlign(oddInv, eiUb + static_cast<uint32_t>(panel + 5) * chunkSize);
        Sub(oddInv, oddInv, oddAcc, rowMask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(invResUb + offset + static_cast<uint32_t>(panel + 5) * chunkSize,
                                                    oddInv, rowMask);
        CGDR_INVERSE_PAIR_PREPARE_ROW(evenInv);
        CGDR_INVERSE_PAIR_PREPARE_INDEX(panel + 4);
        CGDR_INVERSE_PAIR_GATHER(pairedCoefficient0, panel / 2 + 3);
        CGDR_INVERSE_PAIR_FMA(pairAcc[3], pairedCoefficient0);
        CGDR_INVERSE_PAIR_PREPARE_ROW(oddInv);
        CGDR_INVERSE_PAIR_PREPARE_INDEX(panel + 5);
        CGDR_INVERSE_PAIR_GATHER(pairedCoefficient0, panel / 2 + 3);
        CGDR_INVERSE_PAIR_FMA(pairAcc[3], pairedCoefficient0);

        DeInterleave(evenAcc, oddAcc, pairAcc[3], pairAcc[3]);
        LoadAlign(evenInv, eiUb + static_cast<uint32_t>(panel + 6) * chunkSize);
        Sub(evenInv, evenInv, evenAcc, rowMask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(invResUb + offset + static_cast<uint32_t>(panel + 6) * chunkSize,
                                                    evenInv, rowMask);
        DataCopy<float, LoadDist::DIST_BRC_B32>(oddCoefficient,
                                                pairAttnUb + static_cast<uint32_t>(panel / 2 + 3) * inverseSize * 2 +
                                                    static_cast<uint32_t>(2 * (panel + 6) + 1));
        MulAddDst(oddAcc, evenInv, oddCoefficient, rowMask);
        LoadAlign(oddInv, eiUb + static_cast<uint32_t>(panel + 7) * chunkSize);
        Sub(oddInv, oddInv, oddAcc, rowMask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(invResUb + offset + static_cast<uint32_t>(panel + 7) * chunkSize,
                                                    oddInv, rowMask);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }

#undef CGDR_INVERSE_PAIR_FMA
#undef CGDR_INVERSE_PAIR_GATHER
#undef CGDR_INVERSE_PAIR_PREPARE_INDEX
#undef CGDR_INVERSE_PAIR_PREPARE_ROW
}

/* Read each scalar coefficient directly from UB with B32 broadcast. */
__simd_vf__ inline void InverseAIVPanel8DirectBrcVFImpl(__ubuf__ float *attnUb, __ubuf__ float *invResUb,
                                                        __ubuf__ float *eiUb, uint32_t offset, uint32_t chunkSize)
{
    constexpr uint16_t inverseSize = 32;
    constexpr uint16_t panelRows = 8;
    uint32_t maskLen = inverseSize;
    MaskReg mask = UpdateMask<float>(maskLen);
    RegTensor<float> acc[panelRows];
    RegTensor<float> invRow[panelRows];
    RegTensor<float> previousInv;
    RegTensor<float> coefficient;

#define CGDR_INVERSE_DIRECT_FMA(accumulator, row, col, inverseRow) \
    DataCopy<float, LoadDist::DIST_BRC_B32>(coefficient, \
                                            attnUb + offset + static_cast<uint32_t>(row) * chunkSize + (col)); \
    MulAddDst((accumulator), (inverseRow), coefficient, mask)

    for (uint16_t panel = 0; panel < inverseSize; panel += panelRows) {
        Duplicate(acc[0], static_cast<float>(0.0f));
        Duplicate(acc[1], static_cast<float>(0.0f));
        Duplicate(acc[2], static_cast<float>(0.0f));
        Duplicate(acc[3], static_cast<float>(0.0f));
        Duplicate(acc[4], static_cast<float>(0.0f));
        Duplicate(acc[5], static_cast<float>(0.0f));
        Duplicate(acc[6], static_cast<float>(0.0f));
        Duplicate(acc[7], static_cast<float>(0.0f));

        for (uint16_t j = 0; j < panel; ++j) {
            LoadAlign(previousInv, invResUb + offset + static_cast<uint32_t>(j) * chunkSize);
            CGDR_INVERSE_DIRECT_FMA(acc[0], panel, j, previousInv);
            CGDR_INVERSE_DIRECT_FMA(acc[1], panel + 1, j, previousInv);
            CGDR_INVERSE_DIRECT_FMA(acc[2], panel + 2, j, previousInv);
            CGDR_INVERSE_DIRECT_FMA(acc[3], panel + 3, j, previousInv);
            CGDR_INVERSE_DIRECT_FMA(acc[4], panel + 4, j, previousInv);
            CGDR_INVERSE_DIRECT_FMA(acc[5], panel + 5, j, previousInv);
            CGDR_INVERSE_DIRECT_FMA(acc[6], panel + 6, j, previousInv);
            CGDR_INVERSE_DIRECT_FMA(acc[7], panel + 7, j, previousInv);
        }

        LoadAlign(invRow[0], eiUb + static_cast<uint32_t>(panel) * chunkSize);
        Sub(invRow[0], invRow[0], acc[0], mask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(invResUb + offset + static_cast<uint32_t>(panel) * chunkSize,
                                                    invRow[0], mask);

        CGDR_INVERSE_DIRECT_FMA(acc[1], panel + 1, panel, invRow[0]);
        LoadAlign(invRow[1], eiUb + static_cast<uint32_t>(panel + 1) * chunkSize);
        Sub(invRow[1], invRow[1], acc[1], mask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(invResUb + offset + static_cast<uint32_t>(panel + 1) * chunkSize,
                                                    invRow[1], mask);

        CGDR_INVERSE_DIRECT_FMA(acc[2], panel + 2, panel, invRow[0]);
        CGDR_INVERSE_DIRECT_FMA(acc[2], panel + 2, panel + 1, invRow[1]);
        LoadAlign(invRow[2], eiUb + static_cast<uint32_t>(panel + 2) * chunkSize);
        Sub(invRow[2], invRow[2], acc[2], mask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(invResUb + offset + static_cast<uint32_t>(panel + 2) * chunkSize,
                                                    invRow[2], mask);

        CGDR_INVERSE_DIRECT_FMA(acc[3], panel + 3, panel, invRow[0]);
        CGDR_INVERSE_DIRECT_FMA(acc[3], panel + 3, panel + 1, invRow[1]);
        CGDR_INVERSE_DIRECT_FMA(acc[3], panel + 3, panel + 2, invRow[2]);
        LoadAlign(invRow[3], eiUb + static_cast<uint32_t>(panel + 3) * chunkSize);
        Sub(invRow[3], invRow[3], acc[3], mask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(invResUb + offset + static_cast<uint32_t>(panel + 3) * chunkSize,
                                                    invRow[3], mask);

        CGDR_INVERSE_DIRECT_FMA(acc[4], panel + 4, panel, invRow[0]);
        CGDR_INVERSE_DIRECT_FMA(acc[4], panel + 4, panel + 1, invRow[1]);
        CGDR_INVERSE_DIRECT_FMA(acc[4], panel + 4, panel + 2, invRow[2]);
        CGDR_INVERSE_DIRECT_FMA(acc[4], panel + 4, panel + 3, invRow[3]);
        LoadAlign(invRow[4], eiUb + static_cast<uint32_t>(panel + 4) * chunkSize);
        Sub(invRow[4], invRow[4], acc[4], mask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(invResUb + offset + static_cast<uint32_t>(panel + 4) * chunkSize,
                                                    invRow[4], mask);

        CGDR_INVERSE_DIRECT_FMA(acc[5], panel + 5, panel, invRow[0]);
        CGDR_INVERSE_DIRECT_FMA(acc[5], panel + 5, panel + 1, invRow[1]);
        CGDR_INVERSE_DIRECT_FMA(acc[5], panel + 5, panel + 2, invRow[2]);
        CGDR_INVERSE_DIRECT_FMA(acc[5], panel + 5, panel + 3, invRow[3]);
        CGDR_INVERSE_DIRECT_FMA(acc[5], panel + 5, panel + 4, invRow[4]);
        LoadAlign(invRow[5], eiUb + static_cast<uint32_t>(panel + 5) * chunkSize);
        Sub(invRow[5], invRow[5], acc[5], mask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(invResUb + offset + static_cast<uint32_t>(panel + 5) * chunkSize,
                                                    invRow[5], mask);

        CGDR_INVERSE_DIRECT_FMA(acc[6], panel + 6, panel, invRow[0]);
        CGDR_INVERSE_DIRECT_FMA(acc[6], panel + 6, panel + 1, invRow[1]);
        CGDR_INVERSE_DIRECT_FMA(acc[6], panel + 6, panel + 2, invRow[2]);
        CGDR_INVERSE_DIRECT_FMA(acc[6], panel + 6, panel + 3, invRow[3]);
        CGDR_INVERSE_DIRECT_FMA(acc[6], panel + 6, panel + 4, invRow[4]);
        CGDR_INVERSE_DIRECT_FMA(acc[6], panel + 6, panel + 5, invRow[5]);
        LoadAlign(invRow[6], eiUb + static_cast<uint32_t>(panel + 6) * chunkSize);
        Sub(invRow[6], invRow[6], acc[6], mask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(invResUb + offset + static_cast<uint32_t>(panel + 6) * chunkSize,
                                                    invRow[6], mask);

        CGDR_INVERSE_DIRECT_FMA(acc[7], panel + 7, panel, invRow[0]);
        CGDR_INVERSE_DIRECT_FMA(acc[7], panel + 7, panel + 1, invRow[1]);
        CGDR_INVERSE_DIRECT_FMA(acc[7], panel + 7, panel + 2, invRow[2]);
        CGDR_INVERSE_DIRECT_FMA(acc[7], panel + 7, panel + 3, invRow[3]);
        CGDR_INVERSE_DIRECT_FMA(acc[7], panel + 7, panel + 4, invRow[4]);
        CGDR_INVERSE_DIRECT_FMA(acc[7], panel + 7, panel + 5, invRow[5]);
        CGDR_INVERSE_DIRECT_FMA(acc[7], panel + 7, panel + 6, invRow[6]);
        LoadAlign(invRow[7], eiUb + static_cast<uint32_t>(panel + 7) * chunkSize);
        Sub(invRow[7], invRow[7], acc[7], mask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(invResUb + offset + static_cast<uint32_t>(panel + 7) * chunkSize,
                                                    invRow[7], mask);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }

#undef CGDR_INVERSE_DIRECT_FMA
}

__aicore__ inline void InverseAIVPanel8VF(const LocalTensor<float> &attnUb, const LocalTensor<float> &invResUb,
                                          const LocalTensor<float> &eiUb, uint32_t offset, uint32_t chunkSize)
{
    auto attn = reinterpret_cast<__ubuf__ float *>(attnUb.GetPhyAddr());
    auto inv = reinterpret_cast<__ubuf__ float *>(invResUb.GetPhyAddr());
    auto ei = reinterpret_cast<__ubuf__ float *>(eiUb.GetPhyAddr());
    InverseAIVPanel8DirectBrcVFImpl(attn, inv, ei, offset, chunkSize);
}
} // namespace ChunkGatedDeltaRule
#endif // CHUNK_GATED_DELTA_RULE_STAGE1_VF_H
