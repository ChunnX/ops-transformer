/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "basic_api/kernel_tensor.h"

namespace ChunkGatedDeltaRule {
using namespace AscendC;
using namespace Reg;

/*
 * 备用，当前无调用：保留显式生成 Gamma:[C,C] 的方案供后续实验对照。
 * 当前使用 CumSumExpExactVF 和分数后处理按行生成门控系数，避免 Gamma 的 UB 落地。
 * 按传入的 chunkSize 计算 FP32 inclusive cumsum、exp 和 Gamma：
 *   gCum[i] = sum(g[0:i + 1])
 *   gExp[i] = exp(gCum[i])
 *   gamma[i, j] = j < i ? exp(gCum[i] - gCum[j]) : 0
 * gCum 计算完后保留在寄存器中直接生成 Gamma，避免再次从 UB 加载以及两次 Broadcast。
 */
__simd_vf__ inline void CumSumExpVF(__ubuf__ float* gAddr, __ubuf__ float* gCumAddr, __ubuf__ float* gExpAddr,
                                    __ubuf__ float* gammaAddr, uint32_t chunkSize)
{
    // chunkSize <= 单个 FP32 Vector 寄存器的元素数，因此一个寄存器可保存整个 chunk 的 g/gCum
    RegTensor<float> gCumReg;
    RegTensor<float> shiftedReg;
    RegTensor<float> gExpReg;
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
        Gather<float, uint32_t>(shiftedReg, gCumReg, reinterpret_cast<RegTensor<uint32_t>&>(gatherIndexReg));
        Compares<int32_t, CMPMODE::GE>(addMask, laneIndexReg, stride, validMask);
        Add<float, MaskMergeMode::MERGING>(gCumReg, gCumReg, shiftedReg, addMask);
    }

    Exp<float, MaskMergeMode::ZEROING>(gExpReg, gCumReg, validMask);
    StoreAlign<float, StoreDist::DIST_NORM>(gCumAddr, gCumReg, validMask);
    StoreAlign<float, StoreDist::DIST_NORM>(gExpAddr, gExpReg, validMask);

    // stageOneMask 是严格下三角矩阵。直接用 laneIndexReg < row 生成行 mask，
    // 在 Gamma 矩阵中写出对角线及上三角为 0，不再从 GM 搬运 mask 或执行两次 Mul。
    Duplicate(gExpReg, static_cast<float>(0.0));
    StoreAlign<float, StoreDist::DIST_NORM_B32>(gammaAddr, gExpReg, validMask);
    for (uint16_t row = 1; row < static_cast<uint16_t>(chunkSize); ++row) {
        uint32_t rowOffset = static_cast<uint32_t>(row) * chunkSize;
        Duplicate(reinterpret_cast<RegTensor<uint32_t>&>(gatherIndexReg), static_cast<uint32_t>(row));
        Gather<float, uint32_t>(shiftedReg, gCumReg, reinterpret_cast<RegTensor<uint32_t>&>(gatherIndexReg));
        Compares<int32_t, CMPMODE::LT>(gammaMask, laneIndexReg, static_cast<int32_t>(row), validMask);
        Sub<float, MaskMergeMode::ZEROING>(gExpReg, shiftedReg, gCumReg, gammaMask);
        Exp<float, MaskMergeMode::ZEROING>(gExpReg, gExpReg, gammaMask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(gammaAddr + rowOffset, gExpReg, validMask);
    }
}

/*
 * [2a] G=cumsum(g)，expG=exp(G)，逻辑 shape 均为 [C]，物理 C=64。
 * 融合路径中两个 AIV 各自保留完整 64 元素的 G 和 exp(G)，以少量重复计算消除跨 AIV 的前缀和传递。
 * 使用寄存器内 Hillis-Steele 扫描，步长 1/2/4/8/16/32；Gamma 不在此物化，由分数后处理按行生成。
 */
__simd_vf__ inline void CumSumExpExactVF(__ubuf__ float* gAddr, __ubuf__ float* gCumAddr, __ubuf__ float* gExpAddr,
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
        Gather<float, uint32_t>(shiftedReg, gCumReg, reinterpret_cast<RegTensor<uint32_t>&>(gatherIndexReg));
        Compares<int32_t, CMPMODE::GE>(addMask, laneIndexReg, stride, validMask);
        Add<float, MaskMergeMode::MERGING>(gCumReg, gCumReg, shiftedReg, addMask);
    }

    Exp<float, MaskMergeMode::ZEROING>(resultReg, gCumReg, validMask);
    StoreAlign<float, StoreDist::DIST_NORM>(gCumAddr, gCumReg, validMask);
    StoreAlign<float, StoreDist::DIST_NORM>(gExpAddr, resultReg, validMask);
}

/*
 * [2b] ComputeGatedChunkOperandRows 调用的 Dk=Dv=128 常用维度特化。
 * 同一固定宽度循环加载 Q/K/V，在 FP32 寄存器中生成四个 BF16 矩阵：
 * rawKcd=-beta*exp(G)*K，vBeta=beta*V，qPrime=scale*exp(G)*Q，kg=exp(G_last-G)*K。
 * 缩放、类型转换和 compact zN 写出融合在一次 VF 内，省去多个中间 UB 张量和独立的格式转换。
 */
__simd_vf__ inline void PrepareStage1FullDimSliceVF(__ubuf__ bfloat16_t* qAddr, __ubuf__ bfloat16_t* kAddr,
                                                    __ubuf__ bfloat16_t* vAddr, __ubuf__ bfloat16_t* rawKcdZnAddr,
                                                    __ubuf__ bfloat16_t* vBetaZnAddr, __ubuf__ bfloat16_t* qPrimeZnAddr,
                                                    __ubuf__ bfloat16_t* kgZnAddr, __ubuf__ float* betaAddr,
                                                    __ubuf__ float* gCumAddr, __ubuf__ float* gExpAddr, float scale,
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
        DataCopy<float, LoadDist::DIST_BRC_B32>(gExpReg, gExpAddr + rowStart + row);
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

            // [2b] rawKcd_i=-beta_i*exp(G_i)*K_i；负号用于 [6a] 的残差相加。
            Mul(rawKcdReg, betaReg, gExpReg, mask);
            Muls(rawKcdReg, rawKcdReg, static_cast<float>(-1.0f), mask);
            Mul(rawKcdReg, kReg, rawKcdReg, mask);
            Cast<bfloat16_t, float, castFp32ToBf16>(outBf16Reg, rawKcdReg, mask);
            Pack<uint16_t, uint32_t, HighLowPart::LOWEST>(packedBf16Reg,
                                                          reinterpret_cast<RegTensor<uint32_t>&>(outBf16Reg));
            StoreAlign<uint16_t, DataCopyMode::DATA_BLOCK_COPY>(
                reinterpret_cast<__ubuf__ uint16_t*>(rawKcdZnAddr) + znOffset, packedBf16Reg, dstFullRows, packedMask);

            // [2b] Kg_i=exp(G_last-G_i)*K_i，供 [7] 累积块末状态增量。
            Mul(kReg, kReg, tmpReg, mask);
            Cast<bfloat16_t, float, castFp32ToBf16>(outBf16Reg, kReg, mask);
            Pack<uint16_t, uint32_t, HighLowPart::LOWEST>(packedBf16Reg,
                                                          reinterpret_cast<RegTensor<uint32_t>&>(outBf16Reg));
            StoreAlign<uint16_t, DataCopyMode::DATA_BLOCK_COPY>(
                reinterpret_cast<__ubuf__ uint16_t*>(kgZnAddr) + znOffset, packedBf16Reg, dstFullRows, packedMask);

            // [2b] Q'_i=scale*exp(G_i)*Q_i，供 [6b] 计算初态对输出的贡献。
            Muls(qReg, qReg, scale, mask);
            Mul(qReg, qReg, gExpReg, mask);
            Cast<bfloat16_t, float, castFp32ToBf16>(outBf16Reg, qReg, mask);
            Pack<uint16_t, uint32_t, HighLowPart::LOWEST>(packedBf16Reg,
                                                          reinterpret_cast<RegTensor<uint32_t>&>(outBf16Reg));
            StoreAlign<uint16_t, DataCopyMode::DATA_BLOCK_COPY>(
                reinterpret_cast<__ubuf__ uint16_t*>(qPrimeZnAddr) + znOffset, packedBf16Reg, dstFullRows, packedMask);

            // [2b] vBeta_i=beta_i*V_i，供 [5b] 计算 U。
            Mul(vReg, vReg, betaReg, mask);
            Cast<bfloat16_t, float, castFp32ToBf16>(outBf16Reg, vReg, mask);
            Pack<uint16_t, uint32_t, HighLowPart::LOWEST>(packedBf16Reg,
                                                          reinterpret_cast<RegTensor<uint32_t>&>(outBf16Reg));
            StoreAlign<uint16_t, DataCopyMode::DATA_BLOCK_COPY>(
                reinterpret_cast<__ubuf__ uint16_t*>(vBetaZnAddr) + znOffset, packedBf16Reg, dstFullRows, packedMask);
        }
    }
}

/*
 * [2b] 上述四个矩阵的通用维度版本：Q/K 与 V 分别按 Dk/Dv 建掩码，只处理有效特征。
 * 输入为行主序，输出直接按整个 32 行切片的 compact zN 步长写入；两次 16 行输入可填充同一输出 tile。
 * 无效行列由调用方预先清零，避免固定 128 维的后续 Cube 计算读到旧数据。
 */
__simd_vf__ inline void PrepareStage1SliceVF(__ubuf__ bfloat16_t* qAddr, __ubuf__ bfloat16_t* kAddr,
                                             __ubuf__ bfloat16_t* vAddr, __ubuf__ bfloat16_t* rawKcdZnAddr,
                                             __ubuf__ bfloat16_t* vBetaZnAddr, __ubuf__ bfloat16_t* qPrimeZnAddr,
                                             __ubuf__ bfloat16_t* kgZnAddr, __ubuf__ float* betaAddr,
                                             __ubuf__ float* gCumAddr, __ubuf__ float* gExpAddr, float scale,
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
        DataCopy<float, LoadDist::DIST_BRC_B32>(gExpReg, gExpAddr + rowStart + row);
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
                                                              reinterpret_cast<RegTensor<uint32_t>&>(outBf16Reg));
                StoreAlign<uint16_t, DataCopyMode::DATA_BLOCK_COPY>(
                    reinterpret_cast<__ubuf__ uint16_t*>(rawKcdZnAddr) + znOffset, packedBf16Reg, dstFullRows,
                    packedMask);

                Mul(kReg, kReg, tmpReg, qkMask);
                Cast<bfloat16_t, float, castFp32ToBf16>(outBf16Reg, kReg, qkMask);
                Pack<uint16_t, uint32_t, HighLowPart::LOWEST>(packedBf16Reg,
                                                              reinterpret_cast<RegTensor<uint32_t>&>(outBf16Reg));
                StoreAlign<uint16_t, DataCopyMode::DATA_BLOCK_COPY>(
                    reinterpret_cast<__ubuf__ uint16_t*>(kgZnAddr) + znOffset, packedBf16Reg, dstFullRows, packedMask);

                Muls(qReg, qReg, scale, qkMask);
                Mul(qReg, qReg, gExpReg, qkMask);
                Cast<bfloat16_t, float, castFp32ToBf16>(outBf16Reg, qReg, qkMask);
                Pack<uint16_t, uint32_t, HighLowPart::LOWEST>(packedBf16Reg,
                                                              reinterpret_cast<RegTensor<uint32_t>&>(outBf16Reg));
                StoreAlign<uint16_t, DataCopyMode::DATA_BLOCK_COPY>(
                    reinterpret_cast<__ubuf__ uint16_t*>(qPrimeZnAddr) + znOffset, packedBf16Reg, dstFullRows,
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
                                                              reinterpret_cast<RegTensor<uint32_t>&>(outBf16Reg));
                StoreAlign<uint16_t, DataCopyMode::DATA_BLOCK_COPY>(
                    reinterpret_cast<__ubuf__ uint16_t*>(vBetaZnAddr) + znOffset, packedBf16Reg, dstFullRows,
                    packedMask);
            }
        }
    }
}

/*
 * [3a] L_ij=beta_i*KKT_ij*exp(G_i-G_j) (j<i)，M_ij=scale*QKT_ij*exp(G_i-G_j) (j<=i)。
 * 每个 AIV 生成 L/M:[32,64]；lowerAddr 存 L，inverseAddr 暂存 -L，maskedQktZnAddr 存 M。
 * 一次计算 exp(G_i-G_j)，同时用于严格下三角 KKT（L，求逆输入）和含对角线的 QKT（输出注意力）。
 * 前者必须排除当前 token，后者必须包含当前 token；两种 mask 不可混用。
 * 直接对前缀和之差取 exp，避免先算 exp(G_i)/exp(G_j) 时的小数相除；上三角在取 exp 前置零，
 * 随后通过乘法掩码清零。inverse 初始化为 -L，两个对角块稍后被各自的 (I+L_diag)^(-1) 覆盖。
 */
__simd_vf__ inline void BuildLowerAndMaskedQktExactSliceVF(__ubuf__ float* lowerAddr, __ubuf__ float* inverseAddr,
                                                           __ubuf__ bfloat16_t* maskedQktZnAddr,
                                                           __ubuf__ bfloat16_t* kktAddr, __ubuf__ bfloat16_t* qktAddr,
                                                           __ubuf__ float* gCumAddr, __ubuf__ float* betaAddr,
                                                           float scale, uint16_t rowStart, uint16_t rowCount,
                                                           uint16_t rowStride)
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
    RegTensor<float> lowerReg;
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
        Mul<float, MaskMergeMode::ZEROING>(lowerReg, kktFp32Reg, decayReg, strictLowerMask);
        DataCopy<float, LoadDist::DIST_BRC_B32>(betaReg, betaAddr + localRow);
        Mul(lowerReg, lowerReg, betaReg, validMask);
        Muls(inverseInitReg, lowerReg, static_cast<float>(-1.0f), validMask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(lowerAddr + offset, lowerReg, validMask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(inverseAddr + offset, inverseInitReg, validMask);

        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(qktBf16Reg, qktAddr + offset);
        Cast<float, bfloat16_t, castBf16ToFp32>(qktFp32Reg, qktBf16Reg, validMask);
        Muls(qktFp32Reg, qktFp32Reg, scale, validMask);
        Mul<float, MaskMergeMode::ZEROING>(qktFp32Reg, qktFp32Reg, decayReg, lowerInclusiveMask);
        Cast<bfloat16_t, float, castFp32ToBf16>(maskedQktBf16Reg, qktFp32Reg, validMask);
        Pack<uint16_t, uint32_t, HighLowPart::LOWEST>(packedBf16Reg,
                                                      reinterpret_cast<RegTensor<uint32_t>&>(maskedQktBf16Reg));
        StoreAlign<uint16_t, DataCopyMode::DATA_BLOCK_COPY>(
            reinterpret_cast<__ubuf__ uint16_t*>(maskedQktZnAddr) + static_cast<uint32_t>(localRow) * 16, packedBf16Reg,
            rowCount, packedMask);
    }
}

/*
 * [3b] 32x32 对角块的八行分组前代求逆：inv_i=e_i-sum_{j<i}(L_ij*inv_j)。
 * lowerUb/invResUb/eiUb 均为 FP32 [32,64]；offset=0/32 选择当前 AIV 的 32x32 对角子块。
 * lowerUb/invResUb 的地址含列偏移 offset；eiUb 由调用方预填单位阵，读取时不加 offset。
 * 每个 invRow 为 32 个有效 FP32 lane，表示 A 的一行；A=(I+L)^(-1)。
 * 优化方案：每组共同依赖的旧逆行仅加载一次，新求得的八行留在寄存器中供后续行复用。
 * 系数 L_ij 通过 B32 从 UB 直接广播，省去常驻的八个系数行寄存器以及索引构造/Gather。
 * 各行的 j 累加顺序不变；组末的 store->load 屏障保证下一组读到完整的旧逆行。
 */
__simd_vf__ inline void InverseAIVPanel8DirectBrcVFImpl(__ubuf__ float* lowerUb, __ubuf__ float* invResUb,
                                                        __ubuf__ float* eiUb, uint32_t offset, uint32_t chunkSize)
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
                                            lowerUb + offset + static_cast<uint32_t>(row) * chunkSize + (col)); \
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

// [3b] 只求当前 AIV 的 A00/A11:[32,32]；A10 由 Cube [4a/4b] 补齐。
__aicore__ inline void InverseAIVPanel8VF(const LocalTensor<float>& lowerUb, const LocalTensor<float>& invResUb,
                                          const LocalTensor<float>& eiUb, uint32_t offset, uint32_t chunkSize)
{
    auto lower = reinterpret_cast<__ubuf__ float*>(lowerUb.GetPhyAddr());
    auto inv = reinterpret_cast<__ubuf__ float*>(invResUb.GetPhyAddr());
    auto ei = reinterpret_cast<__ubuf__ float*>(eiUb.GetPhyAddr());
    InverseAIVPanel8DirectBrcVFImpl(lower, inv, ei, offset, chunkSize);
}
} // namespace ChunkGatedDeltaRule
