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
 * \file mega_moe_combine_send.h
 * \brief Token-level Combine: read full row from GM -> MX quantize (bf16->fp8_e5m2+e8m0) -> send to remote rank
 */

#ifndef MEGA_MOE_COMBINE_SEND_H
#define MEGA_MOE_COMBINE_SEND_H

#include "kernel_operator.h"
#include "mega_moe_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
using namespace AscendC;

namespace MegaMoeCombineImpl {

constexpr uint32_t QUANT_BATCH_ROWS = 8;

// Copy full rows from GM to UB (row-contiguous layout, each row has width n bf16 elements)
template <typename ElementBf16>
__aicore__ inline void CopyFullRowsFromGM(
    LocalTensor<ElementBf16>& ubDst,
    GlobalTensor<ElementBf16>& gmSrc,
    uint32_t rowStart, uint32_t rowCount, uint32_t n)
{
    uint32_t nAlign32 = Ops::Base::CeilAlign(n, 32U);
    DataCopyPadExtParams<ElementBf16> copyPadParams{false, 0U, 0U, 0U};
    AscendC::DataCopyExtParams gm2UbParams{
        static_cast<uint16_t>(rowCount),
        static_cast<uint32_t>(nAlign32 * sizeof(ElementBf16)),
        0,
        static_cast<uint16_t>((nAlign32 - n) * sizeof(ElementBf16)),
        0};
    AscendC::DataCopyPad(ubDst, gmSrc[rowStart * nAlign32], gm2UbParams, copyPadParams);
}

// ComputeMaxExp for single row: find max exponent per 32-element group
// Returns nScale uint16_t max exponent values (one per MX group of 32 elements)
__aicore__ inline void TokenComputeMaxExp(
    __ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *maxExpAddr,
    uint32_t n, uint32_t nAlign32)
{
    constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
    uint16_t loopNum = Ops::Base::CeilDiv(nAlign32, 256U);
    uint16_t elementAfterReduce = 8;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0, vdExp1;
        AscendC::MicroAPI::RegTensor<uint16_t> vdExpExtract0, vdExpExtract1;
        AscendC::MicroAPI::RegTensor<uint16_t> expMaskBF16;
        AscendC::MicroAPI::Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);
        AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
        AscendC::MicroAPI::MaskReg scaleMask1;
        AscendC::MicroAPI::UnalignReg u1;

        for (uint16_t i = 0; i < loopNum; i++) {
            scaleMask1 = AscendC::MicroAPI::UpdateMask<bfloat16_t>(nAlign32);
            AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr, 128 * 2);
            AscendC::MicroAPI::And(vdExpExtract0, (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp0, expMaskBF16, scaleMask1);
            AscendC::MicroAPI::And(vdExpExtract1, (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp1, expMaskBF16, scaleMask1);
            AscendC::MicroAPI::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, scaleMask1);
            AscendC::MicroAPI::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, scaleMask1);
            AscendC::MicroAPI::DataCopyUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, vdMaxExp, u1, elementAfterReduce);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(maxExpAddr, u1, 0);
    }
}

// ComputeScale: from maxExp -> e8m0 scale + halfScale
__aicore__ inline void TokenComputeScale(
    __ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *mxScaleLocalAddr,
    __ubuf__ uint16_t *halfScaleLocalAddr,
    uint32_t nScale, uint32_t nScaleAlign)
{
    constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
    constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780;
    constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
    constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
    constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
    constexpr int16_t SHR_NUM_FOR_BF16 = 7;

    uint16_t loopNumScale = Ops::Base::CeilDiv(nScaleAlign, 128U);

    static constexpr AscendC::MicroAPI::DivSpecificMode DIV_MODE = {
        AscendC::MicroAPI::MaskMergeMode::ZEROING, true};

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint16_t> expMask, sharedExp, scaleValue, scaleBias, halfScale, fp8NanRegTensor;
        AscendC::MicroAPI::Duplicate(expMask, MAX_EXP_FOR_BF16);
        AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
        AscendC::MicroAPI::RegTensor<uint16_t> maxExpValue, zeroRegTensor, nanRegTensor, specialExpRegTensor;
        AscendC::MicroAPI::Duplicate(maxExpValue, FP8_E5M2_MAX_EXP);
        AscendC::MicroAPI::Duplicate(scaleBias, BF16_EXP_BIAS);
        AscendC::MicroAPI::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
        AscendC::MicroAPI::Duplicate(zeroRegTensor, 0);
        AscendC::MicroAPI::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
        AscendC::MicroAPI::MaskReg cmpResult, zeroMask, invalidDataMask, specialDataMask;
        AscendC::MicroAPI::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);
        AscendC::MicroAPI::MaskReg preMaskScale;

        for (uint16_t i = 0; i < loopNumScale; i++) {
            preMaskScale = AscendC::MicroAPI::UpdateMask<uint16_t>(nScaleAlign);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                vdMaxExp, maxExpAddr, 128);
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(cmpResult, vdMaxExp, expMask, preMaskScale);
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue, preMaskScale);
            AscendC::MicroAPI::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);
            AscendC::MicroAPI::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
            AscendC::MicroAPI::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);
            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(
                mxScaleLocalAddr, scaleValue, 64, preMaskScale);
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
            AscendC::MicroAPI::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                halfScaleLocalAddr, halfScale, 128, preMaskScale);
        }
    }
}

// Quantize bf16 -> fp8_e5m2 using halfScale multiplication
__aicore__ inline void TokenQuantizeDataFp8(
    __ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *halfScaleLocalAddr,
    __ubuf__ int8_t *outLocalAddr,
    uint32_t nAlign32, uint16_t loopNum)
{
    using T = bfloat16_t;
    using DataTypeOut = fp8_e5m2_t;
    constexpr uint16_t OUT_ELE_NUM_ONE_BLK = 64;

    static constexpr AscendC::MicroAPI::CastTrait castTraitZero = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    static constexpr AscendC::MicroAPI::CastTrait castTraitOne = {
        AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    static constexpr AscendC::MicroAPI::CastTrait castTrait32to8 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg dataMask1, dataMask2, dataMask3, dataMask4;
        AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
        AscendC::MicroAPI::RegTensor<T> vdExp0, vdExp1;
        AscendC::MicroAPI::RegTensor<float> vdExp0FP32Zero, vdExp0FP32One, vdExp1FP32Zero, vdExp1FP32One;
        AscendC::MicroAPI::RegTensor<DataTypeOut> vdExp0FP8Zero, vdExp0FP8One, vdExp1FP8Zero, vdExp1FP8One;
        AscendC::MicroAPI::RegTensor<uint16_t> elementAfterReduceReg;
        AscendC::MicroAPI::Duplicate(elementAfterReduceReg, 8);

        for (uint16_t i = 0; i < loopNum; i++) {
            dataMask1 = AscendC::MicroAPI::UpdateMask<T>(nAlign32);
            dataMask2 = AscendC::MicroAPI::UpdateMask<T>(nAlign32);
            uint32_t totalCountInUB2 = nAlign32 * 2;
            dataMask3 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB2);
            dataMask4 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB2);

            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr, 128 * 2);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(
                halfScaleForMul, halfScaleLocalAddr, 8);
            AscendC::MicroAPI::Mul(vdExp0, vdExp0, (AscendC::MicroAPI::RegTensor<T> &)halfScaleForMul, dataMask1);
            AscendC::MicroAPI::Mul(vdExp1, vdExp1, (AscendC::MicroAPI::RegTensor<T> &)halfScaleForMul, dataMask1);
            AscendC::MicroAPI::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
            AscendC::MicroAPI::Cast<float, T, castTraitZero>(vdExp0FP32Zero, vdExp0, dataMask1);
            AscendC::MicroAPI::Cast<float, T, castTraitOne>(vdExp0FP32One, vdExp0, dataMask1);
            AscendC::MicroAPI::Interleave(vdExp0FP32Zero, vdExp0FP32One, vdExp0FP32Zero, vdExp0FP32One);
            AscendC::MicroAPI::Cast<DataTypeOut, float, castTrait32to8>(vdExp0FP8Zero, vdExp0FP32Zero, dataMask3);
            AscendC::MicroAPI::Cast<DataTypeOut, float, castTrait32to8>(vdExp0FP8One, vdExp0FP32One, dataMask3);
            AscendC::MicroAPI::Cast<float, T, castTraitZero>(vdExp1FP32Zero, vdExp1, dataMask2);
            AscendC::MicroAPI::Cast<float, T, castTraitOne>(vdExp1FP32One, vdExp1, dataMask2);
            AscendC::MicroAPI::Interleave(vdExp1FP32Zero, vdExp1FP32One, vdExp1FP32Zero, vdExp1FP32One);
            AscendC::MicroAPI::Cast<DataTypeOut, float, castTrait32to8>(vdExp1FP8Zero, vdExp1FP32Zero, dataMask4);
            AscendC::MicroAPI::Cast<DataTypeOut, float, castTrait32to8>(vdExp1FP8One, vdExp1FP32One, dataMask4);
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp0FP8Zero, OUT_ELE_NUM_ONE_BLK, dataMask3);
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp0FP8One, OUT_ELE_NUM_ONE_BLK, dataMask3);
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp1FP8Zero, OUT_ELE_NUM_ONE_BLK, dataMask4);
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp1FP8One, OUT_ELE_NUM_ONE_BLK, dataMask4);
        }
    }
}

// TokenMxQuantize: quantize batch of rows bf16 -> fp8_e5m2 + e8m0_scale (MX per-32-group)
// Quantize row-by-row, each row gets its own MX scale
template <typename QuantOutType, typename QuantScaleOutType>
__aicore__ inline void TokenMxQuantize(
    LocalTensor<bfloat16_t>& srcBf16,
    LocalTensor<QuantOutType>& dstQuant,
    LocalTensor<QuantScaleOutType>& dstScale,
    uint32_t curRows, uint32_t n)
{
    uint32_t nAlign32 = Ops::Base::CeilAlign(n, 32U);
    uint32_t nScale = Ops::Base::CeilDiv(n, MXFP_SCALE_GROUP_NUM) * MXFP_MULTI_BASE_SIZE;
    uint32_t nScaleAlign32 = Ops::Base::CeilAlign(nScale, 32U);

    for (uint32_t rowIdx = 0; rowIdx < curRows; rowIdx++) {
        __ubuf__ bfloat16_t *srcAddr = (__ubuf__ bfloat16_t *)srcBf16.GetPhyAddr() + rowIdx * nAlign32;
        __ubuf__ int8_t *quantAddr = (__ubuf__ int8_t *)dstQuant.GetPhyAddr() + rowIdx * n;
        __ubuf__ int8_t *scaleAddr = (__ubuf__ int8_t *)dstScale.GetPhyAddr() + rowIdx * nScale;

        uint32_t maxExpBufSize = nScaleAlign32 * sizeof(uint16_t);
        uint32_t halfScaleBufSize = nScaleAlign32 * sizeof(uint16_t);

        LocalTensor<uint16_t> maxExpBuf = srcBf16[curRows * nAlign32 + rowIdx * nScaleAlign32]
            .ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> halfScaleBuf = maxExpBuf[nScaleAlign32];

        TokenComputeMaxExp(srcAddr, (__ubuf__ uint16_t *)maxExpBuf.GetPhyAddr(), n, nAlign32);
        TokenComputeScale((__ubuf__ uint16_t *)maxExpBuf.GetPhyAddr(),
                          (__ubuf__ uint16_t *)scaleAddr,
                          (__ubuf__ uint16_t *)halfScaleBuf.GetPhyAddr(),
                          nScale, nScaleAlign32);
        uint16_t loopNum = Ops::Base::CeilDiv(nAlign32, 256U);
        TokenQuantizeDataFp8(srcAddr, (__ubuf__ uint16_t *)halfScaleBuf.GetPhyAddr(),
                             quantAddr, nAlign32, loopNum);
    }
}

// Send quantized data + scale to remote ranks via peermem
// Per token row: send fp8_e5m2 data (n bytes) + e8m0 scale (nScale bytes) to destination rank
template <typename QuantOutType, typename QuantScaleOutType>
__aicore__ inline void CombineQuantizedTokens(
    uint32_t batchStart, uint32_t curRows, uint32_t n, uint32_t nScale,
    uint32_t groupIdx, uint32_t rankId,
    LocalTensor<QuantOutType>& ubQuant, LocalTensor<QuantScaleOutType>& ubScale,
    const Params& params)
{
    GlobalTensor<int32_t> tokenPerExpertWin, preSumBeforeRankWorkSpace;
    tokenPerExpertWin.SetGlobalBuffer((__gm__ int32_t*)params.peermemInfo.ptrTokenPerExpert);
    preSumBeforeRankWorkSpace.SetGlobalBuffer((__gm__ int32_t*)params.workspaceInfo.ptrSumBeforeRank);

    int32_t perRankStridesInTokenPerExpert = Ops::Base::CeilAlign(
        static_cast<uint32_t>(params.tilingData->epWorldSize * params.tilingData->expertPerRank),
        static_cast<uint32_t>(ALIGN_128));
    int32_t preSumRankInExpert = 0;
    int32_t tileOffset = 0;
    int64_t quantRowSize = n + nScale;

    for (int32_t dstEpIdx = 0; dstEpIdx < params.tilingData->epWorldSize; dstEpIdx++) {
        int32_t lenRankInExpert = tokenPerExpertWin(dstEpIdx * Ops::Base::CeilAlign(
            static_cast<uint32_t>(params.tilingData->epWorldSize * params.tilingData->expertPerRank),
            static_cast<uint32_t>(ALIGN_128)) + rankId * params.tilingData->expertPerRank + groupIdx); // 当前循环的卡给本专家发了多少token
        int32_t dstExpertOffset = preSumBeforeRankWorkSpace(dstEpIdx * params.tilingData->expertPerRank + groupIdx);
        // 本专家从各个卡上的收到token数量的前缀和，这里作为数据发回时的偏移起点
        int32_t stRankInExpert = preSumRankInExpert;
        int32_t edRankInExpert = stRankInExpert + lenRankInExpert;
        preSumRankInExpert += lenRankInExpert;
        // 轮询并维护一个前缀和，stRankInExpert edRankInExpert分别是当前卡发给本专家的token的区间

        if (stRankInExpert >= (int32_t)(batchStart + curRows)) {
            break;
        } else if (edRankInExpert <= (int32_t)batchStart) {
            continue;
        }// 这里在判断当前这个token是否在这个区间内，在区间前则说明已经发过了直接break，在区间后则还没有轮询到目的卡直接continue

        int32_t stData = Blaze::Gemm::Max(stRankInExpert, (int32_t)batchStart);
        int32_t edData = Blaze::Gemm::Min(edRankInExpert, (int32_t)(batchStart + curRows));
        int32_t lenData = edData - stData;
        if (lenData <= 0) {
            continue;
        }

        uint32_t dstOffsetInExpert = 0;
        if ((int32_t)batchStart > stRankInExpert) {
            dstOffsetInExpert = batchStart - stRankInExpert;
        }

        uint32_t ubSrcRowOffset = stData - (int32_t)batchStart;

        AscendC::GlobalTensor<QuantOutType> gmRemoteD;
        uint64_t gmRemoteOffset = params.peermemInfo.ptrD - params.peermemInfo.ptrBase;
        __gm__ void* dstPeermemPtr = GetRankWinAddrWithOffset(dstEpIdx, gmRemoteOffset);
        gmRemoteD.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(dstPeermemPtr));

        uint64_t dstBaseOffset = (dstOffsetInExpert + dstExpertOffset) * quantRowSize;

        AscendC::DataCopyExtParams singleCopyParams{
            static_cast<uint16_t>(lenData),
            static_cast<uint32_t>(quantRowSize),
            0, 0, 0};
        AscendC::DataCopyPad(gmRemoteD[dstBaseOffset],
            ubQuant[ubSrcRowOffset * n], singleCopyParams, {});
    }
}

// DequantizeMxToBf16: fp8_e5m2 * e8m0_scale -> bf16
// Input: quantBuf contains fp8 data, scaleBuf contains e8m0 scale values
// Output: dstBf16 contains dequantized bf16 values
__aicore__ inline void DequantizeMxToBf16(
    LocalTensor<fp8_e5m2_t>& quantBuf,
    LocalTensor<fp8_e8m0_t>& scaleBuf,
    LocalTensor<bfloat16_t>& dstBf16,
    uint32_t n, uint32_t nScale)
{
    uint32_t nAlign32 = Ops::Base::CeilAlign(n, 32U);
    uint32_t nScaleAlign32 = Ops::Base::CeilAlign(nScale, 32U);
    constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
    constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780;

    uint16_t loopNumData = Ops::Base::CeilDiv(nAlign32, 256U);
    uint16_t loopNumScale = Ops::Base::CeilDiv(nScaleAlign32, 128U);

    __ubuf__ fp8_e5m2_t *quantAddr = (__ubuf__ fp8_e5m2_t *)quantBuf.GetPhyAddr();
    __ubuf__ fp8_e8m0_t *scaleAddr = (__ubuf__ fp8_e8m0_t *)scaleBuf.GetPhyAddr();
    __ubuf__ bfloat16_t *dstAddr = (__ubuf__ bfloat16_t *)dstBf16.GetPhyAddr();

    static constexpr AscendC::MicroAPI::CastTrait ctFp8ToFp32 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    static constexpr AscendC::MicroAPI::CastTrait ctInt8ToUint16 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    static constexpr AscendC::MicroAPI::CastTrait ctFp32ToBf16 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint16_t> scaleReg, sharedExpReg;
        AscendC::MicroAPI::RegTensor<uint16_t> biasReg, maxExpReg;
        AscendC::MicroAPI::Duplicate(biasReg, BF16_EXP_BIAS);
        AscendC::MicroAPI::Duplicate(maxExpReg, FP8_E5M2_MAX_EXP);
        AscendC::MicroAPI::RegTensor<fp8_e5m2_t> vDataLow, vDataHigh;
        AscendC::MicroAPI::RegTensor<float> vFp32Low, vFp32High;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vBf16Result;
        AscendC::MicroAPI::MaskReg dataMask;

        uint32_t scaleIdx = 0;
        uint32_t scaleElementsPerLoop = MXFP_SCALE_GROUP_NUM;

        for (uint16_t i = 0; i < loopNumData; i++) {
            dataMask = AscendC::MicroAPI::UpdateMask<fp8_e5m2_t>(256);
            AscendC::MicroAPI::DataCopy<fp8_e5m2_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                vDataLow, vDataHigh, quantAddr, 128 * 2);
            AscendC::MicroAPI::Cast<float, fp8_e5m2_t, ctFp8ToFp32>(vFp32Low, vDataLow, dataMask);
            AscendC::MicroAPI::Cast<float, fp8_e5m2_t, ctFp8ToFp32>(vFp32High, vDataHigh, dataMask);

            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                scaleReg, scaleAddr + scaleIdx, 8);
            AscendC::MicroAPI::Add(sharedExpReg, scaleReg, maxExpReg, dataMask);

            AscendC::MicroAPI::RegTensor<float> scaleFp32;
            AscendC::MicroAPI::Cast<float, uint16_t, ctInt8ToUint16>(scaleFp32, sharedExpReg, dataMask);
            AscendC::MicroAPI::Mul(vFp32Low, vFp32Low, scaleFp32, dataMask);
            AscendC::MicroAPI::Mul(vFp32High, vFp32High, scaleFp32, dataMask);

            AscendC::MicroAPI::Cast<bfloat16_t, float, ctFp32ToBf16>(vBf16Result, vFp32Low, dataMask);
            AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                dstAddr, vBf16Result, 128);
            AscendC::MicroAPI::Cast<bfloat16_t, float, ctFp32ToBf16>(vBf16Result, vFp32High, dataMask);
            AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                dstAddr, vBf16Result, 128);

            scaleIdx += MXFP_SCALE_GROUP_NUM;
        }
    }
}

// CombineRowGroup: process a completed row-group (read full rows -> quantize -> send)
template <typename QuantOutType, typename QuantScaleOutType>
__aicore__ inline void CombineRowGroup(
    uint32_t rowStart, uint32_t rowCount, uint32_t n, uint32_t groupIdx, uint32_t rankId,
    GM_ADDR gmm2OutAddr, const Params& params,
    LocalTensor<bfloat16_t>& combineUbTensor)
{
    GlobalTensor<bfloat16_t> gmm2OutGm;
    gmm2OutGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(gmm2OutAddr));

    uint32_t nScale = Ops::Base::CeilDiv(n, MXFP_SCALE_GROUP_NUM) * MXFP_MULTI_BASE_SIZE;// scale的长度
    uint32_t nAlign32 = Ops::Base::CeilAlign(n, 32U);// token对齐后的长度

    for (uint32_t tokenIdx = rowStart; tokenIdx < rowStart + rowCount; tokenIdx ++) {
        LocalTensor<bfloat16_t> ubBf16 = combineUbTensor;// 原始数据
        LocalTensor<QuantOutType> ubQuantData = combineUbTensor[nAlign32].ReinterpretCast<QuantOutType>();// 量化后的数据
        LocalTensor<QuantScaleOutType> ubQuantScale = combineUbTensor[nAlign32 + n].ReinterpretCast<QuantScaleOutType>();// 量化后scale
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        CopyFullRowsFromGM(ubBf16, gmm2OutGm, tokenIdx, 1, n);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        TokenMxQuantize<QuantOutType, QuantScaleOutType>(
            ubBf16, ubQuantData, ubQuantScale, 1, n);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        CombineQuantizedTokens<QuantOutType, QuantScaleOutType>(
            tokenIdx, 1, n, nScale, groupIdx, rankId,
            ubQuantData, ubQuantScale, params);
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    }
}

}  // namespace MegaMoeCombineImpl

#endif  // MEGA_MOE_COMBINE_SEND_H