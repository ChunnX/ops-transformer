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
#include "../../moe_distribute_dispatch_v2/op_kernel/quantize_functions.h"
using namespace AscendC;

constexpr uint8_t MXFP8_E5M2_COMM_QUANT = 3;
constexpr uint8_t MXFP8_E4M3_COMM_QUANT = 4;
namespace MegaMoeCombineImpl {
template<uint8_t QuantMode, typename ExpandXType>
__aicore__ inline void QuantMxFp8(LocalTensor<ExpandXType>& outLocal, LocalTensor<ExpandXType>& inLocal, 
    LocalTensor<float>& floatTemp, int32_t processLen)
{
    PipeBarrier<PIPE_V>();
    uint32_t mxScaleNum = Align2(Ceil32(processLen));
    LocalTensor<fp8_e5m2_t> castFp8LocalTensor = outLocal.template ReinterpretCast<fp8_e5m2_t>();
    __ubuf__ ExpandXType* srcAddr = (__ubuf__ ExpandXType*)inLocal.GetPhyAddr();
    __ubuf__ uint16_t* maxExpAddr = (__ubuf__ uint16_t*)floatTemp.GetPhyAddr();
    __ubuf__ uint16_t* halfScaleLocalAddr = (__ubuf__ uint16_t*)floatTemp[Align32(mxScaleNum)].GetPhyAddr();
    __ubuf__ int8_t* outLocalAddr = (__ubuf__ int8_t*)castFp8LocalTensor.GetPhyAddr();
    __ubuf__ uint16_t* mxScaleLocalAddr =
        (__ubuf__ uint16_t*)castFp8LocalTensor[processLen].GetPhyAddr();
    quant::ComputeMaxExp(srcAddr, maxExpAddr, processLen); // 计算最大Exp
    if constexpr (QuantMode == MXFP8_E5M2_COMM_QUANT) {
        // 计算scales并填充
        quant::ComputeScale<fp8_e5m2_t>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, mxScaleNum);
        quant::ComputeFp8Data<ExpandXType, fp8_e5m2_t,
            AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
            srcAddr, halfScaleLocalAddr, outLocalAddr, processLen); // 计算量化后的expandx并填充
    } else if constexpr (QuantMode == MXFP8_E4M3_COMM_QUANT) {
        // 计算scales并填充
        quant::ComputeScale<fp8_e4m3fn_t>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, mxScaleNum);//确认下scale排列
        quant::ComputeFp8Data<ExpandXType, fp8_e4m3fn_t,
            AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
            srcAddr, halfScaleLocalAddr, outLocalAddr, processLen); // 计算量化后的expandx并填充
    }
}

template <typename T, typename XType>
__aicore__ inline void DeQuantMxFp8(LocalTensor<XType>& inLocal, LocalTensor<float>& sumTensor, 
    LocalTensor<bfloat16_t>& scaleBf16Tensor, LocalTensor<float>& scaleFP32Tensor,
    uint32_t scaleLen, uint32_t tokenLen)
{
    LocalTensor<T> castFp8LocalTensor_ = inLocal.template ReinterpretCast<T>();
    // bf16/fp16量化为mxfp8后，字节差2倍
    LocalTensor<fp8_e8m0_t> scaleDivFp8Tensor_ =
        inLocal[Align256<uint32_t>(tokenLen) / 2].template ReinterpretCast<fp8_e8m0_t>();
    __ubuf__ bfloat16_t *dyScaleBf16Ptr = (__ubuf__ bfloat16_t *)scaleBf16Tensor.GetPhyAddr();
    __ubuf__ float *dyScaleFp32Ptr = (__ubuf__ float *)scaleFP32Tensor.GetPhyAddr(); 
    __ubuf__ fp8_e8m0_t *srcPtr0 = (__ubuf__ fp8_e8m0_t *)scaleDivFp8Tensor_.GetPhyAddr();
    __ubuf__ T *tokenPtr0 = (__ubuf__ T *)castFp8LocalTensor_.GetPhyAddr();
    __ubuf__ float *sumDstPtr = (__ubuf__ float *)sumTensor.GetPhyAddr();
    uint32_t bf16RepeatSize = quant::GetVRegSizeDispatch() / sizeof(bfloat16_t);
    uint32_t fp32RepeatSize = quant::GetVRegSizeDispatch() / sizeof(float);
    uint16_t repeatTimes = Ceil(scaleLen, bf16RepeatSize);
    uint16_t fp32RepeatTimes = Ceil(tokenLen, fp32RepeatSize);
    uint16_t repeatTimes2 = Ceil(scaleLen * 2, fp32RepeatSize);
    uint32_t quantCount2 = scaleLen * 2;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<fp8_e8m0_t> vSrcReg;
        AscendC::MicroAPI::RegTensor<T> tokenSrcReg;
        AscendC::MicroAPI::RegTensor<float> tokenFp32SrcReg;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vDstReg;
        AscendC::MicroAPI::RegTensor<bfloat16_t> dyScaleBf16Reg;
        AscendC::MicroAPI::RegTensor<float> dyScaleFp32Reg;
        AscendC::MicroAPI::RegTensor<float> sumDstReg;
        AscendC::MicroAPI::RegTensor<float> sumLocalDstReg;
        static constexpr AscendC::MicroAPI::CastTrait FP82BF16CastTraitZero = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait FP162FP32CastTraitZero = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
        AscendC::MicroAPI::MaskReg maskReg;
        AscendC::MicroAPI::MaskReg maskReg1;
        AscendC::MicroAPI::MaskReg maskReg2;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskReg = AscendC::MicroAPI::UpdateMask<bfloat16_t>(scaleLen);
            MicroAPI::DataCopy<fp8_e8m0_t, MicroAPI::LoadDist::DIST_UNPACK_B8>(vSrcReg,
                srcPtr0 + i * bf16RepeatSize);
            MicroAPI::Cast<bfloat16_t, fp8_e8m0_t, FP82BF16CastTraitZero>(vDstReg, vSrcReg, maskReg);
            MicroAPI::DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(
                dyScaleBf16Ptr + i * bf16RepeatSize * 2, vDstReg, vDstReg, maskReg);
        }
        MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t i = 0; i < repeatTimes2; i++) {
            maskReg1 = AscendC::MicroAPI::UpdateMask<float>(quantCount2);
            MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(dyScaleBf16Reg,
                dyScaleBf16Ptr + i * fp32RepeatSize); // 128/2=64 搬入64个u16 unpack成64个u32
            MicroAPI::Cast<float, bfloat16_t, FP162FP32CastTraitZero>(dyScaleFp32Reg, dyScaleBf16Reg, maskReg1);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_INTLV_B32>(
                dyScaleFp32Ptr + i * fp32RepeatSize * 2, dyScaleFp32Reg, dyScaleFp32Reg, maskReg1);
        }
        MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t i = 0; i < fp32RepeatTimes; i++) {
            maskReg2 = AscendC::MicroAPI::UpdateMask<float>(tokenLen);
            // 广播4B->32B，8倍
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_E2B_B32>(dyScaleFp32Reg, dyScaleFp32Ptr + i * 8);
            // 接收到的token float8_e5m2_t
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(tokenSrcReg,
                tokenPtr0 + i * fp32RepeatSize);
            MicroAPI::Cast<float, T, FP82BF16CastTraitZero>(tokenFp32SrcReg, tokenSrcReg, maskReg2);
            MicroAPI::Mul(sumLocalDstReg, dyScaleFp32Reg, tokenFp32SrcReg, maskReg2); // token与量化参数相乘
            MicroAPI::DataCopy(sumDstPtr + i * fp32RepeatSize, sumLocalDstReg, maskReg2); // 最后搬出 float类型
        }
    }
}
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

// Send quantized data + scale to remote ranks via peermem
// Per token row: send fp8_e5m2 data (n bytes) + e8m0 scale (nScale bytes) to destination rank
template <typename QuantOutType>
__aicore__ inline void CombineQuantizedTokens(
    uint32_t batchStart, uint32_t curRows, uint32_t n, uint32_t nScale,
    uint32_t groupIdx, uint32_t rankId, LocalTensor<QuantOutType>& ubQuant, const Params& params)
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
        AscendC::DataCopyPad(gmRemoteD[dstBaseOffset], ubQuant[ubSrcRowOffset * quantRowSize], singleCopyParams);
    }
}

// CombineRowGroup: process a completed row-group (read full rows -> quantize -> send)
template <uint8_t QuantMode, typename T>
__aicore__ inline void CombineRowGroup(
    uint32_t rowStart, uint32_t rowCount, uint32_t n, uint32_t groupIdx, uint32_t rankId,
    GM_ADDR gmm2OutAddr, const Params& params,
    LocalTensor<T>& combineUbTensor, int64_t offSet, uint32_t quantRowSizeBytes)
{
    GlobalTensor<T> gmm2OutGm;
    gmm2OutGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gmm2OutAddr));

    uint32_t nScale = Ops::Base::CeilDiv(n, uint32_t(MXFP_SCALE_GROUP_NUM));
    uint32_t nAlign32 = Ops::Base::CeilAlign(n, 32U);
    uint32_t floatTempSize = nScale + nScale / 2;
    LocalTensor<float> floatTemp = LocalTensor<float>(TPosition::VECIN, offSet, floatTempSize);
    uint32_t singleTokenElems = (nAlign32 * sizeof(T) + quantRowSizeBytes) / sizeof(T);
    for (uint32_t i = 0; i < rowCount; i++) {
        uint32_t pingPong = i % 2;
        LocalTensor<T> ubBf16 = combineUbTensor[pingPong * singleTokenElems];
        LocalTensor<T> ubQuantData = ubBf16[nAlign32];
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        CopyFullRowsFromGM(ubBf16, gmm2OutGm, rowStart + i, 1, n);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        QuantMxFp8<QuantMode, T>(ubQuantData, ubBf16, floatTemp, n);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        LocalTensor<fp8_e5m2_t> ubQuantDataFp8 = ubQuantData.template ReinterpretCast<fp8_e5m2_t>();
        CombineQuantizedTokens<fp8_e5m2_t>(rowStart + i, 1, n, nScale, groupIdx, rankId, ubQuantDataFp8, params);
    }
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
}

}  // namespace MegaMoeCombineImpl

#endif  // MEGA_MOE_COMBINE_SEND_H