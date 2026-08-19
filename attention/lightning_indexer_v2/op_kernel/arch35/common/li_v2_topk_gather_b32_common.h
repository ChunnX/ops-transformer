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
 * \file li_v2_topk_gather_b32_common.h
 * \brief LightningIndexerV2 与 QuantLightningIndexerV2 共用的 b32 topk gather 内核
 *        Shared b32 topk gather kernels used by LightningIndexerV2 and QuantLightningIndexerV2
 */
#ifndef LIGHTNING_INDEXER_V2_TOPK_GATHER_B32_COMMON_H
#define LIGHTNING_INDEXER_V2_TOPK_GATHER_B32_COMMON_H

#include "kernel_operator.h"

namespace liTopkB32GatherCommon {

__simd_vf__ void FindKthVFImpl(__ubuf__ uint32_t *kValue, __ubuf__ uint32_t *histogramsBuf, __ubuf__ uint32_t *idx0Buf,
                               __ubuf__ uint32_t *idx1Buf, __ubuf__ uint32_t *idx2Buf, __ubuf__ uint32_t *idx3Buf)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    MicroAPI::UnalignRegForStore alignIdx3;

    MicroAPI::RegTensor<uint32_t> btmK3;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(btmK3, kValue);

    for (uint16_t i = 0; i < (uint16_t)(4); ++i) {
        MicroAPI::RegTensor<int32_t> idxC;
        MicroAPI::RegTensor<uint32_t> cout;
        MicroAPI::RegTensor<uint32_t> sqzIdx3;

        MicroAPI::MaskReg pregGE = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Arange(idxC, i * 64);
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(cout, histogramsBuf + i * 64);
        MicroAPI::Compare<uint32_t, CMPMODE::GE>(pregGE, cout, btmK3, pregB32);
        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(sqzIdx3, (MicroAPI::RegTensor<uint32_t> &)idxC,
                                                                         pregGE);
        MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(idx3Buf, sqzIdx3, alignIdx3);
    }
    MicroAPI::StoreUnAlignPost(idx3Buf, alignIdx3);

    MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();

    MicroAPI::RegTensor<uint32_t> idx0;
    MicroAPI::RegTensor<uint32_t> idx1;
    MicroAPI::RegTensor<uint32_t> idx2;
    MicroAPI::RegTensor<uint32_t> idx3;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B32>(idx0, idx0Buf);
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B32>(idx1, idx1Buf);
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B32>(idx2, idx2Buf);
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B32>(idx3, idx3Buf);

    MicroAPI::ShiftLefts(idx0, idx0, (int16_t)24, pregB32);
    MicroAPI::ShiftLefts(idx1, idx1, (int16_t)16, pregB32);
    MicroAPI::ShiftLefts(idx2, idx2, (int16_t)8, pregB32);

    // ADD
    MicroAPI::Add(idx0, idx0, idx1, pregB32);
    MicroAPI::Add(idx0, idx0, idx2, pregB32);
    MicroAPI::Add(idx0, idx0, idx3, pregB32);

    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(kValue, idx0, pregB32);
}

__simd_vf__ void FindIdxGTOutputVFImpl(__ubuf__ uint32_t *outputIdxBuf, __ubuf__ uint32_t *inputBuf, uint32_t beginIdx,
                                       __ubuf__ uint32_t *kValue, uint16_t vfLoop)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    MicroAPI::UnalignRegForStore alignIdx;

    MicroAPI::RegTensor<uint32_t> kthValue;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(kthValue, kValue);

    MicroAPI::RegTensor<uint32_t> vregInput;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        MicroAPI::RegTensor<int32_t> idxC;
        MicroAPI::Arange(idxC, beginIdx + i * 64);

        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(vregInput, inputBuf + i * 64);

        MicroAPI::MaskReg poutGT = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

        MicroAPI::RegTensor<uint32_t> sqzIdxOut;
        MicroAPI::Compare<uint32_t, CMPMODE::GT>(poutGT, vregInput, kthValue, pregB32);

        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(sqzIdxOut,
                                                                         (MicroAPI::RegTensor<uint32_t> &)idxC, poutGT);
        MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(outputIdxBuf, sqzIdxOut, alignIdx);
    }
    MicroAPI::StoreUnAlignPost(outputIdxBuf, alignIdx);
}

// 已确认高位的字节直方图结果写回：将两段16bit桶计数扩展为32bit并交错写回
__simd_callee__ inline void StoreHistogramResult(__ubuf__ uint32_t *histogramsBuf, MicroAPI::RegTensor<uint16_t> &cout0,
                                                 MicroAPI::RegTensor<uint16_t> &cout1, MicroAPI::MaskReg &pregB16,
                                                 MicroAPI::MaskReg &pregB32)
{
    MicroAPI::RegTensor<uint32_t> cout0U32Even;
    MicroAPI::RegTensor<uint32_t> cout0U32Odd;
    MicroAPI::RegTensor<uint32_t> cout1U32Even;
    MicroAPI::RegTensor<uint32_t> cout1U32Odd;

    static constexpr MicroAPI::CastTrait CAST_TRAIT_UINT16_TOUINT32_EVEN = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    static constexpr MicroAPI::CastTrait CAST_TRAIT_UINT16_TOUINT32_ODD = {
        MicroAPI::RegLayout::ONE, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout0U32Even, cout0, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout0U32Odd, cout0, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout1U32Even, cout1, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout1U32Odd, cout1, pregB16);

    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(histogramsBuf, cout0U32Even, cout0U32Odd,
                                                                        pregB32);
    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(histogramsBuf + 128, cout1U32Even, cout1U32Odd,
                                                                        pregB32);
}

// 字节级直方图: 根据已确认的高位字节(GATE_NUM个)过滤后, 对下一字节两段桶做直方图
template <int GATE_NUM>
__simd_callee__ inline void HistogramsByteVFImpl(__ubuf__ uint32_t *histogramsBuf, __ubuf__ uint32_t *inputBuf,
                                                 __ubuf__ uint32_t *idx0Buf, __ubuf__ uint32_t *idx1Buf,
                                                 __ubuf__ uint32_t *idx2Buf, uint16_t vfLoop, bool init)
{
    (void)init;
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregB16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregB8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();

    // 计算直方图cout0 0-127 cout1 128-255
    MicroAPI::RegTensor<uint16_t> cout0;
    MicroAPI::RegTensor<uint16_t> cout1;
    MicroAPI::Duplicate(cout0, 0);
    MicroAPI::Duplicate(cout1, 0);

    MicroAPI::RegTensor<uint32_t> idx0;
    MicroAPI::RegTensor<uint32_t> idx1;
    MicroAPI::RegTensor<uint32_t> idx2;
    if constexpr (GATE_NUM >= 1) {
        // 0x000000fc -> 0xfcfcfcfc
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B8>(idx0, idx0Buf);
    }
    if constexpr (GATE_NUM >= 2) {
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B8>(idx1, idx1Buf);
    }
    if constexpr (GATE_NUM >= 3) {
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B8>(idx2, idx2Buf);
    }

    // 32bit 高16bit
    MicroAPI::RegTensor<uint32_t> vreg0U16;
    // 32bit 低16bit
    MicroAPI::RegTensor<uint32_t> vreg1U16;
    MicroAPI::RegTensor<uint32_t> vreg2U16;
    MicroAPI::RegTensor<uint32_t> vreg3U16;

    MicroAPI::RegTensor<uint8_t> vreg0;
    MicroAPI::RegTensor<uint8_t> vreg1;
    MicroAPI::RegTensor<uint8_t> vreg2;
    MicroAPI::RegTensor<uint8_t> vreg3;

    for (uint16_t i = 0; i < vfLoop; ++i) {
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B16>(vreg1U16, vreg0U16, inputBuf + i * 256);
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B16>(vreg3U16, vreg2U16,
                                                                           inputBuf + (i * 256) + 128);

        MicroAPI::DeInterleave(vreg1, vreg0, (MicroAPI::RegTensor<uint8_t> &)vreg0U16,
                               (MicroAPI::RegTensor<uint8_t> &)vreg2U16);
        if constexpr (GATE_NUM >= 2) {
            MicroAPI::DeInterleave(vreg3, vreg2, (MicroAPI::RegTensor<uint8_t> &)vreg1U16,
                                   (MicroAPI::RegTensor<uint8_t> &)vreg3U16);
        }

        if constexpr (GATE_NUM == 0) {
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                                 MicroAPI::HistogramsType::ACCUMULATE>(cout0, vreg0, pregB8);
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                                 MicroAPI::HistogramsType::ACCUMULATE>(cout1, vreg0, pregB8);
        } else if constexpr (GATE_NUM == 1) {
            MicroAPI::MaskReg pregEQ = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
            MicroAPI::Compare<uint8_t, CMPMODE::EQ>(pregEQ, vreg0, (MicroAPI::RegTensor<uint8_t> &)idx0, pregB8);

            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                                 MicroAPI::HistogramsType::ACCUMULATE>(cout0, vreg1, pregEQ);
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                                 MicroAPI::HistogramsType::ACCUMULATE>(cout1, vreg1, pregEQ);
        } else if constexpr (GATE_NUM == 2) {
            MicroAPI::MaskReg pregEQ0 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg pregEQ1 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
            MicroAPI::Compare<uint8_t, CMPMODE::EQ>(pregEQ0, vreg0, (MicroAPI::RegTensor<uint8_t> &)idx0, pregB8);
            MicroAPI::Compare<uint8_t, CMPMODE::EQ>(pregEQ1, vreg1, (MicroAPI::RegTensor<uint8_t> &)idx1, pregB8);
            MicroAPI::MaskReg pregEQ = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
            MicroAPI::And(pregEQ, pregEQ0, pregEQ1, pregB8);

            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                                 MicroAPI::HistogramsType::ACCUMULATE>(cout0, vreg2, pregEQ);
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                                 MicroAPI::HistogramsType::ACCUMULATE>(cout1, vreg2, pregEQ);
        } else {
            MicroAPI::MaskReg pregEQ0 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg pregEQ1 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg pregEQ2 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
            MicroAPI::Compare<uint8_t, CMPMODE::EQ>(pregEQ0, vreg0, (MicroAPI::RegTensor<uint8_t> &)idx0, pregB8);
            MicroAPI::Compare<uint8_t, CMPMODE::EQ>(pregEQ1, vreg1, (MicroAPI::RegTensor<uint8_t> &)idx1, pregB8);
            MicroAPI::Compare<uint8_t, CMPMODE::EQ>(pregEQ2, vreg2, (MicroAPI::RegTensor<uint8_t> &)idx2, pregB8);

            MicroAPI::MaskReg pregEQ0And1 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg pregEQAll = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
            MicroAPI::And(pregEQ0And1, pregEQ0, pregEQ1, pregB8);
            MicroAPI::And(pregEQAll, pregEQ0And1, pregEQ2, pregB8);

            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                                 MicroAPI::HistogramsType::ACCUMULATE>(cout0, vreg3, pregEQAll);
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                                 MicroAPI::HistogramsType::ACCUMULATE>(cout1, vreg3, pregEQAll);
        }
    }

    StoreHistogramResult(histogramsBuf, cout0, cout1, pregB16, pregB32);
}

template <typename T>
__simd_vf__ void HistogramsFirstVFImpl(__ubuf__ uint32_t *histogramsBuf, __ubuf__ uint32_t *inputBuf, uint16_t vfLoop,
                                       bool init)
{
    HistogramsByteVFImpl<0>(histogramsBuf, inputBuf, nullptr, nullptr, nullptr, vfLoop, init);
}

template <typename T>
__simd_vf__ void HistogramsSecondVFImpl(__ubuf__ uint32_t *histogramsBuf, __ubuf__ uint32_t *inputBuf,
                                        __ubuf__ uint32_t *idx0Buf, uint16_t vfLoop, bool init)
{
    HistogramsByteVFImpl<1>(histogramsBuf, inputBuf, idx0Buf, nullptr, nullptr, vfLoop, init);
}

template <typename T>
__simd_vf__ void HistogramsThirdVFImpl(__ubuf__ uint32_t *histogramsBuf, __ubuf__ uint32_t *inputBuf,
                                       __ubuf__ uint32_t *idx0Buf, __ubuf__ uint32_t *idx1Buf, uint16_t vfLoop,
                                       bool init)
{
    HistogramsByteVFImpl<2>(histogramsBuf, inputBuf, idx0Buf, idx1Buf, nullptr, vfLoop, init);
}

template <typename T>
__simd_vf__ void HistogramsLastVFImpl(__ubuf__ uint32_t *histogramsBuf, __ubuf__ uint32_t *inputBuf,
                                      __ubuf__ uint32_t *idx0Buf, __ubuf__ uint32_t *idx1Buf,
                                      __ubuf__ uint32_t *idx2Buf, uint16_t vfLoop, bool init)
{
    HistogramsByteVFImpl<3>(histogramsBuf, inputBuf, idx0Buf, idx1Buf, idx2Buf, vfLoop, init);
}

} // namespace liTopkB32GatherCommon

#endif // LIGHTNING_INDEXER_V2_TOPK_GATHER_B32_COMMON_H
