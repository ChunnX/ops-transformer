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
 * \file vf_topk.h
 * \brief
 */

#ifndef VF_TOP_K_H
#define VF_TOP_K_H

#include "kernel_operator.h"
#if __has_include("../../../lightning_indexer_v2/arch35/vf/common/li_v2_topk_gather_b32_common.h")
#include "../../../lightning_indexer_v2/arch35/vf/common/li_v2_topk_gather_b32_common.h"
#else
#include "../../../../lightning_indexer_v2/op_kernel/arch35/vf/common/li_v2_topk_gather_b32_common.h"
#endif

namespace topkb32 {
using liTopkB32GatherCommon::FindKthVFImpl;
using liTopkB32GatherCommon::FindIdxGTOutputVFImpl;
using liTopkB32GatherCommon::StoreHistogramResult;
using liTopkB32GatherCommon::HistogramsByteVFImpl;
using liTopkB32GatherCommon::HistogramsFirstVFImpl;
using liTopkB32GatherCommon::HistogramsSecondVFImpl;
using liTopkB32GatherCommon::HistogramsThirdVFImpl;
using liTopkB32GatherCommon::HistogramsLastVFImpl;

__simd_callee__ inline void FindTargetBinAndUpdateNextK(__ubuf__ uint32_t *idxBuf, __ubuf__ uint32_t *nkValueBuf,
                                                        __ubuf__ uint32_t *histogramsBuf,
                                                        MicroAPI::RegTensor<uint32_t> &btmK, MicroAPI::MaskReg &pregB32)
{
    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    MicroAPI::UnalignRegForStore alignIdx;

    for (uint16_t i = 0; i < (uint16_t)(4); ++i) {
        MicroAPI::RegTensor<int32_t> idxC;
        MicroAPI::RegTensor<uint32_t> cout;
        MicroAPI::RegTensor<uint32_t> sqzIdx;

        MicroAPI::MaskReg pregGE = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Arange(idxC, i * 64);
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(cout, histogramsBuf + i * 64);
        MicroAPI::Compare<uint32_t, CMPMODE::GE>(pregGE, cout, btmK, pregB32);
        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(sqzIdx, (MicroAPI::RegTensor<uint32_t> &)idxC,
                                                                         pregGE);
        MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(idxBuf, sqzIdx, alignIdx);
    }
    MicroAPI::StoreUnAlignPost(idxBuf, alignIdx);

    MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();

    MicroAPI::RegTensor<uint32_t> idx;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B8>(idx, idxBuf);

    MicroAPI::RegTensor<uint8_t> idxAll1;
    MicroAPI::RegTensor<uint32_t> idxPrev;
    MicroAPI::RegTensor<uint32_t> prevBinValue;
    MicroAPI::Duplicate(idxAll1, 1);

    MicroAPI::RegTensor<uint32_t> zeroAll;
    MicroAPI::Duplicate(zeroAll, 0);

    MicroAPI::MaskReg pregZero = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Compare<uint32_t, CMPMODE::EQ>(pregZero, idx, zeroAll, pregB32);
    MicroAPI::Sub(idxPrev, idx, (MicroAPI::RegTensor<uint32_t> &)idxAll1, pregB32);
    MicroAPI::ShiftRights(idxPrev, idxPrev, (int16_t)24, pregB32);

    MicroAPI::Gather(prevBinValue, histogramsBuf, idxPrev, pregB32);
    MicroAPI::Select(prevBinValue, zeroAll, prevBinValue, pregZero);

    MicroAPI::RegTensor<uint32_t> nextK;
    MicroAPI::Sub(nextK, btmK, prevBinValue, pregB32);
    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(nkValueBuf, nextK, pregB32);
}

__simd_vf__ void FindFirstTargetBinVFImpl(__ubuf__ uint32_t *idx0Buf, __ubuf__ uint32_t *nkValueBuf,
                                          __ubuf__ uint32_t *histogramsBuf, uint32_t bottomK)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::RegTensor<uint32_t> btmK;
    MicroAPI::Duplicate(btmK, bottomK);

    FindTargetBinAndUpdateNextK(idx0Buf, nkValueBuf, histogramsBuf, btmK, pregB32);
}

// kValue新的bottomK
__simd_vf__ void FindSecondTargetBinVFImpl(__ubuf__ uint32_t *idx1Buf, __ubuf__ uint32_t *nkValueBuf,
                                           __ubuf__ uint32_t *kValue, __ubuf__ uint32_t *histogramsBuf)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::RegTensor<uint32_t> btmK1;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(btmK1, kValue);

    FindTargetBinAndUpdateNextK(idx1Buf, nkValueBuf, histogramsBuf, btmK1, pregB32);
}

__simd_vf__ void FindThirdTargetBinVFImpl(__ubuf__ uint32_t *idx2Buf, __ubuf__ uint32_t *nkValueBuf,
                                          __ubuf__ uint32_t *kValue, __ubuf__ uint32_t *histogramsBuf)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::RegTensor<uint32_t> btmK2;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(btmK2, kValue);

    FindTargetBinAndUpdateNextK(idx2Buf, nkValueBuf, histogramsBuf, btmK2, pregB32);
}

__simd_vf__ void FindIdxEQOutputVFImpl(__ubuf__ uint32_t *outputIdxBuf, __ubuf__ uint32_t *inputBuf, uint32_t beginIdx,
                                       __ubuf__ uint32_t *kValue)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::UnalignRegForStore alignIdx;

    MicroAPI::RegTensor<uint32_t> kthValue;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(kthValue, kValue);

    MicroAPI::RegTensor<uint32_t> vregInput;

    MicroAPI::RegTensor<int32_t> idxC;
    MicroAPI::Arange(idxC, beginIdx);

    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(vregInput, inputBuf);

    MicroAPI::MaskReg poutEQ = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::RegTensor<uint32_t> sqzIdxOut;
    MicroAPI::Compare<uint32_t, CMPMODE::EQ>(poutEQ, vregInput, kthValue, pregB32);

    MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(sqzIdxOut, (MicroAPI::RegTensor<uint32_t> &)idxC,
                                                                     poutEQ);
    MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(outputIdxBuf, sqzIdxOut, alignIdx);
    MicroAPI::StoreUnAlignPost(outputIdxBuf, alignIdx);
}

__simd_vf__ void FindValueGTOutputVFImpl(__ubuf__ uint32_t *outputValueBuf, __ubuf__ uint32_t *inputBuf,
                                         __ubuf__ uint32_t *kValue, uint16_t vfLoop)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    MicroAPI::UnalignRegForStore alignValue;

    MicroAPI::RegTensor<uint32_t> kthValue;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(kthValue, kValue);

    MicroAPI::RegTensor<uint32_t> vregInput;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(vregInput, inputBuf + i * 64);

        MicroAPI::MaskReg poutGT = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

        MicroAPI::RegTensor<uint32_t> sqzValueOut;
        MicroAPI::Compare<uint32_t, CMPMODE::GT>(poutGT, vregInput, kthValue, pregB32);

        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(sqzValueOut, vregInput, poutGT);
        MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(outputValueBuf, sqzValueOut,
                                                                                  alignValue);
    }
    MicroAPI::StoreUnAlignPost(outputValueBuf, alignValue);
}

__simd_vf__ void FindValueEQOutputVFImpl(__ubuf__ uint32_t *outputValueBuf, __ubuf__ uint32_t *inputBuf,
                                         __ubuf__ uint32_t *kValue)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::UnalignRegForStore alignValue;

    MicroAPI::RegTensor<uint32_t> kthValue;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(kthValue, kValue);

    MicroAPI::RegTensor<uint32_t> vregInput;

    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(vregInput, inputBuf);

    MicroAPI::MaskReg poutEQ = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::RegTensor<uint32_t> sqzValueOut;
    MicroAPI::Compare<uint32_t, CMPMODE::EQ>(poutEQ, vregInput, kthValue, pregB32);

    MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(sqzValueOut, vregInput, poutEQ);
    MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(outputValueBuf, sqzValueOut, alignValue);
    MicroAPI::StoreUnAlignPost(outputValueBuf, alignValue);
}

__aicore__ inline void LiTopKVF(const LocalTensor<uint32_t> &outputIdxLocal,
                                const LocalTensor<uint32_t> &outputValueLocal, const LocalTensor<uint32_t> &inputLocal,
                                const LocalTensor<uint32_t> &tmpIdxLocal, const LocalTensor<uint32_t> &tmpValueLocal,
                                const LocalTensor<uint32_t> &histogramsLocal, const LocalTensor<uint32_t> &idx0Local,
                                const LocalTensor<uint32_t> &idx1Local, const LocalTensor<uint32_t> &idx2Local,
                                const LocalTensor<uint32_t> &idx3Local, const LocalTensor<uint32_t> &nkValueLocal,
                                uint32_t topK, uint32_t s2SeqLen)
{
    __ubuf__ uint32_t *outputIdxBuf = (__ubuf__ uint32_t *)outputIdxLocal.GetPhyAddr();
    __ubuf__ uint32_t *outputValueBuf = (__ubuf__ uint32_t *)outputValueLocal.GetPhyAddr();
    __ubuf__ uint32_t *inputBuf = (__ubuf__ uint32_t *)inputLocal.GetPhyAddr();
    __ubuf__ uint32_t *tmpIdxBuf = (__ubuf__ uint32_t *)tmpIdxLocal.GetPhyAddr();
    __ubuf__ uint32_t *tmpValueBuf = (__ubuf__ uint32_t *)tmpValueLocal.GetPhyAddr();
    __ubuf__ uint32_t *histogramsBuf = (__ubuf__ uint32_t *)histogramsLocal.GetPhyAddr();
    __ubuf__ uint32_t *idx0Buf = (__ubuf__ uint32_t *)idx0Local.GetPhyAddr();
    __ubuf__ uint32_t *idx1Buf = (__ubuf__ uint32_t *)idx1Local.GetPhyAddr();
    __ubuf__ uint32_t *idx2Buf = (__ubuf__ uint32_t *)idx2Local.GetPhyAddr();
    __ubuf__ uint32_t *idx3Buf = (__ubuf__ uint32_t *)idx3Local.GetPhyAddr();
    __ubuf__ uint32_t *nkValueBuf = (__ubuf__ uint32_t *)nkValueLocal.GetPhyAddr();

    uint32_t bottomK = s2SeqLen - topK + 1;
    uint32_t beginIdx = 0;
    bool flag = true;

    const uint16_t repeatSize8 = 256;
    const uint16_t repeatSize32 = 64;

    uint16_t histogramsLoopNum = (s2SeqLen + repeatSize8 - 1) / repeatSize8;
    uint16_t inputLoopNum = (s2SeqLen + repeatSize32 - 1) / repeatSize32;
    uint16_t topkLoopNum = (topK + 64 - 1) / 64;

    // find kth-value
    HistogramsFirstVFImpl<uint32_t>(histogramsBuf, inputBuf, histogramsLoopNum, flag);
    FindFirstTargetBinVFImpl(idx0Buf, nkValueBuf, histogramsBuf, bottomK);
    HistogramsSecondVFImpl<uint32_t>(histogramsBuf, inputBuf, idx0Buf, histogramsLoopNum, flag);
    FindSecondTargetBinVFImpl(idx1Buf, nkValueBuf, nkValueBuf, histogramsBuf);
    HistogramsThirdVFImpl<uint32_t>(histogramsBuf, inputBuf, idx0Buf, idx1Buf, histogramsLoopNum, flag);
    FindThirdTargetBinVFImpl(idx2Buf, nkValueBuf, nkValueBuf, histogramsBuf);
    HistogramsLastVFImpl<uint32_t>(histogramsBuf, inputBuf, idx0Buf, idx1Buf, idx2Buf, histogramsLoopNum, flag);
    FindKthVFImpl(nkValueBuf, histogramsBuf, idx0Buf, idx1Buf, idx2Buf, idx3Buf);

    // filter
    // 输出大于k-value的值value
    FindValueGTOutputVFImpl(outputValueBuf, inputBuf, nkValueBuf, inputLoopNum);
    // value-当前偏移大于k-value的值在AR特殊寄存器中的有效字节数
    int64_t arValueNum = AscendC::GetSpr<AscendC::SpecialPurposeReg::AR>();
    // value-剩余需要输出等于k-value的数量
    int64_t remainValueNum = topK - (arValueNum / sizeof(uint32_t));
    for (uint16_t i = 0; i < inputLoopNum; ++i) {
        int64_t arValueNumPerLoop = AscendC::GetSpr<AscendC::SpecialPurposeReg::AR>();
        if (((arValueNumPerLoop - arValueNum) / sizeof(uint32_t)) < remainValueNum) {
            // 调用一次查找等于k-value情况的过程；64: 单次循环处理的元素块大小
            FindValueEQOutputVFImpl(outputValueBuf, inputBuf + i * 64, nkValueBuf);
        } else {
            break;
        }
    }

    // 输出大于k-value的值idx
    FindIdxGTOutputVFImpl(outputIdxBuf, inputBuf, (uint32_t)(0), nkValueBuf, inputLoopNum);
    // idx-当前偏移大于k-value的值在AR特殊寄存器中的有效字节数
    int64_t arIdxNum = AscendC::GetSpr<AscendC::SpecialPurposeReg::AR>();
    int64_t remainIdxNum = topK - (arIdxNum / sizeof(uint32_t));
    for (uint16_t i = 0; i < inputLoopNum; ++i) {
        int64_t arIdxNumPerLoop = AscendC::GetSpr<AscendC::SpecialPurposeReg::AR>();
        if (((arIdxNumPerLoop - arIdxNum) / sizeof(uint32_t)) < remainIdxNum) {
            // 调用一次查找等于k-value情况的过程
            beginIdx = i * 64;                                                            // 64: 块起始偏移量
            FindIdxEQOutputVFImpl(outputIdxBuf, inputBuf + i * 64, beginIdx, nkValueBuf); // 64: 块起始偏移量
        } else {
            break;
        }
    }
}
} // namespace topkb32
#endif
