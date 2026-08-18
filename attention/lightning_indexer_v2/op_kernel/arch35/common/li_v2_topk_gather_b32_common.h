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

} // namespace liTopkB32GatherCommon

#endif // LIGHTNING_INDEXER_V2_TOPK_GATHER_B32_COMMON_H
