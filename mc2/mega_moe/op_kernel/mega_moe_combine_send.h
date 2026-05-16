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
 * \brief
 */

#ifndef MEGA_MOE_COMBINE_SEND_H
#define MEGA_MOE_COMBINE_SEND_H

#include "kernel_operator.h"
#include "mega_moe_base.h"
using namespace AscendC;

namespace MegaMoeCombineImpl {
template <typename ElementMMadOut2, typename BlockShape>
__aicore__ inline void CombineTokens(
    uint32_t mLoc, uint32_t nLoc, uint32_t n, uint32_t groupIdx, uint32_t rankId, uint32_t mBaseOffset,
    LocalTensor<ElementMMadOut2>& l0cOutUbGMM2, BlockShape& actualBlockShape, const Params& params)
{
    int32_t lenTile = Get<M_VALUE>(actualBlockShape);
    uint64_t gmRemoteOffset = params.peermemInfo.ptrD - params.peermemInfo.ptrBase;
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockCount = 1;
    ub2GmParams.blockLen = Get<N_VALUE>(actualBlockShape) * sizeof(ElementMMadOut2);
    for (int32_t tileRow = 0; tileRow < lenTile; ++tileRow) {
        uint32_t tripletRow = mBaseOffset + mLoc + tileRow;
        int32_t packedTriplet = ReadGmByPassDCache((__gm__ int32_t*)params.workspaceInfo.ptrTriplet + tripletRow);
        uint32_t toRankId = DecodeTripletRank(packedTriplet);
        uint32_t tokenId = DecodeTripletToken(packedTriplet);
        uint32_t topkId = DecodeTripletTopk(packedTriplet);

        AscendC::GlobalTensor<ElementMMadOut2> gmRemoteD;
        __gm__ void* dstPeermemPtr = GetRankWinAddrWithOffset(toRankId, gmRemoteOffset);
        gmRemoteD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMMadOut2*>(dstPeermemPtr));
        uint64_t gmDstOffset = (tokenId * params.tilingData->topK + topkId) * n + nLoc;
        auto gmRowD = gmRemoteD[gmDstOffset];
        AscendC::DataCopyPad(gmRowD, l0cOutUbGMM2[tileRow * Get<N_VALUE>(actualBlockShape)], ub2GmParams);
        DataSyncBarrier<MemDsbT::DDR>();
        DataCacheCleanAndInvalid<ElementMMadOut2, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(gmRemoteD);
        __gm__ int32_t* remoteRowReady = (__gm__ int32_t*)GetRankWinAddrWithOffset(toRankId, 48 * 1024);
        uint32_t rowIdx = tokenId * params.tilingData->topK + topkId;
        AscendC::AtomicAdd(remoteRowReady + rowIdx, 1);
    }
}
}  // namespace MegaMoeCombineImpl

#endif  // MEGA_MOE_COMBINE_SEND_H
