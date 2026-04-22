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

#include "catlass/gemm_coord.hpp"
#include "kernel_operator.h"
#include "mega_moe_base.h"
using namespace AscendC;

namespace MegaMoeImpl {
template <typename ElementMMadOut2, typename GemmCoord>
__aicore__ inline void CombineTokens(
    uint32_t mLoc, uint32_t nLoc, uint32_t n, uint32_t groupIdx, 
    LocalTensor<ElementMMadOut2>& l0cOutUbGMM2, GemmCoord& actualBlockShape, const Params& params)
{
    int32_t lenTile = actualBlockShape.m();
    int32_t stTile = mLoc;
    int32_t edTile = stTile + lenTile;
    int32_t preSumRankInExpert = 0;
    int32_t tileOffset = 0;
    GlobalTensor<int32_t> tokenPerExpertWin_, preSumBeforeRankWorkSpace_;
    tokenPerExpertWin_.SetGlobalBuffer((__gm__ int32_t*)params.peermemInfo.ptrTokenPerExpert);
    preSumBeforeRankWorkSpace_.SetGlobalBuffer((__gm__ int32_t*)params.workspaceInfo.ptrSumBeforeRank);
    for (int32_t dstEpIdx = 0; dstEpIdx < params.tilingData->EP; dstEpIdx++) {
        uint32_t lenRankInExpert = tokenPerExpertWin_(dstEpIdx * CeilAlign(params.tilingData->EP * 
                                    params.tilingData->expertPerRank, ALIGN_128) + groupIdx);
        uint32_t dstExpertOffset = preSumBeforeRankWorkSpace_(dstEpIdx * params.tilingData->expertPerRank + groupIdx);
        uint32_t stRankInExpert = preSumRankInExpert;
        uint32_t edRankInExpert = stRankInExpert + lenRankInExpert;
        preSumRankInExpert += lenRankInExpert;

        if (stRankInExpert >= edTile) {
            break;
        } else if (edRankInExpert <= stTile) {
            continue;
        }

        uint32_t stData = tla::max(stRankInExpert, stTile); 
        uint32_t edData = tla::min(edRankInExpert, edTile);
        uint32_t lenData = edData - stData;
        if (lenData <= 0){
            continue;
        }

        uint32_t dstOffsetInExpert = 0;
        if (stTile > stRankInExpert) {
            dstOffsetInExpert = stTile - stRankInExpert;
        }

        AscendC::GlobalTensor<ElementMMadOut2> gmRemoteD;
        uint64_t gmRemoteOffset = params.peermemInfo.ptrD - winRankAddr_[dstEpIdx];
        __gm__ void* dstPeermemPtr = GetRankWinAddrWithOffset(dstEpIdx, gmRemoteOffset);
        gmRemoteD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMMadOut2*>(dstPeermemPtr));
        
        // 远端偏移地址
        uint64_t gmDstOffset = (dstOffsetInExpert + dstExpertOffset) * n + nLoc;
        auto gmTileD = gmRemoteD[gmDstOffset];

        AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
        ub2GmParams.blockCount = lenData;
        ub2GmParams.blockLen = actualBlockShape.n() * sizeof(ElementMMadOut2);
        ub2GmParams.dstStride = (n - actualBlockShape.n()) * sizeof(ElementMMadOut2);

        AscendC::DataCopyPad(gmTileD, l0cOutUbGMM2[tileOffset *  actualBlockShape.n()], ub2GmParams);
        tileOffset += lenData;
    }
}
}  // namespace MegaMoeImpl

#endif  // MEGA_MOE_COMBINE_SEND_H