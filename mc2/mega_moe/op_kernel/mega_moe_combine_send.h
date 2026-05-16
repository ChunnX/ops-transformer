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
    uint32_t mLoc, uint32_t nLoc, uint32_t n, uint32_t groupCnt, uint32_t expertIdx, uint32_t rankId,
    LocalTensor<ElementMMadOut2>& l0cOutUbGMM2, BlockShape& actualBlockShape, const Params& params)
{
    uint64_t t0 = GetSystemCycle();
    int32_t lenTile = Get<M_VALUE>(actualBlockShape);
    AscendC::GlobalTensor<ElementMMadOut2> gmRemoteD;
    uint64_t gmRemoteBaseOffset = params.peermemInfo.combineSendPtr - params.peermemInfo.rankSyncInWorldPtr; // rankSyncInWorldPtr是起始地址
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockCount = 1;
    ub2GmParams.blockLen = Get<N_VALUE>(actualBlockShape) * sizeof(ElementMMadOut2);        // N_VALUE是当前tile块的n长度
    for (int32_t tileIdx = 0; tileIdx < lenTile; ++tileIdx) {
        uint32_t tripletRow = mLoc + tileIdx; // m方向起始偏移 + tileIdx
        uint32_t toRankId = ReadGmByPassDCache((__gm__ int32_t*)(params.workspaceInfo.tripleInfoPtr + (groupCnt + tripletRow) * 32));
        uint32_t tokenIdx = ReadGmByPassDCache((__gm__ int32_t*)(params.workspaceInfo.tripleInfoPtr + (groupCnt + tripletRow) * 32 + 4));
        uint32_t topkIdx = ReadGmByPassDCache((__gm__ int32_t*)(params.workspaceInfo.tripleInfoPtr + (groupCnt + tripletRow) * 32 + 8));

        gmRemoteD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMMadOut2*>(GetRankWinAddrWithOffset(toRankId, gmRemoteBaseOffset)));
        uint64_t gmDstOffset = (tokenIdx * params.tilingData->topK + topkIdx) * n + nLoc;
        AscendC::DataCopyPad(gmRemoteD[gmDstOffset], l0cOutUbGMM2[tileIdx * Get<N_VALUE>(actualBlockShape)], ub2GmParams);
    }
    uint64_t t1 = GetSystemCycle();
    if (expertIdx < 4) {
        params.timeCombineTokensPerExpert[expertIdx] += t1 - t0;
    }
}
}  // namespace MegaMoeCombineImpl

#endif  // MEGA_MOE_COMBINE_SEND_H