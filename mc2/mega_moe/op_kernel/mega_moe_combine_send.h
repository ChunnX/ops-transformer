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
#include "../../common/op_kernel/mc2_kernel_utils.h"
using namespace AscendC;

namespace MegaMoeCombineImpl {

// 按tile粒度发送，GM→UB→GM，计算逻辑和stepOne的CombineTokens一致
template <typename ElementMMadOut2>
__aicore__ inline void CombineTile(
    uint32_t expertIdx, uint32_t rowStart, uint32_t rowCount,
    uint32_t colStart, uint32_t colCount, uint32_t n, uint32_t rankId,
    GM_ADDR gmm2OutAddr, const Params& params,
    LocalTensor<ElementMMadOut2>& ubBuffer)
{
    int32_t stTile = rowStart;
    int32_t edTile = stTile + rowCount;
    int32_t preSumRankInExpert = 0;
    int32_t tileOffset = 0;

    GlobalTensor<int32_t> tokenPerExpertWin, preSumBeforeRankWorkSpace;
    tokenPerExpertWin.SetGlobalBuffer((__gm__ int32_t*)params.peermemInfo.ptrTokenPerExpert);
    preSumBeforeRankWorkSpace.SetGlobalBuffer((__gm__ int32_t*)params.workspaceInfo.ptrSumBeforeRank);

    GlobalTensor<ElementMMadOut2> gmm2OutGm;
    gmm2OutGm.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMMadOut2*>(gmm2OutAddr));

    for (int32_t dstEpIdx = 0; dstEpIdx < params.tilingData->epWorldSize; dstEpIdx++) {
        int32_t lenRankInExpert = tokenPerExpertWin(dstEpIdx * Ops::Base::CeilAlign(
            static_cast<uint32_t>(params.tilingData->epWorldSize * params.tilingData->expertPerRank),
            static_cast<uint32_t>(ALIGN_128)) + rankId * params.tilingData->expertPerRank + expertIdx);
        int32_t dstExpertOffset = preSumBeforeRankWorkSpace(dstEpIdx * params.tilingData->expertPerRank + expertIdx);
        int32_t stRankInExpert = preSumRankInExpert;
        int32_t edRankInExpert = stRankInExpert + lenRankInExpert;
        preSumRankInExpert += lenRankInExpert;

        if (stRankInExpert >= edTile) {
            break;
        } else if (edRankInExpert <= stTile) {
            continue;
        }

        // 和stepOne一致的交集计算
        int32_t stData = Blaze::Gemm::Max(stRankInExpert, stTile);
        int32_t edData = Blaze::Gemm::Min(edRankInExpert, edTile);
        int32_t lenData = edData - stData;
        if (lenData <= 0) {
            continue;
        }

        uint32_t dstOffsetInExpert = 0;
        if (stTile > stRankInExpert) {
            dstOffsetInExpert = stTile - stRankInExpert;
        }

        AscendC::GlobalTensor<ElementMMadOut2> gmRemoteD;
        uint64_t gmRemoteOffset = params.peermemInfo.ptrD - params.peermemInfo.ptrBase;
        __gm__ void* dstPeermemPtr = GetRankWinAddrWithOffset(dstEpIdx, gmRemoteOffset);
        gmRemoteD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMMadOut2*>(dstPeermemPtr));

        // 远端偏移地址（和stepOne一致）
        uint64_t gmDstOffset = (dstOffsetInExpert + dstExpertOffset) * n + colStart;
        auto gmTileD = gmRemoteD[gmDstOffset];

        // 从GM读取到UB（源地址从GM读取，按行读取）
        DataCopyPadExtParams<ElementMMadOut2> copyPadParams{false, 0U, 0U, 0U};
        AscendC::DataCopyExtParams gm2UbParams{
            static_cast<uint16_t>(lenData),
            static_cast<uint32_t>(colCount * sizeof(ElementMMadOut2)),
            static_cast<uint16_t>((n - colCount) * sizeof(ElementMMadOut2)),
            0, 0};
        // 源地址：GM上从tileOffset行开始，colStart列开始
        AscendC::DataCopyPad(ubBuffer[tileOffset * colCount],
            gmm2OutGm[(stData) * n + colStart], gm2UbParams, copyPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_MTE3>();

        // 从UB发送到远端GM（和stepOne一致的参数）
        AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
        ub2GmParams.blockCount = lenData;
        ub2GmParams.blockLen = colCount * sizeof(ElementMMadOut2);
        ub2GmParams.dstStride = (n - colCount) * sizeof(ElementMMadOut2);

        AscendC::DataCopyPad(gmTileD, ubBuffer[tileOffset * colCount], ub2GmParams);
        tileOffset += lenData;
    }
}
}  // namespace MegaMoeCombineImpl

#endif  // MEGA_MOE_COMBINE_SEND_H
