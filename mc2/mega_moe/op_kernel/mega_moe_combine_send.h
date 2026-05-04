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

// 按tile粒度发送（128x256），GM→UB→GM
template <typename ElementMMadOut2>
__aicore__ inline void CombineTile(
    uint32_t expertIdx, uint32_t rowStart, uint32_t rowCount,
    uint32_t colStart, uint32_t colCount, uint32_t n,
    GM_ADDR gmm2OutAddr, const Params& params,
    LocalTensor<ElementMMadOut2>& ubBuffer)
{
    GlobalTensor<int32_t> tokenPerExpertWin, preSumBeforeRankWorkSpace;
    tokenPerExpertWin.SetGlobalBuffer((__gm__ int32_t*)params.peermemInfo.ptrTokenPerExpert);
    preSumBeforeRankWorkSpace.SetGlobalBuffer((__gm__ int32_t*)params.workspaceInfo.ptrSumBeforeRank);

    GlobalTensor<ElementMMadOut2> gmm2OutGm;
    gmm2OutGm.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMMadOut2*>(gmm2OutAddr));

    int32_t preSumRankInExpert = 0;
    for (int32_t dstEpIdx = 0; dstEpIdx < params.tilingData->epWorldSize; dstEpIdx++) {
        int32_t lenRankInExpert = tokenPerExpertWin(dstEpIdx * Ops::Base::CeilAlign(
            static_cast<uint32_t>(params.tilingData->epWorldSize * params.tilingData->expertPerRank),
            static_cast<uint32_t>(ALIGN_128)) + params.tilingData->rankId * params.tilingData->expertPerRank + expertIdx);
        int32_t dstExpertOffset = preSumBeforeRankWorkSpace(dstEpIdx * params.tilingData->expertPerRank + expertIdx);
        int32_t stRankInExpert = preSumRankInExpert;
        preSumRankInExpert += lenRankInExpert;

        if (lenRankInExpert <= 0) {
            continue;
        }

        // 计算当前tile行范围与该rank的交集
        uint32_t tileRowEnd = rowStart + rowCount;
        uint32_t rankRowEnd = stRankInExpert + lenRankInExpert;
        uint32_t sendRowStart = max(rowStart, static_cast<uint32_t>(stRankInExpert));
        uint32_t sendRowEnd = min(tileRowEnd, rankRowEnd);
        if (sendRowStart >= sendRowEnd) {
            continue;
        }

        AscendC::GlobalTensor<ElementMMadOut2> gmRemoteD;
        uint64_t gmRemoteOffset = params.peermemInfo.ptrD - params.peermemInfo.ptrBase;
        __gm__ void* dstPeermemPtr = GetRankWinAddrWithOffset(dstEpIdx, gmRemoteOffset);
        gmRemoteD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMMadOut2*>(dstPeermemPtr));

        uint32_t sendRowCount = sendRowEnd - sendRowStart;
        uint32_t srcRowOffset = sendRowStart - rowStart;
        uint32_t dstRowOffset = sendRowStart - stRankInExpert;

        // GM→UB：从GM读取到UB

        // DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(m_ * topK_ * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<ElementMMadOut2> copyPadParams{false, 0U, 0U, 0U};

        AscendC::DataCopyExtParams gm2UbParams{
            static_cast<uint16_t>(sendRowCount),
            static_cast<uint32_t>(colCount * sizeof(ElementMMadOut2)),
            static_cast<uint16_t>((n - colCount) * sizeof(ElementMMadOut2)),
            0, 0};
        AscendC::DataCopyPad(ubBuffer, gmm2OutGm[(rowStart + srcRowOffset) * n + colStart], gm2UbParams, copyPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
        // UB→GM：从UB写到远端GM
        AscendC::DataCopyExtParams ub2GmParams{
            static_cast<uint16_t>(sendRowCount),
            static_cast<uint32_t>(colCount * sizeof(ElementMMadOut2)),
            static_cast<uint16_t>((n - colCount) * sizeof(ElementMMadOut2)),
            0, 0};
        AscendC::DataCopyPad(gmRemoteD[(dstExpertOffset + dstRowOffset) * n + colStart], ubBuffer, ub2GmParams);
    }
}

}  // namespace MegaMoeCombineImpl

#endif  // MEGA_MOE_COMBINE_SEND_H
