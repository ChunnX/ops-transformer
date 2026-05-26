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
__aicore__ inline void CombineTokensOptimize(
    uint32_t mLoc, uint32_t nLoc, uint32_t n, uint32_t expertIdx, uint32_t rankId, LocalTensor<int32_t>& tripleTensor,
    LocalTensor<ElementMMadOut2>& l0cOutUbGMM2, BlockShape& actualBlockShape, const ParamsOptimize& params)
{
    int32_t lenTile = Get<M_VALUE>(actualBlockShape);
    AscendC::GlobalTensor<ElementMMadOut2> gmRemoteD;
    uint64_t gmRemoteBaseOffset = params.peermemInfo.combineSendPtr - params.peermemInfo.rankSyncInWorldPtr; // rankSyncInWorldPtr是起始地址
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockCount = 1;
    ub2GmParams.blockLen = Get<N_VALUE>(actualBlockShape) * sizeof(ElementMMadOut2);        // N_VALUE是当前tile块的n长度
    for (int32_t tileIdx = 0; tileIdx < lenTile; ++tileIdx) {
        uint32_t toRankId = tripleTensor.GetValue(tileIdx * 8);
        uint32_t tokenIdx = tripleTensor.GetValue(tileIdx * 8 + 1);
        uint32_t topkIdx = tripleTensor.GetValue(tileIdx * 8 + 2);
        gmRemoteD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMMadOut2*>(GetRankWinAddrWithOffset(toRankId, gmRemoteBaseOffset)));
        uint64_t gmDstOffset = (tokenIdx * params.tilingData->topK + topkIdx) * n + nLoc;
        AscendC::DataCopyPad(gmRemoteD[gmDstOffset], l0cOutUbGMM2[tileIdx * Get<N_VALUE>(actualBlockShape)], ub2GmParams);
    }
}
// 把整个 tile 从 GM (gmm2OutAddr) 搬到 UB
template <typename ElementMMadOut2>
__aicore__ inline void CopyTileFromGM(
    LocalTensor<ElementMMadOut2>& ubDst,
    GlobalTensor<ElementMMadOut2>& gmSrc,
    uint32_t rowStart, uint32_t rowCount, uint32_t colStart, uint32_t colCount, uint32_t gmNStride)
{
    DataCopyPadExtParams<ElementMMadOut2> copyPadParams{false, 0U, 0U, 0U};
    AscendC::DataCopyExtParams gm2UbParams{
        static_cast<uint16_t>(rowCount),
        static_cast<uint32_t>(colCount * sizeof(ElementMMadOut2)),
        static_cast<uint16_t>((gmNStride - colCount) * sizeof(ElementMMadOut2)),  // 跟 pr_5806 一致
        0, 0};
    AscendC::DataCopyPad(ubDst, gmSrc[rowStart * gmNStride + colStart], gm2UbParams, copyPadParams);
}

// CombineTileOptimize = 读 GM tile 到 UB + 加载 triple + CombineTokensOptimize
// groupCnt: 当前 expert 在 tripleInfoPtr 中的累计起始 token 数（含 waveRowOffset）
// waveTraceIdx / isFirstTile: 用于细分打点（CopyTileFromGM 完成、triple 完成）
template <typename ElementMMadOut2, typename BlockShape>
__aicore__ inline void CombineTileOptimize(
    uint32_t mLoc, uint32_t nLoc, uint32_t n, uint32_t expertIdx, uint32_t rankId,
    GM_ADDR gmm2OutAddr, LocalTensor<ElementMMadOut2>& ubBuffer,
    BlockShape& actualBlockShape, const ParamsOptimize& params, uint32_t groupCnt,
    int waveTraceIdx = -1, bool isFirstTile = false)
{
    GlobalTensor<ElementMMadOut2> gmm2OutGm;
    gmm2OutGm.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMMadOut2*>(gmm2OutAddr));

    uint32_t rowCount = Get<M_VALUE>(actualBlockShape);
    uint32_t colCount = Get<N_VALUE>(actualBlockShape);

    SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, 0>();
    CopyTileFromGM(ubBuffer, gmm2OutGm, mLoc, rowCount, nLoc, colCount, n);
    if (isFirstTile) {
        MegaMoeTrace::RecordWavePoint(params.y2GmAddr, waveTraceIdx,
                                      MegaMoeTrace::SP_GMM2_CB_AFTER_COPY_TILE);
    }

    // 从 tripleInfoPtr 加载本 tile 的三元组信息（groupCnt 是 expert + wave 累计起点）
    AscendC::GlobalTensor<int32_t> tripleGm;
    LocalTensor<int32_t> tripleTensor = LocalTensor<int32_t>(TPosition::VECCALC, 200 * 1024, rowCount * 8);
    tripleGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
        params.workspaceInfo.tripleInfoPtr + (groupCnt + mLoc) * 32));
    AscendC::DataCopy(tripleTensor, tripleGm, rowCount * 8);
    SyncFuncStatic<AscendC::HardEvent::MTE2_S, 0>();
    if (isFirstTile) {
        MegaMoeTrace::RecordWavePoint(params.y2GmAddr, waveTraceIdx,
                                      MegaMoeTrace::SP_GMM2_CB_AFTER_TRIPLE);
    }

    CombineTokensOptimize<ElementMMadOut2, BlockShape>(
        mLoc, nLoc, n, expertIdx, rankId, tripleTensor, ubBuffer, actualBlockShape, params);
}

}  // namespace MegaMoeCombineImpl

#endif  // MEGA_MOE_COMBINE_SEND_H