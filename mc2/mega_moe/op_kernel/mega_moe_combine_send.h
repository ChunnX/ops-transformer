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

// 一次性从GM搬整个tile到UB（gmNStride为layout对齐后的真实行跨度）
template <typename ElementMMadOut2>
__aicore__ inline void CopyTileFromGM(
    LocalTensor<ElementMMadOut2>& ubDst,
    GlobalTensor<ElementMMadOut2>& gmSrc,
    uint32_t rowStart, uint32_t rowCount, uint32_t colStart, uint32_t colCount, uint32_t gmNStride)
{
    // 从GM读取tile数据到UB
    DataCopyPadExtParams<ElementMMadOut2> copyPadParams{false, 0U, 0U, 0U};
    AscendC::DataCopyExtParams gm2UbParams{
        static_cast<uint16_t>(rowCount),
        static_cast<uint32_t>(colCount * sizeof(ElementMMadOut2)),
        static_cast<uint16_t>((gmNStride - colCount) * sizeof(ElementMMadOut2)),
        0, 0};
    AscendC::DataCopyPad(ubDst, gmSrc[rowStart * gmNStride + colStart], gm2UbParams, copyPadParams);
}

// 从UB发送到远端GM（复用stepOne的CombineTokens逻辑）
// expertMLoc: 当前 tile 第一行在 expert 内部的行号（= wave_row_offset + 调度器内 mLoc）
template <typename ElementMMadOut2, typename BlockShape>
__aicore__ inline void CombineTokens(
    uint32_t expertMLoc, uint32_t nLoc, uint32_t n, uint32_t groupIdx, uint32_t rankId,
    LocalTensor<ElementMMadOut2>& l0cOutUbGMM2, BlockShape& actualBlockShape, const Params& params)
{
    int32_t lenTile = Get<M_VALUE>(actualBlockShape);
    int32_t stTile = expertMLoc;
    int32_t edTile = stTile + lenTile;
    int32_t preSumRankInExpert = 0;
    int32_t tileOffset = 0;
    GlobalTensor<int32_t> tokenPerExpertWin, preSumBeforeRankWorkSpace;
    tokenPerExpertWin.SetGlobalBuffer((__gm__ int32_t*)params.peermemInfo.ptrTokenPerExpert);
    preSumBeforeRankWorkSpace.SetGlobalBuffer((__gm__ int32_t*)params.workspaceInfo.ptrSumBeforeRank);
    for (int32_t dstEpIdx = 0; dstEpIdx < params.tilingData->epWorldSize; dstEpIdx++) {
        // int32_t lenRankInExpert = tokenPerExpertWin(dstEpIdx * Ops::Base::CeilAlign(
        //     static_cast<uint32_t>(params.tilingData->epWorldSize * params.tilingData->expertPerRank),
        //     static_cast<uint32_t>(ALIGN_128)) + rankId * params.tilingData->expertPerRank + groupIdx);
        // int32_t dstExpertOffset = preSumBeforeRankWorkSpace(dstEpIdx * params.tilingData->expertPerRank + groupIdx);
        __gm__ int32_t* tokenPerExpertWinAddr = (__gm__ int32_t*)tokenPerExpertWin.GetPhyAddr() +
            (dstEpIdx * Ops::Base::CeilAlign(
                static_cast<uint32_t>(params.tilingData->epWorldSize * params.tilingData->expertPerRank),
                static_cast<uint32_t>(ALIGN_128)) + rankId * params.tilingData->expertPerRank + groupIdx);
        __gm__ int32_t* preSumBeforeRankWorkSpaceAddr = (__gm__ int32_t*)preSumBeforeRankWorkSpace.GetPhyAddr() +
            (dstEpIdx * params.tilingData->expertPerRank + groupIdx);
        int32_t lenRankInExpert = AscendC::ReadGmByPassDCache(tokenPerExpertWinAddr);
        int32_t dstExpertOffset = AscendC::ReadGmByPassDCache(preSumBeforeRankWorkSpaceAddr);
        int32_t stRankInExpert = preSumRankInExpert;
        int32_t edRankInExpert = stRankInExpert + lenRankInExpert;
        preSumRankInExpert += lenRankInExpert;

        if (stRankInExpert >= edTile) {
            break;
        } else if (edRankInExpert <= stTile) {
            continue;
        }

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

        // 远端偏移地址
        uint64_t gmDstOffset = (dstOffsetInExpert + dstExpertOffset) * n + nLoc;
        auto gmTileD = gmRemoteD[gmDstOffset];

        AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
        ub2GmParams.blockCount = lenData;
        ub2GmParams.blockLen = Get<N_VALUE>(actualBlockShape) * sizeof(ElementMMadOut2);
        ub2GmParams.dstStride = (n - Get<N_VALUE>(actualBlockShape)) * sizeof(ElementMMadOut2);

        AscendC::DataCopyPad(gmTileD, l0cOutUbGMM2[tileOffset * Get<N_VALUE>(actualBlockShape)], ub2GmParams);
        tileOffset += lenData;
    }
}

// CombineTile = CopyTileFromGM + CombineTokens
// mLoc: 调度器内的 wave 内部行号，用来从 wave-base 的 gmm2OutAddr 读取 UB tile
// waveRowOffset: 当前 wave 在 expert 内的起始行，用于把 mLoc 翻译成 expert 内部行号
template <typename ElementMMadOut2, typename BlockShape>
__aicore__ inline void CombineTile(
    uint32_t groupIdx, uint32_t mLoc, uint32_t nLoc, uint32_t n, uint32_t rankId,
    GM_ADDR gmm2OutAddr, const Params& params,
    LocalTensor<ElementMMadOut2>& ubBuffer, BlockShape& actualBlockShape,
    uint64_t waveRowOffset)
{
    GlobalTensor<ElementMMadOut2> gmm2OutGm;
    gmm2OutGm.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMMadOut2*>(gmm2OutAddr));

    uint32_t rowCount = Get<M_VALUE>(actualBlockShape);
    uint32_t colCount = Get<N_VALUE>(actualBlockShape);

    // // 第一步：从GM一次性搬整个tile到UB
    // if (mLoc < 2 && nLoc < 4) {
    //     PipeBarrier<PIPE_ALL>();
    //     AscendC::DataCacheCleanAndInvalid<ElementMMadOut2, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(gmm2OutGm);
    //     uint32_t base = mLoc * n + nLoc;
    //     AscendC::PRINTF("COMB GM e%d mLoc=%u nLoc=%u row=%u col=%u n=%u GM[0-7]=%f %f %f %f %xf %f %f %f\n",
    //         groupIdx, mLoc, nLoc, rowCount, colCount, n,
    //         gmm2OutGm.GetValue(base + 0), gmm2OutGm.GetValue(base + 1),
    //         gmm2OutGm.GetValue(base + 2), gmm2OutGm.GetValue(base + 3),
    //         gmm2OutGm.GetValue(base + 4), gmm2OutGm.GetValue(base + 5),
    //         gmm2OutGm.GetValue(base + 6), gmm2OutGm.GetValue(base + 7));
    // }
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    CopyTileFromGM(ubBuffer, gmm2OutGm, mLoc, rowCount, nLoc, colCount, n);
    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
    // if (mLoc < 2 && nLoc < 4) {
    //     PipeBarrier<PIPE_ALL>();
    //     AscendC::PRINTF("COMB UB e%d mLoc=%u nLoc=%u UB[0-7]=%f %f %f %f %f %f %f %f\n", groupIdx, mLoc, nLoc,
    //         ubBuffer.GetValue(0), ubBuffer.GetValue(1),
    //         ubBuffer.GetValue(2), ubBuffer.GetValue(3),
    //         ubBuffer.GetValue(4), ubBuffer.GetValue(5),
    //         ubBuffer.GetValue(6), ubBuffer.GetValue(7));
    // }

    // 第二步：从UB发送到远端GM。CombineTokens 需要 expert 粒度的行号，否则 wave>0 时
    // 会把数据按 wave 内 [0, waveM) 去匹配 rank 边界，导致写到错误的远端位置。
    uint32_t expertMLoc = mLoc + static_cast<uint32_t>(waveRowOffset);
    CombineTokens(expertMLoc, nLoc, n, groupIdx, rankId, ubBuffer, actualBlockShape, params);
}
}  // namespace MegaMoeCombineImpl

#endif  // MEGA_MOE_COMBINE_SEND_H
