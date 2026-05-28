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

// 批量扫描版 combine send。
// ----------------------------------------------------------------------------
// 旧版按 row 逐条发,每行 1 次 DataCopyPad,256 行的 tile 要 ~256 次 MTE3 startup,
// 单次 startup ~150ns 累积 ~38us。
//
// 新版按"同 toRankId + 同 dst-pos-stride"的连续段批量发:
//   - tile 内每行 dst pos = tokenIdx * topK + topkIdx,在 dispatchRevData 顺序下
//     单调递增(stripe 写入 + mask 顺序返回保证)。
//   - 同 src/dst rank 的连续段内,如果相邻行 dst pos 差是常数,就用 DataCopyPad
//     的 blockCount + dstStride 一把发掉(blockLen 写整行 N 个元素,dstStride
//     补齐到 stride_pos × n 个元素之间的字节间隔)。
//   - 一旦 toRankId 变 / stride 变,就 flush 当前段、起新段。
//
// 最差情况(随机路由,每对相邻行 stride 都变):退化成每段 1 行,与旧版相同。
// 典型情况(一个 token 多个 topk 都落同一 expert,或连续 token 同 expert 同 topk
// 槽):runLen 普遍 2-8,MTE3 op 数减少到原来的 1/3 - 1/8。
//
// ⚠️ 已知问题:此版本在实测中精度对比下"大部分 token 稍微差一点,少数对",
// 总耗时下降但输出错误。怀疑 stride>1 的多行 DataCopyPad 在某些条件下
// dst 落点跟我推导的公式不一致(详见 archive/profiling-analysis.md 报告)。
// 保留此版本用于后续 print-debug;若要先恢复精度,改成只合并 stride=1 的
// 保守版本即可(见 pr_5806 的 CombineTokens 实现)。
// ----------------------------------------------------------------------------
template <typename ElementMMadOut2, typename BlockShape>
__aicore__ inline void CombineTokensOptimize(
    uint32_t mLoc, uint32_t nLoc, uint32_t n, uint32_t expertIdx, uint32_t rankId, LocalTensor<int32_t>& tripleTensor,
    LocalTensor<ElementMMadOut2>& l0cOutUbGMM2, BlockShape& actualBlockShape, const ParamsOptimize& params)
{
    int32_t lenTile = Get<M_VALUE>(actualBlockShape);
    uint32_t N = static_cast<uint32_t>(Get<N_VALUE>(actualBlockShape));
    uint32_t topK = static_cast<uint32_t>(params.tilingData->topK);
    AscendC::GlobalTensor<ElementMMadOut2> gmRemoteD;
    uint64_t gmRemoteBaseOffset =
        params.peermemInfo.combineSendPtr - params.peermemInfo.rankSyncInWorldPtr;
    constexpr uint64_t kElemBytes = sizeof(ElementMMadOut2);

    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockLen = N * kElemBytes;

    int32_t i = 0;
    while (i < lenTile) {
        // 段首
        uint32_t curRank = tripleTensor.GetValue(i * 8);
        uint32_t tokI = tripleTensor.GetValue(i * 8 + 1);
        uint32_t tkI = tripleTensor.GetValue(i * 8 + 2);
        uint64_t posI = static_cast<uint64_t>(tokI) * topK + tkI;
        uint64_t prevPos = posI;

        // 找尾部能并进同一段的最大 j
        //   段内要求: tripleTensor[k].rank == curRank, 且相邻 row 的 dst-pos 差恒定。
        //   stridePos < 0 时表示段内只有 1 行,还没确定 stride。
        int32_t j = i + 1;
        int64_t stridePos = -1;
        while (j < lenTile) {
            uint32_t r = tripleTensor.GetValue(j * 8);
            if (r != curRank) break;
            uint32_t tok = tripleTensor.GetValue(j * 8 + 1);
            uint32_t tk  = tripleTensor.GetValue(j * 8 + 2);
            uint64_t pos = static_cast<uint64_t>(tok) * topK + tk;
            int64_t step = static_cast<int64_t>(pos) - static_cast<int64_t>(prevPos);
            if (stridePos < 0) {
                stridePos = step;
            } else if (step != stridePos) {
                break;
            }
            prevPos = pos;
            ++j;
        }

        int32_t runLen = j - i;
        gmRemoteD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMMadOut2*>(
            GetRankWinAddrWithOffset(curRank, gmRemoteBaseOffset)));
        uint64_t gmDstOffset = posI * n + nLoc;

        ub2GmParams.blockCount = static_cast<uint16_t>(runLen);
        if (runLen > 1) {
            // dst 第 k 行起点(字节) = (posI + k * stridePos) * n * kElemBytes + nLoc * kElemBytes
            // DataCopyPad 第 k 行起点(字节) = k * (blockLen + dstStride) = k * (N + dstStride/kElemBytes) * kElemBytes
            //   ⇒ N + dstStride/kElemBytes = stridePos * n
            //   ⇒ dstStride = (stridePos * n - N) * kElemBytes
            ub2GmParams.dstStride = static_cast<uint32_t>(
                static_cast<uint64_t>(stridePos) * static_cast<uint64_t>(n) * kElemBytes
                - static_cast<uint64_t>(N) * kElemBytes);
        } else {
            ub2GmParams.dstStride = 0;
        }

        AscendC::DataCopyPad(gmRemoteD[gmDstOffset], l0cOutUbGMM2[i * N], ub2GmParams);

        i = j;
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