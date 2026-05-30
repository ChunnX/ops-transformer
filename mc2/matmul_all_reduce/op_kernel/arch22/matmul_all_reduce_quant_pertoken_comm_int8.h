/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_all_reduce_quant_pertoken_comm_int8.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_QUANT_PERTOKEN_COMM_INT8_H
#define MATMUL_ALL_REDUCE_QUANT_PERTOKEN_COMM_INT8_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#ifdef __CCE_KT_TEST__
#include "rac_server_stub.h"
#else
#include "lib/hccl/hccl.h"
#endif
#include "../common.h"
#include "../../3rd/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_pertoken.h"
#include "matmul_all_reduce_add_x3.h"
#include "matmul_all_reduce_quant_perchannel.h"
#include "matmul_all_reduce_dequant_perchannel.h"
#include "matmul_all_reduce_quant_reduce_sum.h"
#include "quant_matmul_all_reduce_tiling_data.h"

constexpr uint32_t MAX_HANDLE_ID_NUM = 16;
constexpr uint32_t NUM_TWO_PERTOKEN = 2;

namespace MatmulAllReduceImpl {
using namespace AscendC;
using namespace MatmulAllReduceReduceSumImpl;
using namespace MatmulAllReduceQuantPerchannelImpl;
using namespace MatmulAllReduceDequantPerchannelImpl;
template <typename xType, typename wType, int fFormat, int wFormat, typename scaleType, typename yType,
          typename commType, bool aTrans, bool bTrans>
class MatmulAllReduceQuantPertokenInt8 {
public:
    __aicore__ inline MatmulAllReduceQuantPertokenInt8()
    {
    }
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR addGM, GM_ADDR dequantFactorGM,
                                GM_ADDR pertokenScaleGM, GM_ADDR commQuantScale1GM, GM_ADDR commQuantScale2GM,
                                GM_ADDR cGM, GM_ADDR workspaceGM, Mc2Tiling::QuantMatmulAllReduceTilingData *tilDat,
                                TPipe *tPipe);
    __aicore__ inline void
    Process(BmmDequantPertoken<xType, wType, fFormat, wFormat, scaleType, yType, aTrans, bTrans, true> &opTile,
            BmmDequantPertoken<xType, wType, fFormat, wFormat, scaleType, yType, aTrans, bTrans, true> &opTail);

private:
    __aicore__ inline void
    InnerProcess(BmmDequantPertoken<xType, wType, fFormat, wFormat, scaleType, yType, aTrans, bTrans, true> &op,
                 uint32_t tileCnt, uint32_t padM, Mc2QuantBatchMatmulV3TilingData *mmTil16, uint32_t isAdd,
                 uint32_t needUbBuffer, bool isTailFlag);
    __aicore__ inline void PrepareInit();
    __aicore__ inline uint32_t SendCountCheck(uint32_t prepareIndex);

    Mc2Tiling::QuantMatmulAllReduceTilingData *tilDat_;
    TPipe *tPipe_;
    GM_ADDR aGM_;
    GM_ADDR bGM_;
    GM_ADDR addGM_;
    GM_ADDR biasGM_;
    GM_ADDR dqFactorGM_;
    GM_ADDR pertokenSclGM_;
    GM_ADDR commQScale1GM_;
    GM_ADDR commQScale2GM_;
    GM_ADDR cGM_;
    GM_ADDR workspaceGM_;
    GM_ADDR outGM_;
    GM_ADDR tempBuffWinOrGM_;
    GM_ADDR tempBuffGM_;
    GM_ADDR tempBuffDequantGM_;
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;

    // 仅在0核上使用
    AscendC::HcclHandle allToAllHandleId_[MAX_HANDLE_ID_NUM] = {0};
    AscendC::HcclHandle allGatherHandleId_[MAX_HANDLE_ID_NUM] = {0};
    GM_ADDR allToAllsendGM_[MAX_HANDLE_ID_NUM] = {0};
    GM_ADDR aGatherSendGM_[MAX_HANDLE_ID_NUM] = {0};
    GM_ADDR allRecvOutGM_[MAX_HANDLE_ID_NUM] = {0};
    int alltoAllCommitIdx_ = 0;
    int allToAllWaitIdx_ = 0;
    int allGatherCommitIdx_ = 0;
    int allGatherWaitIdx_ = 0;
    // 所有核
    bool isSendTileFlag_ = false;
    uint32_t tilePadM = 0;
    uint32_t tailPadM = 0;
    uint32_t tilePadDataCnt = 0;
    uint32_t tailPadDataCnt = 0;
};

template <typename xType, typename wType, int fFormat, int wFormat, typename scaleType, typename yType,
          typename commType, bool aTrans, bool bTrans>
__aicore__ inline void
MatmulAllReduceQuantPertokenInt8<xType, wType, fFormat, wFormat, scaleType, yType, commType, aTrans, bTrans>::
    InnerProcess(BmmDequantPertoken<xType, wType, fFormat, wFormat, scaleType, yType, aTrans, bTrans, true> &op,
                 uint32_t tileCnt, uint32_t curPadM, Mc2QuantBatchMatmulV3TilingData *mmTil16, uint32_t isAdd,
                 uint32_t needUbBuffer, bool isTailFlag)
{
    const uint64_t aOffset = CalcShapeOffset(sizeof(xType), mmTil16->matmulTiling.M, mmTil16->matmulTiling.Ka);
    const uint64_t cOffset = CalcShapeOffset(sizeof(yType), mmTil16->matmulTiling.M, mmTil16->matmulTiling.N);
    const uint64_t mOffset = CalcShapeOffset(sizeof(int8_t), curPadM, mmTil16->matmulTiling.N);
    const uint64_t pertokenOffset = sizeof(float) * mmTil16->matmulTiling.M;
    for (uint32_t i = 1U; i <= tileCnt; ++i) {
        tPipe_->Reset();
        op.Init(aGM_, bGM_, biasGM_, dqFactorGM_, pertokenSclGM_, cGM_, workspaceGM_, mmTil16, tPipe_);
        op.UpdateGlobalAddr(aGM_, bGM_, biasGM_, dqFactorGM_, pertokenSclGM_, cGM_, workspaceGM_);
        op.Process();
        if (isAdd) {
            SyncAll();
            Matmul_All_Reduce_Add_X3<yType>(cGM_, addGM_, cOffset / sizeof(yType), tilDat_->param.addX3UbCnt,
                                            tPipe_);
            addGM_ += cOffset;
        }
        SyncAll();
        MatmulAllReduceQuantPerchannelCommInt8<yType>(cGM_, commQScale1GM_, tempBuffWinOrGM_, tPipe_,
                                                      mmTil16->matmulTiling.N, mmTil16->matmulTiling.M);
        SyncAll();

        if (GetBlockIdx() == 0) {
            hccl_.Commit(allToAllHandleId_[alltoAllCommitIdx_]);
            alltoAllCommitIdx_++;
        }
        if (isSendTileFlag_) {
            if (GetBlockIdx() == 0) {
                hccl_.Wait(allToAllHandleId_[allToAllWaitIdx_]);
            }
            SyncAll();
            if (isTailFlag && i == 1U) {
                MatmulAllReduceReduceSumInt8<yType>(tempBuffGM_, commQScale1GM_, commQScale2GM_, tilePadM,
                                                    mmTil16->matmulTiling.N, tPipe_, hccl_);
                tempBuffGM_ += tilePadM * mmTil16->matmulTiling.N;
            } else {
                MatmulAllReduceReduceSumInt8<yType>(tempBuffGM_, commQScale1GM_, commQScale2GM_, curPadM,
                                                    mmTil16->matmulTiling.N, tPipe_, hccl_);
                tempBuffGM_ += curPadM * mmTil16->matmulTiling.N;
            }
            SyncAll();
            if (GetBlockIdx() == 0) {
                hccl_.Commit(allGatherHandleId_[allToAllWaitIdx_]);
                allToAllWaitIdx_++;
            }
        }

        aGM_ += aOffset;
        cGM_ += cOffset;
        pertokenSclGM_ += pertokenOffset;
        tempBuffWinOrGM_ += mOffset;
        isSendTileFlag_ = true;
    }
}

template <typename xType, typename wType, int fFormat, int wFormat, typename scaleType, typename yType,
          typename commType, bool aTrans, bool bTrans>
__aicore__ inline void
MatmulAllReduceQuantPertokenInt8<xType, wType, fFormat, wFormat, scaleType, yType, commType, aTrans, bTrans>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR addGM, GM_ADDR dequantFactorGM, GM_ADDR pertokenScaleGM,
    GM_ADDR commQuantScale1GM, GM_ADDR commQuantScale2GM, GM_ADDR cGM, GM_ADDR workspaceGM,
    Mc2Tiling::QuantMatmulAllReduceTilingData *tilDat, TPipe *tPipe)
{
    __gm__ HcclCombinOpParam *context = (__gm__ HcclCombinOpParam *)(GetHcclContext<0>());
    OOMInit(context);
    hccl_.Init(GetHcclContext<0>());
    cGM_ = cGM;
    tilDat_ = tilDat;
    tPipe_ = tPipe;
    aGM_ = aGM;
    bGM_ = bGM;
    addGM_ = addGM;
    biasGM_ = biasGM;
    dqFactorGM_ = dequantFactorGM;
    pertokenSclGM_ = pertokenScaleGM;
    commQScale1GM_ = commQuantScale1GM;
    commQScale2GM_ = commQuantScale2GM;
    workspaceGM_ = workspaceGM;
    outGM_ = cGM;
    tempBuffGM_ = workspaceGM_ + tilDat_->param.commWorkSpaceSize;
    tempBuffDequantGM_ = workspaceGM_ + tilDat_->param.commWorkSpaceSize;

    // tiling 侧控制不走winTowin, 赋值tempBuffWinOrGM_
    if (tilDat->msg.useBufferType == MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_WINDOW_IN) {
        tempBuffWinOrGM_ = hccl_.GetWindowsInAddr(hccl_.GetRankId());
    } else {
        tempBuffWinOrGM_ = tempBuffGM_;
    }
}

template <typename xType, typename wType, int fFormat, int wFormat, typename scaleType, typename yType,
          typename commType, bool aTrans, bool bTrans>
__aicore__ inline uint32_t MatmulAllReduceQuantPertokenInt8<xType, wType, fFormat, wFormat, scaleType, yType, commType,
                                                            aTrans, bTrans>::SendCountCheck(uint32_t prepareIndex)
{
    uint32_t txCnt = tilePadDataCnt / tilDat_->param.rankDim;
    if (prepareIndex >= tilDat_->param.tileCnt) {
        txCnt = tailPadDataCnt / tilDat_->param.rankDim;
    }
    return txCnt;
}

template <typename xType, typename wType, int fFormat, int wFormat, typename scaleType, typename yType,
          typename commType, bool aTrans, bool bTrans>
__aicore__ inline void MatmulAllReduceQuantPertokenInt8<xType, wType, fFormat, wFormat, scaleType, yType, commType,
                                                        aTrans, bTrans>::PrepareInit()
{
    auto &&mc2Cfg = tilDat_->param;
    uint32_t rankCnt = mc2Cfg.rankDim;
    uint32_t tileMatM = tilDat_->tilematmulTiling.matmulTiling.M; // 头块pad大小更新
    tilePadM = tileMatM;
    if (tileMatM % rankCnt != 0) { // 按照M来pad
        tilePadM += rankCnt - (tileMatM % rankCnt);
    }
    uint32_t tailMatM = tilDat_->tailmatmulTiling.matmulTiling.M; // 尾块pad大小更新
    tailPadM = tailMatM;
    if (tailMatM % rankCnt != 0) {
        tailPadM += rankCnt - (tailMatM % rankCnt);
    }

    tilePadDataCnt = tilePadM * tilDat_->tilematmulTiling.matmulTiling.N;
    tailPadDataCnt = tailPadM * tilDat_->tailmatmulTiling.matmulTiling.N;
    const uint64_t tempBufOffsetTilePad =
        tilePadDataCnt * sizeof(int8_t); // 偏移计算，gather也是gather所有，dequant时只取有效位
    const uint64_t tempBufOffsetTailPad = tailPadDataCnt * sizeof(int8_t);
    const uint64_t tempBufOffsetTilePadSingle = tempBufOffsetTilePad / rankCnt;
    const uint64_t tempBufOffsetTailPadSingle = tempBufOffsetTailPad / rankCnt;

    uint32_t idx_ = 0;
    for (uint32_t idx = 0; idx < mc2Cfg.tileCnt; i++) { // 头块偏移
        const uint64_t indexOffsetTile = i * tempBufOffsetTilePad;
        const uint64_t indexGatherOffsetTile = tempBufOffsetTilePadSingle * hccl_.GetRankId();
        allToAllsendGM_[idx_] = tempBuffWinOrGM_ + indexOffsetTile;
        allRecvOutGM_[idx_] = tempBuffGM_ + indexOffsetTile;
        aGatherSendGM_[idx_] = tempBuffGM_ + indexOffsetTile + indexGatherOffsetTile;
        idx_++;
    }
    for (uint32_t idx = 0; idx < mc2Cfg.tailCnt; i++) { // 尾块偏移
        const uint64_t indexOffsetTail = mc2Cfg.tileCnt * tempBufOffsetTilePad + i * tempBufOffsetTailPad;
        const uint64_t indexGatherOffsetTail = tempBufOffsetTailPadSingle * hccl_.GetRankId();
        allToAllsendGM_[idx_] = tempBuffWinOrGM_ + indexOffsetTail;
        allRecvOutGM_[idx_] = tempBuffGM_ + indexOffsetTail;
        aGatherSendGM_[idx_] = tempBuffGM_ + indexOffsetTail + indexGatherOffsetTail;
        idx_++;
    }

    if (GetBlockIdx() == 0) { // V核的0核下发通信任务
        int allToAllIdx_ = 0;
        int allGatherIdx_ = 0;
        int N = (mc2Cfg.tileCnt + mc2Cfg.tailCnt) / NUM_TWO_PERTOKEN;
        int ReN = (mc2Cfg.tileCnt + mc2Cfg.tailCnt) % NUM_TWO_PERTOKEN;

        for (uint32_t idx = 0; idx < N; i++) { // 按总核数下发
            allToAllHandleId_[allToAllIdx_] =
                hccl_.AlltoAll<false>(allToAllsendGM_[allToAllIdx_], allRecvOutGM_[allToAllIdx_],
                                      SendCountCheck(allToAllIdx_), AscendC::HCCL_DATA_TYPE_INT8);
            allToAllIdx_++;
            allToAllHandleId_[allToAllIdx_] =
                hccl_.AlltoAll<false>(allToAllsendGM_[allToAllIdx_], allRecvOutGM_[allToAllIdx_],
                                      SendCountCheck(allToAllIdx_), AscendC::HCCL_DATA_TYPE_INT8);
            allToAllIdx_++;
            allGatherHandleId_[allGatherIdx_] =
                hccl_.AllGather<false>(aGatherSendGM_[allGatherIdx_], allRecvOutGM_[allGatherIdx_],
                                       SendCountCheck(allGatherIdx_), AscendC::HCCL_DATA_TYPE_INT8, 0);
            allGatherIdx_++;
            allGatherHandleId_[allGatherIdx_] =
                hccl_.AllGather<false>(aGatherSendGM_[allGatherIdx_], allRecvOutGM_[allGatherIdx_],
                                       SendCountCheck(allGatherIdx_), AscendC::HCCL_DATA_TYPE_INT8, 0);
            allGatherIdx_++;
        }

        if (ReN != 0) { // 余数下发
            allToAllHandleId_[allToAllIdx_] =
                hccl_.AlltoAll<false>(allToAllsendGM_[allToAllIdx_], allRecvOutGM_[allToAllIdx_],
                                      SendCountCheck(allToAllIdx_), AscendC::HCCL_DATA_TYPE_INT8);
            allToAllIdx_++;
            allGatherHandleId_[allGatherIdx_] =
                hccl_.AllGather<false>(aGatherSendGM_[allGatherIdx_], allRecvOutGM_[allGatherIdx_],
                                       SendCountCheck(allGatherIdx_), AscendC::HCCL_DATA_TYPE_INT8, 0);
            allGatherIdx_++;
        }
    }
}

template <typename xType, typename wType, int fFormat, int wFormat, typename scaleType, typename yType,
          typename commType, bool aTrans, bool bTrans>
__aicore__ inline void
MatmulAllReduceQuantPertokenInt8<xType, wType, fFormat, wFormat, scaleType, yType, commType, aTrans, bTrans>::Process(
    BmmDequantPertoken<xType, wType, fFormat, wFormat, scaleType, yType, aTrans, bTrans, true> &opTile,
    BmmDequantPertoken<xType, wType, fFormat, wFormat, scaleType, yType, aTrans, bTrans, true> &opTail)
{
    auto &&mc2Cfg = tilDat_->param;
    // all2all allGather prepare + InnerProcess
    PrepareInit();
    InnerProcess(opTile, mc2Cfg.tileCnt, tilePadM, &tilDat_->tilematmulTiling, mc2Cfg.isAdd,
                 mc2Cfg.needUbBuffer, false);
    if (mc2Cfg.tailMatM != 0U) {
        InnerProcess(opTail, mc2Cfg.tailCnt, tailPadM, &tilDat_->tailmatmulTiling, mc2Cfg.isAdd,
                     mc2Cfg.needUbBuffer, true);
    }
    if (GetBlockIdx() == 0) {
        hccl_.Wait(allToAllHandleId_[allToAllWaitIdx_]);
    }
    SyncAll();
    uint32_t padM = tilePadM;
    uint32_t lastN = tilDat_->tilematmulTiling.matmulTiling.N;
    if (mc2Cfg.tailMatM != 0U) { // 有尾块时最后一块为尾块
        padM = tailPadM;
        lastN = tilDat_->tailmatmulTiling.matmulTiling.N;
    }
    MatmulAllReduceReduceSumInt8<yType>(tempBuffGM_, commQScale1GM_, commQScale2GM_, padM, lastN, tPipe_,
                                        hccl_);
    SyncAll();
    if (GetBlockIdx() == 0) {
        hccl_.Commit(allGatherHandleId_[allToAllWaitIdx_]);
        allToAllWaitIdx_++;
    }

    // wait所有allGather任务+dequant
    const uint64_t outGmTileOffset =
        static_cast<uint64_t>(tilDat_->tilematmulTiling.matmulTiling.M) * \
        static_cast<uint64_t>(tilDat_->tilematmulTiling.matmulTiling.N) * sizeof(yType);
    const uint64_t outGmTailOffset =
        static_cast<uint64_t>(tilDat_->tailmatmulTiling.matmulTiling.M) * \
        static_cast<uint64_t>(tilDat_->tailmatmulTiling.matmulTiling.N) * sizeof(yType);
    for (uint32_t idx = 0; idx < mc2Cfg.tileCnt + mc2Cfg.tailCnt; i++) {
        if (block_idx == 0) {
            hccl_.Wait(allGatherHandleId_[i]);
        }
        SyncAll();
        if (i < mc2Cfg.tileCnt) { // dequant只传有效大小
            MatmulAllReduceDequantPerchannelCommInt8<yType>(tempBuffDequantGM_, commQScale2GM_, outGM_, tPipe_,
                                                            tilDat_->tilematmulTiling.matmulTiling.N,
                                                            tilDat_->tilematmulTiling.matmulTiling.M);
            SyncAll();
            tempBuffDequantGM_ += tilePadDataCnt * sizeof(int8_t); // dequant的输入偏移pad后的大小
            outGM_ += outGmTileOffset;                             // 输出偏移原始大小
        } else {
            MatmulAllReduceDequantPerchannelCommInt8<yType>(tempBuffDequantGM_, commQScale2GM_, outGM_, tPipe_,
                                                            tilDat_->tailmatmulTiling.matmulTiling.N,
                                                            tilDat_->tailmatmulTiling.matmulTiling.M);
            SyncAll();
            tempBuffDequantGM_ += tailPadDataCnt * sizeof(int8_t);
            outGM_ += outGmTailOffset;
        }
    }
    if (GetBlockIdx() == 0) {
        hccl_.Finalize();
    }
}
} // namespace MatmulAllReduceImpl
#endif // MATMUL_ALL_REDUCE_QUANT_PERTOKEN_COMM_INT8_H