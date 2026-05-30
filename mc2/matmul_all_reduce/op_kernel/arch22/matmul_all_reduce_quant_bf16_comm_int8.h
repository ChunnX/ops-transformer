/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * \file matmul_all_reduce_quant_bf16_comm_int8.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_QUANT_BF16_COMM_INT8_H
#define MATMUL_ALL_REDUCE_QUANT_BF16_COMM_INT8_H

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
#include "../../3rd/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_bf16.h"
#include "matmul_all_reduce_add_x3.h"
#include "matmul_all_reduce_quant_perchannel.h"
#include "matmul_all_reduce_quant_reduce_sum.h"
#include "matmul_all_reduce_dequant_perchannel.h"

namespace MatmulAllReduceImpl {

constexpr uint32_t MAX_HANDLE_NUM = 16U;
constexpr uint32_t NUM_TWO = 2U;

using namespace AscendC;
using namespace MatmulAllReduceReduceSumImpl;
using namespace MatmulAllReduceQuantPerchannelImpl;
using namespace MatmulAllReduceDequantPerchannelImpl;

template <typename xType, typename wType, int fFormat, int wFormat, typename scaleType, typename yType,
          typename commType, bool aTrans, bool bTrans>
class MatmulAllReduceQuantBF16CommInt8 {
public:
    __aicore__ inline MatmulAllReduceQuantBF16CommInt8()
    {
    }
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR addGM, GM_ADDR dequantScaleGM,
                                GM_ADDR commQuantScale1GM, GM_ADDR commQuantScale2GM, GM_ADDR cGM, GM_ADDR workspaceGM,
                                Mc2Tiling::QuantMatmulAllReduceTilingData *tilDat, TPipe *tPipe);
    __aicore__ inline void
    Process(BmmDequantBf16<xType, wType, fFormat, wFormat, yType, yType, aTrans, bTrans, true> &opTile,
            BmmDequantBf16<xType, wType, fFormat, wFormat, yType, yType, aTrans, bTrans, true> &opTail);

private:
    __aicore__ inline void
    InnerProcess(BmmDequantBf16<xType, wType, fFormat, wFormat, yType, yType, aTrans, bTrans, true> &op,
                 uint32_t tileCnt, Mc2QuantBatchMatmulV3TilingData *mmTiling, uint32_t isAdd, uint32_t needUbBuffer,
                 uint32_t curPadM, bool isTailFlag);
    __aicore__ inline void PrepareInit();
    __aicore__ inline uint32_t SendCountCheck(uint32_t prepareIndex);

    Mc2Tiling::QuantMatmulAllReduceTilingData *tilDat_;
    TPipe *tPipe_;
    GM_ADDR aGM_;
    GM_ADDR bGM_;
    GM_ADDR biasGM_;
    GM_ADDR addGM_;
    GM_ADDR dequantScaleGM_;
    GM_ADDR commQuantScale1GM_;
    GM_ADDR commQuantScale2GM_;
    GM_ADDR cGM_;
    GM_ADDR workspaceGM_;
    GM_ADDR tempBuffWinOrGM_;
    GM_ADDR tempBuffGM_;
    GM_ADDR tempBuffDequantGM_;
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    GM_ADDR outGM_;
    uint32_t tilePadMatM_ = 0U;
    uint32_t tailPadMatM_ = 0U;
    uint32_t tilePadCnt_ = 0U;
    uint32_t tailPadCnt_ = 0U;

    // 仅在0核上使用
    AscendC::HcclHandle allToAllHandleId_[MAX_HANDLE_NUM] = {0};
    AscendC::HcclHandle allGatherHandleId_[MAX_HANDLE_NUM] = {0};
    GM_ADDR all2AllSendGM_[MAX_HANDLE_NUM] = {0};
    GM_ADDR aGatherSendGM_[MAX_HANDLE_NUM] = {0};
    GM_ADDR allRecvOutGM_[MAX_HANDLE_NUM] = {0};
    int all2AllCommitIdx_ = 0;
    int all2AllWaitIdx_ = 0;
    int allGatherCommitIdx_ = 0;
    int allGatherWaitIdx_ = 0;
    // 所有核
    bool isSendTileFlag_ = false;
};

template <typename xType, typename wType, int fFormat, int wFormat, typename scaleType, typename yType,
          typename commType, bool aTrans, bool bTrans>
__aicore__ inline void
MatmulAllReduceQuantBF16CommInt8<xType, wType, fFormat, wFormat, scaleType, yType, commType, aTrans, bTrans>::
    InnerProcess(BmmDequantBf16<xType, wType, fFormat, wFormat, yType, yType, aTrans, bTrans, true> &op,
                 uint32_t tileCnt, Mc2QuantBatchMatmulV3TilingData *mmTiling, uint32_t isAdd, uint32_t needUbBuffer,
                 uint32_t curPadM, bool isTailFlag)
{
    const uint64_t aOffset = CalcShapeOffset(sizeof(xType), mmTiling->matmulTiling.M, mmTiling->matmulTiling.Ka);
    const uint64_t cOffset = CalcShapeOffset(sizeof(yType), mmTiling->matmulTiling.M, mmTiling->matmulTiling.N);
    const uint64_t tempOffset = CalcShapeOffset(sizeof(int8_t), curPadM, mmTiling->matmulTiling.N);

    for (uint32_t i = 0U; i < tileCnt; ++i) {
        tPipe_->Reset();
        op.Init(aGM_, bGM_, biasGM_, dequantScaleGM_, cGM_, workspaceGM_, mmTiling, tPipe_);
        op.Process();
        if (isAdd) {
            SyncAll();
            Matmul_All_Reduce_Add_X3<yType>(cGM_, addGM_, cOffset / sizeof(yType), tilDat_->param.addX3UbCnt,
                                            tPipe_);
            addGM_ += cOffset;
        }
        SyncAll();
        // bf16->int8 存在 tempBuffWinOrGM_
        MatmulAllReduceQuantPerchannelCommInt8<yType>(cGM_, commQuantScale1GM_, tempBuffWinOrGM_, tPipe_,
                                                      mmTiling->matmulTiling.N, mmTiling->matmulTiling.M);
        SyncAll();
        if (GetBlockIdx() == 0) {
            hccl_.Commit(allToAllHandleId_[all2AllCommitIdx_]);
            all2AllCommitIdx_++;
        }
        if (isSendTileFlag_) { // 需要同步前一块
            if (GetBlockIdx() == 0) {
                hccl_.Wait(allToAllHandleId_[all2AllWaitIdx_]);
            }
            SyncAll();
            if (isTailFlag && (i == 0U)) {
                MatmulAllReduceReduceSumInt8<yType>(tempBuffGM_, commQuantScale1GM_, commQuantScale2GM_, tilePadMatM_,
                                                    mmTiling->matmulTiling.N, tPipe_, hccl_);
                tempBuffGM_ += tilePadMatM_ * mmTiling->matmulTiling.N;
            } else {
                MatmulAllReduceReduceSumInt8<yType>(tempBuffGM_, commQuantScale1GM_, commQuantScale2GM_, curPadM,
                                                    mmTiling->matmulTiling.N, tPipe_, hccl_);
                tempBuffGM_ += curPadM * mmTiling->matmulTiling.N;
            }
            SyncAll();
            if (GetBlockIdx() == 0) {
                hccl_.Commit(allGatherHandleId_[all2AllWaitIdx_]);
                all2AllWaitIdx_++;
            }
        }
        isSendTileFlag_ = true;
        aGM_ += aOffset;
        cGM_ += cOffset;                // 偏原始大小
        tempBuffWinOrGM_ += tempOffset; // 偏移 padM*N 大小
    }
}

template <typename xType, typename wType, int fFormat, int wFormat, typename scaleType, typename yType,
          typename commType, bool aTrans, bool bTrans>
__aicore__ inline void
MatmulAllReduceQuantBF16CommInt8<xType, wType, fFormat, wFormat, scaleType, yType, commType, aTrans, bTrans>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR addGM, GM_ADDR dequantScaleGM, GM_ADDR commQuantScale1GM,
    GM_ADDR commQuantScale2GM, GM_ADDR cGM, GM_ADDR workspaceGM, Mc2Tiling::QuantMatmulAllReduceTilingData *tilDat,
    TPipe *tPipe)
{
    __gm__ HcclCombinOpParam *context = (__gm__ HcclCombinOpParam *)(GetHcclContext<0>());
    OOMInit(context);
    hccl_.Init(GetHcclContext<0>());
    tilDat_ = tilDat;
    tPipe_ = tPipe;
    aGM_ = aGM;
    bGM_ = bGM;
    biasGM_ = biasGM;
    addGM_ = addGM;
    dequantScaleGM_ = dequantScaleGM;
    commQuantScale1GM_ = commQuantScale1GM;
    commQuantScale2GM_ = commQuantScale2GM;
    workspaceGM_ = workspaceGM;
    cGM_ = cGM;
    outGM_ = cGM;
    tempBuffGM_ = workspaceGM_ + tilDat_->param.commWorkSpaceSize;
    tempBuffDequantGM_ = workspaceGM_ + tilDat_->param.commWorkSpaceSize;

    // tiling 侧控制不走 winTowin，赋值 tempBuffWinOrGM_
    if (tilDat->msg.useBufferType == MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_WINDOW_IN) {
        tempBuffWinOrGM_ = hccl_.GetWindowsInAddr(hccl_.GetRankId());
    } else {
        tempBuffWinOrGM_ = tempBuffGM_;
    }
}

template <typename xType, typename wType, int fFormat, int wFormat, typename scaleType, typename yType,
          typename commType, bool aTrans, bool bTrans>
__aicore__ inline void MatmulAllReduceQuantBF16CommInt8<xType, wType, fFormat, wFormat, scaleType, yType, commType,
                                                        aTrans, bTrans>::PrepareInit()
{
    auto &&mc2Cfg = tilDat_->param;
    uint32_t rankCnt = mc2Cfg.rankDim;
    uint32_t tileMatM = tilDat_->tilematmulTiling.matmulTiling.M; // 头块 pad 大小更新
    tilePadMatM_ = tileMatM;
    if ((tileMatM % rankCnt) != 0) { // 按照 M 来 pad
        tilePadMatM_ += rankCnt - (tileMatM % rankCnt);
    }
    uint32_t tailMatM = tilDat_->tailmatmulTiling.matmulTiling.M;
    tailPadMatM_ = tailMatM;
    if ((tailMatM % rankCnt) != 0) {
        tailPadMatM_ += rankCnt - (tailMatM % rankCnt);
    }

    tilePadCnt_ = tilePadMatM_ * tilDat_->tilematmulTiling.matmulTiling.N;
    tailPadCnt_ = tailPadMatM_ * tilDat_->tailmatmulTiling.matmulTiling.N;
    const uint64_t tempBufOffsetTilePad = tilePadCnt_ * sizeof(int8_t); // 偏移计算
    const uint64_t tempBufOffsetTailPad = tailPadCnt_ * sizeof(int8_t);
    const uint64_t tempBufOffsetTilePadSingle = tempBufOffsetTilePad / rankCnt;
    const uint64_t tempBufOffsetTailPadSingle = tempBufOffsetTailPad / rankCnt;

    for (uint32_t i = 0U; i < mc2Cfg.tileCnt; ++i) { // 头块偏移
        const uint64_t indexOffsetTile = i * tempBufOffsetTilePad;
        const uint64_t indexGatherOffsetTile = tempBufOffsetTilePadSingle * hccl_.GetRankId();
        all2AllSendGM_[i] = tempBuffWinOrGM_ + indexOffsetTile;
        allRecvOutGM_[i] = tempBuffGM_ + indexOffsetTile;
        aGatherSendGM_[i] = tempBuffGM_ + indexOffsetTile + indexGatherOffsetTile;
    }
    for (uint32_t i = 0U; i < mc2Cfg.tailCnt; ++i) { // 尾块偏移
        const uint64_t indexOffsetTail = mc2Cfg.tileCnt * tempBufOffsetTilePad + i * tempBufOffsetTailPad;
        const uint64_t indexGatherOffsetTail = tempBufOffsetTailPadSingle * hccl_.GetRankId();
        all2AllSendGM_[mc2Cfg.tileCnt + i] = tempBuffWinOrGM_ + indexOffsetTail;
        allRecvOutGM_[mc2Cfg.tileCnt + i] = tempBuffGM_ + indexOffsetTail;
        aGatherSendGM_[mc2Cfg.tileCnt + i] = tempBuffGM_ + indexOffsetTail + indexGatherOffsetTail;
    }

    if (GetBlockIdx() == 0) {
        uint32_t nowAll2AllIdx = 0U;
        uint32_t nowAllGatherIdx = 0U;
        uint32_t numN = (mc2Cfg.tileCnt + mc2Cfg.tailCnt) / NUM_TWO;
        uint32_t numReN = (mc2Cfg.tileCnt + mc2Cfg.tailCnt) % NUM_TWO;
        for (uint32_t i = 0U; i < numN; ++i) { // 按总核数下发
            allToAllHandleId_[nowAll2AllIdx] =
                hccl_.AlltoAll<false>(all2AllSendGM_[nowAll2AllIdx], allRecvOutGM_[nowAll2AllIdx],
                                      SendCountCheck(nowAll2AllIdx), AscendC::HCCL_DATA_TYPE_INT8, 0);
            nowAll2AllIdx++;
            allToAllHandleId_[nowAll2AllIdx] =
                hccl_.AlltoAll<false>(all2AllSendGM_[nowAll2AllIdx], allRecvOutGM_[nowAll2AllIdx],
                                      SendCountCheck(nowAll2AllIdx), AscendC::HCCL_DATA_TYPE_INT8, 0);
            nowAll2AllIdx++;

            allGatherHandleId_[nowAllGatherIdx] =
                hccl_.AllGather<false>(aGatherSendGM_[nowAllGatherIdx], allRecvOutGM_[nowAllGatherIdx],
                                       SendCountCheck(nowAllGatherIdx), AscendC::HCCL_DATA_TYPE_INT8, 0);
            nowAllGatherIdx++;
            allGatherHandleId_[nowAllGatherIdx] =
                hccl_.AllGather<false>(aGatherSendGM_[nowAllGatherIdx], allRecvOutGM_[nowAllGatherIdx],
                                       SendCountCheck(nowAllGatherIdx), AscendC::HCCL_DATA_TYPE_INT8, 0);
            nowAllGatherIdx++;
        }

        if (numReN != 0U) { // 余数下发
            allToAllHandleId_[nowAll2AllIdx] =
                hccl_.AlltoAll<false>(all2AllSendGM_[nowAll2AllIdx], allRecvOutGM_[nowAll2AllIdx],
                                      SendCountCheck(nowAll2AllIdx), AscendC::HCCL_DATA_TYPE_INT8, 0);
            allGatherHandleId_[nowAllGatherIdx] =
                hccl_.AllGather<false>(aGatherSendGM_[nowAllGatherIdx], allRecvOutGM_[nowAllGatherIdx],
                                       SendCountCheck(nowAllGatherIdx), AscendC::HCCL_DATA_TYPE_INT8, 0);
        }
    }
}

template <typename xType, typename wType, int fFormat, int wFormat, typename scaleType, typename yType,
          typename commType, bool aTrans, bool bTrans>
__aicore__ inline uint32_t MatmulAllReduceQuantBF16CommInt8<xType, wType, fFormat, wFormat, scaleType, yType, commType,
                                                            aTrans, bTrans>::SendCountCheck(uint32_t prepareIndex)
{
    uint32_t txCnt = tilePadCnt_ / tilDat_->param.rankDim;
    if (prepareIndex >= tilDat_->param.tileCnt) {
        txCnt = tailPadCnt_ / tilDat_->param.rankDim;
    }
    return txCnt;
}

template <typename xType, typename wType, int fFormat, int wFormat, typename scaleType, typename yType,
          typename commType, bool aTrans, bool bTrans>
__aicore__ inline void
MatmulAllReduceQuantBF16CommInt8<xType, wType, fFormat, wFormat, scaleType, yType, commType, aTrans, bTrans>::Process(
    BmmDequantBf16<xType, wType, fFormat, wFormat, yType, yType, aTrans, bTrans, true> &opTile,
    BmmDequantBf16<xType, wType, fFormat, wFormat, yType, yType, aTrans, bTrans, true> &opTail)
{
    auto &&mc2Cfg = tilDat_->param;

    // all2all allgather Prepare + InnerProcess
    PrepareInit();
    InnerProcess(opTile, mc2Cfg.tileCnt, &tilDat_->tilematmulTiling, mc2Cfg.isAdd, mc2Cfg.needUbBuffer,
                 tilePadMatM_, false);
    if (mc2Cfg.tailMatM != 0U) {
        InnerProcess(opTail, mc2Cfg.tailCnt, &tilDat_->tailmatmulTiling, mc2Cfg.isAdd, mc2Cfg.needUbBuffer,
                     tailPadMatM_, true);
    }

    // 最后一次的 all2all 任务没有 wait+reudseSum，以及最后一次的 allgather 任务没有下发
    if (GetBlockIdx() == 0) {
        hccl_.Wait(allToAllHandleId_[all2AllWaitIdx_]);
    }
    SyncAll();
    uint32_t padM = tilePadMatM_;
    uint32_t lastN = tilDat_->tilematmulTiling.matmulTiling.N;
    if (mc2Cfg.tailMatM != 0U) {
        padM = tailPadMatM_;
        lastN = tilDat_->tailmatmulTiling.matmulTiling.N;
    }
    MatmulAllReduceReduceSumInt8<yType>(tempBuffGM_, commQuantScale1GM_, commQuantScale2GM_, padM, lastN, tPipe_,
                                        hccl_);
    SyncAll();
    if (GetBlockIdx() == 0) {
        hccl_.Commit(allGatherHandleId_[all2AllWaitIdx_]);
        all2AllWaitIdx_++;
    }

    const uint64_t outGmTileOffset =
        static_cast<uint64_t>(tilDat_->tilematmulTiling.matmulTiling.M) * \
        static_cast<uint64_t>(tilDat_->tilematmulTiling.matmulTiling.N) * sizeof(yType);
    const uint64_t outGmTailOffset =
        static_cast<uint64_t>(tilDat_->tailmatmulTiling.matmulTiling.M) * \
        static_cast<uint64_t>(tilDat_->tailmatmulTiling.matmulTiling.N) * sizeof(yType);
    for (uint32_t i = 0U; i < (mc2Cfg.tileCnt + mc2Cfg.tailCnt); ++i) { // 尾块偏移
        if (GetBlockIdx() == 0) {
            hccl_.Wait(allGatherHandleId_[i]);
        }
        SyncAll();
        if (i < mc2Cfg.tileCnt) {
            MatmulAllReduceDequantPerchannelCommInt8<yType>(tempBuffDequantGM_, commQuantScale2GM_, outGM_, tPipe_,
                                                            tilDat_->tilematmulTiling.matmulTiling.N,
                                                            tilDat_->tilematmulTiling.matmulTiling.M);
            tempBuffDequantGM_ += tilePadCnt_ * sizeof(int8_t);
            outGM_ += outGmTileOffset;
            SyncAll();
        } else {
            MatmulAllReduceDequantPerchannelCommInt8<yType>(tempBuffDequantGM_, commQuantScale2GM_, outGM_, tPipe_,
                                                            tilDat_->tailmatmulTiling.matmulTiling.N,
                                                            tilDat_->tailmatmulTiling.matmulTiling.M);
            tempBuffDequantGM_ += tailPadCnt_ * sizeof(int8_t);
            outGM_ += outGmTailOffset;
            SyncAll();
        }
    }
    if (GetBlockIdx() == 0) {
        hccl_.Finalize();
    }
}
} // namespace MatmulAllReduceImpl
#endif // MATMUL_ALL_REDUCE_QUANT_BF16_COMM_INT8_H