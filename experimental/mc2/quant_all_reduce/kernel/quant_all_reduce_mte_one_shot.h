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
 * \file quant_all_gather_mte_one_shot.h
 * \brief quant_all_gather mte_one_shot方式通信，通过1步allgather的kernel代码逻辑
 *        仅完成AllGather：将各卡Win区中的X数据按rank顺序拼接到output GM
 */

#ifndef QUANT_ALL_REDUCE_MTE_ONE_SHOT_H
#define QUANT_ALL_REDUCE_MTE_ONE_SHOT_H

#include "basic_api/kernel_basic_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "quant_all_reduce_tiling_data.h"
#include "utils.h"
#include "mte_comm.h"

namespace QuantAllGatherImpl {

using namespace QuantMTECommImpl;
using namespace AscendC;

constexpr static uint32_t X_PRE_BLOCK_NUM = 1024U;

template<TemplateTypeClass>
class QuantAllGatherMteOneShot {
public:
    __aicore__ inline QuantAllGatherMteOneShot() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR output,
                                TPipe *pipe, const QuantAllReduceTilingData *tilingData);
    __aicore__ inline void InitWithMc2Context(GM_ADDR x, GM_ADDR output,
                                GM_ADDR mc2Context, GM_ADDR tilingGM, TPipe *pipe);
    __aicore__ inline void Process();
private:
    __aicore__ inline void ExecuteAllGather();

    uint64_t xSize_{0};
    uint64_t xNums_{0};
    uint64_t totalWinSize_{0};
    uint64_t tailXNums_{0};
    uint32_t totalBlockNums_{0};

    __gm__ QuantAllReduceTilingData* tilingDataOut;
    MTECommunication<TemplateType> mteComm_;

    GlobalTensor<XType> remoteWinXTensor_;

    TQue<QuePosition::VECIN, 1> xInQueue_;
};

template <TemplateTypeClass>
__aicore__ inline void QuantAllGatherMteOneShot<TemplateType>::Init(GM_ADDR x, GM_ADDR output,
    TPipe *tPipe, const QuantAllReduceTilingData *tilingData)
{
    mteComm_.InitHcclContext();

    auto&& tiliingDatainfo = tilingData->quantAllReduceTilingInfo;
    totalWinSize_ = tiliingDatainfo.totalWinSize;
    xNums_ = tiliingDatainfo.bs * tiliingDatainfo.hiddenSize;
    xSize_ = xNums_ * sizeof(XType);
    tailXNums_ = BlockAlignMod(xNums_, X_PRE_BLOCK_NUM);
    totalBlockNums_ = CeilDiv(xSize_, X_BLOCK_BYTES);
    mteComm_.round_ = totalBlockNums_ / tiliingDatainfo.aivNum;
    mteComm_.tailBlockNums_ = totalBlockNums_ % tiliingDatainfo.aivNum;
    mteComm_.ComputeTailAivId(tiliingDatainfo.aivNum);
    tPipe->Reset();
    tPipe->InitBuffer(xInQueue_, BUFFER_NUM, X_BLOCK_BYTES);

    mteComm_.SetBlockSize(X_PRE_BLOCK_NUM, tiliingDatainfo.aivNum, tailXNums_);

    mteComm_.InitParams();

    mteComm_.InitBuffer(tPipe);

    mteComm_.InitGMTensor(x, output, xNums_, xSize_, totalWinSize_);
}

template <TemplateTypeClass>
__aicore__ inline void QuantAllGatherMteOneShot<TemplateType>::InitWithMc2Context(GM_ADDR x, GM_ADDR output,
    GM_ADDR mc2Context, GM_ADDR tilingGM, TPipe *tPipe)
{
    mteComm_.InitHcclContextByAddr(mc2Context);
    tilingDataOut = (__gm__ QuantAllReduceTilingData*)tilingGM;

    auto&& tiliingDatainfo = tilingDataOut->quantAllReduceTilingInfo;
    totalWinSize_ = tiliingDatainfo.totalWinSize;
    xNums_ = tiliingDatainfo.bs * tiliingDatainfo.hiddenSize;
    xSize_ = xNums_ * sizeof(XType);
    tailXNums_ = BlockAlignMod(xNums_, X_PRE_BLOCK_NUM);
    totalBlockNums_ = CeilDiv(xSize_, X_BLOCK_BYTES);
    mteComm_.round_ = totalBlockNums_ / tiliingDatainfo.aivNum;
    mteComm_.tailBlockNums_ = totalBlockNums_ % tiliingDatainfo.aivNum;
    mteComm_.ComputeTailAivId(tiliingDatainfo.aivNum);
    tPipe->Reset();
    tPipe->InitBuffer(xInQueue_, BUFFER_NUM, X_BLOCK_BYTES);

    mteComm_.SetBlockSize(X_PRE_BLOCK_NUM, tiliingDatainfo.aivNum, tailXNums_);

    mteComm_.InitParams();

    mteComm_.InitBuffer(tPipe);

    mteComm_.InitGMTensor(x, output, xNums_, xSize_, totalWinSize_);
}

template <TemplateTypeClass>
__aicore__ inline void QuantAllGatherMteOneShot<TemplateType>::ExecuteAllGather()
{
    mteComm_.ReadStatus();

    for (uint64_t curBlock = 0; curBlock < mteComm_.assignedBlockNums_; ++curBlock) {
        uint64_t curXOffset = mteComm_.xOffset_ + curBlock * X_PRE_BLOCK_NUM;

        uint32_t copyBlockNum = X_PRE_BLOCK_NUM;
        if ((mteComm_.aivId_ == mteComm_.lastAivId_) && (curBlock == mteComm_.assignedBlockNums_ - 1)) {
            copyBlockNum = tailXNums_;
        }

        for (uint32_t remoteRankId = 0; remoteRankId < mteComm_.hcclContext_->rankDim; ++remoteRankId) {
            GM_ADDR remoteDataSpaceGm = mteComm_.GetWinDataAddrGm(remoteRankId, mteComm_.winBufferFlags_);
            remoteWinXTensor_.SetGlobalBuffer((__gm__ XType*)remoteDataSpaceGm);

            uint64_t outOffset = remoteRankId * xNums_ + curXOffset;

            LocalTensor<XType> xTmpTensor = xInQueue_.AllocTensor<XType>();
            DataCopy(xTmpTensor, remoteWinXTensor_[curXOffset], copyBlockNum);
            xInQueue_.EnQue(xTmpTensor);
            xTmpTensor = xInQueue_.DeQue<XType>();
            DataCopy(mteComm_.outputTensor_[outOffset], xTmpTensor, copyBlockNum);
            xInQueue_.FreeTensor(xTmpTensor);
        }
    }
}

template <TemplateTypeClass>
__aicore__ inline void QuantAllGatherMteOneShot<TemplateType>::Process()
{
    if ASCEND_IS_AIC {
        return;
    }

    mteComm_.CopyDataToWin();
    mteComm_.WriteStatusToWin();
    ExecuteAllGather();
}
} // QuantAllGatherImpl
#endif  // QUANT_ALL_REDUCE_MTE_ONE_SHOT_H
