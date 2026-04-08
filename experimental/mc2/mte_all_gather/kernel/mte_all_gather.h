/*!
 * \file mte_all_gather.h
 * \brief mte_all_gather 通过1步allgather的kernel代码逻辑
 *        完成AllGather：将各卡Win区中的X数据按rank顺序拼接到output GM
 */

#ifndef MTE_ALL_GATHER_H
#define MTE_ALL_GATHER_H

#include "basic_api/kernel_basic_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mte_all_gather_tiling_data.h"
#include "utils.h"
#include "mte_comm.h"

namespace MteAllGatherImpl {

using namespace MTECommImpl;
using namespace AscendC;

constexpr static uint32_t X_PRE_BLOCK_NUM = 1024U;

template<typename XType>
class MteAllGather {
public:
    __aicore__ inline MteAllGather() {};
    __aicore__ inline void InitWithMc2Context(GM_ADDR x, GM_ADDR output,
                                GM_ADDR mc2Context, GM_ADDR tilingGM, TPipe *pipe);
    __aicore__ inline void Process();
private:
    __aicore__ inline void ExecuteAllGather();

    uint64_t xSize_{0};
    uint64_t xNums_{0};
    uint64_t tailXNums_{0};
    uint32_t totalBlockNums_{0};

    __gm__ MteAllGatherTilingData* tilingDataOut;
    MTECommunication<XType> mteComm_;

    GlobalTensor<XType> remoteWinXTensor_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xInQueue_;
};

template <typename XType>
__aicore__ inline void MteAllGather<XType>::InitWithMc2Context(GM_ADDR x, GM_ADDR output,
    GM_ADDR mc2Context, GM_ADDR tilingGM, TPipe *tPipe)
{
    mteComm_.InitHcclContextByAddr(mc2Context);
    tilingDataOut = (__gm__ MteAllGatherTilingData*)tilingGM;

    auto&& tiliingDatainfo = tilingDataOut->mteAllGatherTilingInfo;
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

    mteComm_.InitGMTensor(x, output, xNums_, xSize_);
}

template <typename XType>
__aicore__ inline void MteAllGather<XType>::ExecuteAllGather()
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
            DataCopyPad(xTmpTensor, remoteWinXTensor_[curXOffset], 
                {1, static_cast<uint32_t>(copyBlockNum * sizeof(XType)), 0, 0, 0}, 
                {false, 0, 0, 0});
            xInQueue_.EnQue(xTmpTensor);
            xTmpTensor = xInQueue_.DeQue<XType>();
            DataCopyPad(mteComm_.outputTensor_[outOffset], xTmpTensor, 
                {1, static_cast<uint32_t>(copyBlockNum * sizeof(XType)), 0, 0, 0});
            xInQueue_.FreeTensor(xTmpTensor);
        }
    }
}

template <typename XType>
__aicore__ inline void MteAllGather<XType>::Process()
{
    if ASCEND_IS_AIC {
        return;
    }

    mteComm_.CopyDataToWin();
    mteComm_.WriteStatusToWin();
    ExecuteAllGather();
}
} // MteAllGatherImpl
#endif  // MTE_ALL_GATHER_H
