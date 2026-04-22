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
 * \file quant_all_reduce_mte_one_shot.h
 * \brief quant_all_reduce mte_one_shot方式通信，通过1步allgather的kernel代码逻辑
 */

#ifndef QUANT_ALL_REDUCE_MTE_ONE_SHOT_H
#define QUANT_ALL_REDUCE_MTE_ONE_SHOT_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "quant_all_reduce_tiling_data.h"
#include "../quant_reduce_scatter/utils.h"
#include "../quant_reduce_scatter/mte_comm.h"
#include "../quant_reduce_scatter/vec_comp.h"

namespace QuantAllReduceImpl {

using namespace QuantMTECommImpl;
using namespace VectorComputeImpl;
using namespace AscendC;

// 之后可修改成从tiling侧获取数据切块大小
constexpr static uint32_t X_PRE_BLOCK_NUM = 1024U;  // 当前一次搬运一个x数据块，x dtype为 8bit 时对应 1024个x数据. 对于fp4需要另外算

template<TemplateTypeClass>
class QuantAllReduceMteOneShot {
public:
    __aicore__ inline QuantAllReduceMteOneShot() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scales, GM_ADDR output,
                                TPipe *pipe, const QuantAllReduceTilingData *tilingData);
    __aicore__ inline void Process();
private:
    __aicore__ inline void ClearSumTensor();
    __aicore__ inline void ReadDataBlockReduceSum(uint64_t curXOffset, uint64_t curScaleOffset);
    __aicore__ inline void ExecuteAllReduce();

    uint64_t xSize_{0};
    uint64_t xNums_{0};
    uint64_t totalWinSize_{0};
    uint64_t tailXNums_{0};
    uint32_t totalBlockNums_{0};

    MTECommunication<TemplateType> mteComm_; // MTE 通信相关实现
    VectorCompute<TemplateType> vecComp_; // vector 计算相关实现

    GlobalTensor<XType> remoteWinXTensor_;
    GlobalTensor<ScalesType> remoteWinScaleTensor_;
    LocalTensor<float> sumTensor_;

    TQue<QuePosition::VECIN, 1> xInQueue_, scaleInQue; // 用于读数据和反量化求和的通算并行
    TBuf<> sumBuf_; // 用于Reduce_sum 求和
};

template <TemplateTypeClass>
__aicore__ inline void QuantAllReduceMteOneShot<TemplateType>::Init(GM_ADDR x, GM_ADDR scales,
    GM_ADDR output, TPipe *tPipe, const QuantAllReduceTilingData *tilingData)
{
    // 初始化HcclContext
    mteComm_.InitHcclContext();

    // 动态计算数据块大小
    mteComm_.ComputeBlockParams();

    /* quant_all_reduce自己的数据 */
    auto&& tiliingDatainfo = tilingData->quantAllReduceTilingInfo;
    totalWinSize_ = tiliingDatainfo.totalWinSize;
    xNums_ = tiliingDatainfo.bs * tiliingDatainfo.hiddenSize; // 总的x数据个数， bs * h
    xSize_ = xNums_ * sizeof(XType); // 总的x数据量，B
    
    // AIV 分组：每个 rank 分配的核数
    mteComm_.SetSendCoreNumPerRank(tiliingDatainfo.aivNum / mteComm_.hcclContext_->rankDim);
    
    // AllReduce 输出按 rankDim 分区，每核负责 xNums_ / rankDim 的数据
    uint64_t perRankOutSize = xNums_ / mteComm_.hcclContext_->rankDim;
    tailXNums_ = BlockAlignMod(perRankOutSize, mteComm_.xNumPerBlock_); // 每分区尾块大小
    uint32_t xBlockBytes = mteComm_.xNumPerBlock_ * sizeof(XType);
    
    // 按 sendCoreNumPerRank 切分数据块（每个核处理 perRankOutSize 的数据）
    uint64_t perRankBlockNums_ = CeilDiv(perRankOutSize * sizeof(XType), xBlockBytes);
    mteComm_.round_ = perRankBlockNums_ / mteComm_.sendCoreNumPerRank_;
    mteComm_.tailBlockNums_ = perRankBlockNums_ % mteComm_.sendCoreNumPerRank_;
    mteComm_.ComputeTailAivId(tiliingDatainfo.aivNum);
    
    tPipe->Reset();
    tPipe->InitBuffer(xInQueue_, BUFFER_NUM, xBlockBytes);
    tPipe->InitBuffer(scaleInQue, BUFFER_NUM, UB_ALIGN_BYTES);
    tPipe->InitBuffer(sumBuf_, mteComm_.xNumPerBlock_ * sizeof(float));
    sumTensor_ = sumBuf_.Get<float>();

    mteComm_.SetBlockSize(mteComm_.xNumPerBlock_, tiliingDatainfo.aivNum, tailXNums_);
    vecComp_.SetBlockSize(mteComm_.xNumPerBlock_);  

    // 公共MTE搬运参数计算（AIV 分组模式，传入每分区大小）
    mteComm_.InitParamsForAIVGroup(perRankOutSize);

    mteComm_.InitBuffer(tPipe);
    vecComp_.InitBuffer(tPipe);

    mteComm_.InitGMTensor(x, scales, output, xSize_, totalWinSize_);
}

template <TemplateTypeClass>
__aicore__ inline void QuantAllReduceMteOneShot<TemplateType>::ReadDataBlockReduceSum(uint64_t curXOffset,
    uint64_t curScaleOffset)
{
    /* 读取 x 从 Win -> UB */
    LocalTensor<XType> xTmpTensor = xInQueue_.AllocTensor<XType>();
    if constexpr (AscendC::IsSameType<XType, int8_t>::value) {
        DataCopyExtParams xCopyInParams{1, static_cast<uint32_t>(mteComm_.xNumPerBlock_ * sizeof(XType)), 0, 0, 0};
        DataCopyPadExtParams<XType> xPadParams{false, 0, 0, 0};
        DataCopyPad(xTmpTensor, remoteWinXTensor_[curXOffset], xCopyInParams, xPadParams);
    } else {
        DataCopy(xTmpTensor, remoteWinXTensor_[curXOffset], mteComm_.xNumPerBlock_);
    }
    xInQueue_.EnQue(xTmpTensor);
    xTmpTensor = xInQueue_.DeQue<XType>();

    /* 读取 scale 从 Win -> UB */
    LocalTensor<ScalesType> scaleTmpTensor = scaleInQue.AllocTensor<ScalesType>();
    if constexpr (AscendC::IsSameType<ScalesType, float>::value) {
        DataCopyExtParams scaleCopyInParams{1, static_cast<uint32_t>(mteComm_.scaleNumsPerBlcok_ * sizeof(ScalesType)), 0, 0, 0};
        DataCopyPadExtParams<ScalesType> scalePadParams{false, 0, 0, 0};
        DataCopyPad(scaleTmpTensor, remoteWinScaleTensor_[curScaleOffset], scaleCopyInParams, scalePadParams);
    } else {
        DataCopy(scaleTmpTensor, remoteWinScaleTensor_[curScaleOffset], mteComm_.scaleNumsPerBlcok_);
    }
    scaleInQue.EnQue(scaleTmpTensor);
    scaleTmpTensor = scaleInQue.DeQue<ScalesType>();

    /* 反量化计算与ReduceSum求和 */ 
    vecComp_.DequantReduceSum(xTmpTensor, scaleTmpTensor, sumTensor_); 
    xInQueue_.FreeTensor(xTmpTensor);
    scaleInQue.FreeTensor(scaleTmpTensor);
}

template <TemplateTypeClass>
__aicore__ inline void QuantAllReduceMteOneShot<TemplateType>::ClearSumTensor()
{
    Duplicate<float>(sumTensor_, (float)0.0, mteComm_.xNumPerBlock_); // sumTensor 清零，使用动态块大小
}

template <TemplateTypeClass>
__aicore__ inline void QuantAllReduceMteOneShot<TemplateType>::ExecuteAllReduce()
{   
    // 读状态位，软同步
    mteComm_.ReadStatus();
    
    // 边界检查：xOffset_ >= 每分区大小时跳过（该核无数据处理）
    uint64_t perRankOutSize = xNums_ / mteComm_.hcclContext_->rankDim;
    if (mteComm_.xOffset_ >= perRankOutSize) {
        PipeBarrier<PIPE_ALL>();
        return;
    }
    
    // 遍历需要搬运的数据块
    uint32_t lastCoreIndexForRank = mteComm_.sendCoreNumPerRank_ - 1;
    for (uint64_t curBlock = 0; curBlock < mteComm_.assignedBlockNums_; ++curBlock) {
        // 分区内偏移
        uint64_t curXOffset = mteComm_.xOffset_ + curBlock * mteComm_.xNumPerBlock_;
        uint64_t curScaleOffset = mteComm_.scaleOffset_ + curBlock * mteComm_.scaleNumsPerBlcok_;
        ClearSumTensor(); // sumTensor 清零
        
        // 遍历每张卡，读取其Win区的数据，采取错卡序读取，从自己卡上读起
        uint32_t startRankId = mteComm_.hcclContext_->rankId;
        for (uint32_t i = 0; i < mteComm_.hcclContext_->rankDim; ++i) {
            uint32_t remoteRankId = (startRankId + i) % mteComm_.hcclContext_->rankDim;

            // 获取对端Win区中数据区相关的地址
            GM_ADDR remoteDataSpaceGm = mteComm_.GetWinDataAddrGm(remoteRankId, mteComm_.winBufferFlags_);
            remoteWinXTensor_.SetGlobalBuffer((__gm__ XType*)remoteDataSpaceGm);
            remoteWinScaleTensor_.SetGlobalBuffer((__gm__ ScalesType*)(remoteDataSpaceGm + xSize_));

            // AIV分组：读取 Win 区时加上分区偏移
            // Win 区存储全量数据，读取位置 = 分区偏移 + 分区内偏移
            uint64_t winXOffset = mteComm_.partitionOffset_ + curXOffset;
            uint64_t winScaleOffset = mteComm_.partitionOffset_ / 128 + curScaleOffset;
            ReadDataBlockReduceSum(winXOffset, winScaleOffset);
        }

        // 将计算好的数据拷贝到输出tensor（AIV分组：输出位置 = 分区偏移 + 分区内偏移）
        uint32_t copyBlockNum = mteComm_.xNumPerBlock_;
        if ((mteComm_.coreInnerIndex_ == lastCoreIndexForRank) && (curBlock == mteComm_.assignedBlockNums_ - 1)) {
            copyBlockNum = tailXNums_;
        }
        uint64_t curOutOffset = mteComm_.partitionOffset_ + curXOffset;
        mteComm_.CopyResultToOutput(curOutOffset, sumTensor_, copyBlockNum);
    }
}

template <TemplateTypeClass>
__aicore__ inline void QuantAllReduceMteOneShot<TemplateType>::Process()
{
    // 纯AIV过程
    if ASCEND_IS_AIC {
        return;
    }

    // 边界检查：xOffset_ >= 每分区大小时跳过整个流程（该核无数据处理）
    uint64_t perRankOutSize = xNums_ / mteComm_.hcclContext_->rankDim;
    if (mteComm_.xOffset_ >= perRankOutSize) {
        return;
    }

    // 一次性拷贝完所有数据到本地卡win区
    mteComm_.CopyDataToWin();
    // 写入状态到状态区
    mteComm_.WriteStatusToWin();
    // 执行AllReduce过程：等待状态区同步，读取数据并进行反量化ReduceSum
    ExecuteAllReduce();
}
} // QuantAllReduceImpl
#endif  // QUANT_ALL_REDUCE_MTE_ONE_SHOT_H