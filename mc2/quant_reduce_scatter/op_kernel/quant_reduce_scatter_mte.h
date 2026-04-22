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
 * \file quant_reduce_scatter_mte.h
 * \brief quant_reduce_scatter mte通信kernel代码逻辑
 */

#ifndef QUANT_REDUCE_SCATTER_MTE_H
#define QUANT_REDUCE_SCATTER_MTE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/hccl/hccl.h"
#include "adv_api/reduce/sum.h"
#include "adv_api/pad/broadcast.h"
#include "kernel_tiling/kernel_tiling.h"
#include "quant_reduce_scatter_tiling_data.h"
#include "utils.h"
#include "mte_comm.h"
#include "vec_comp.h"

namespace QuantReduceScatterImpl {

using namespace QuantMTECommImpl;
using namespace VectorComputeImpl;
using namespace AscendC;

// 之后可修改成从tiling侧获取数据切块大小
constexpr static uint32_t X_PRE_BLOCK_NUM = 1024U;  // 当前一次搬运一个x数据块，x dtype为 8bit 时对应 1024个x数据. 对于fp4需要另外算
constexpr static uint64_t MX_SCALES_LAST_DIM = 2U; // MX量化scales最后一维的大小

template<TemplateTypeClass>
class QuantReduceScatterMte {
public:
    __aicore__ inline QuantReduceScatterMte() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scales, GM_ADDR output,
                                TPipe *pipe, const QuantReduceScatterTilingData *tilingData);
    __aicore__ inline void Process();
private:
    __aicore__ inline void ClearSumTensor();
    __aicore__ inline void ReadDataBlockReduceSum(uint64_t curXOffset, uint64_t curScaleOffset);
    __aicore__ inline void ExecuteReduceScatter();

    MTECommunication<TemplateType> mteComm_; // MTE 通信相关实现
    VectorCompute<TemplateType> vecComp_; // vector 计算相关实现

    GlobalTensor<XType> remoteWinXTensor_;
    GlobalTensor<ScalesType> remoteWinScaleTensor_;
    LocalTensor<float> sumTensor_;

    TQue<QuePosition::VECIN, 1> xInQueue_, scaleInQue; // 用于读数据和反量化求和的通算并行
    TBuf<> sumBuf_; // 用于Reduce_sum 求和

    uint64_t xSize_{0};
    uint64_t totalWinSize_{0};
    uint64_t tailXNums_{0};
    uint32_t totalBlockNums_{0};
    uint64_t alignedXSize_{0};
    // reduceScater进行all2all过程，数据要按卡数进行均分，以下为计算切分相关的参数
    uint64_t scaleSize_{0};
    uint64_t xSliceSize_{0};
    uint64_t xSliceSizeNums_{0};
    uint64_t scaleSliceNums_{0};
};

template <TemplateTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateType>::Init(GM_ADDR x, GM_ADDR scales,
    GM_ADDR output, TPipe *tPipe, const QuantReduceScatterTilingData *tilingData)
{
    mteComm_.InitHcclContext();
    mteComm_.ComputeBlockParams();

    auto&& tiliingDatainfo = tilingData->quantReduceScatterTilingInfo;
    totalWinSize_ = tiliingDatainfo.totalWinSize;
    xSize_ = tiliingDatainfo.bs * tiliingDatainfo.hiddenSize * sizeof(XType);
    scaleSize_ = tiliingDatainfo.bs * tiliingDatainfo.scaleHiddenSize * sizeof(ScalesType);
    if constexpr(AscendC::IsSameType<ScalesType, fp8_e8m0_t>::value) {
        scaleSize_ *= MX_SCALES_LAST_DIM;
    }
    xSliceSize_ = xSize_ / mteComm_.hcclContext_->rankDim;
    xSliceSizeNums_ = xSliceSize_ / sizeof(XType);
    scaleSliceNums_ = scaleSize_ / (mteComm_.hcclContext_->rankDim * sizeof(ScalesType));
    tailXNums_ = BlockAlignMod(xSliceSizeNums_, mteComm_.xNumPerBlock_);
    uint32_t xBlockBytes = mteComm_.xNumPerBlock_ * sizeof(XType);
    
    // [PUT模式] 每核处理xSliceSizeNums_的一部分写入对应targetRank
    // AIV分组：sendCoreNumPerRank核负责一个targetRank的写入
    mteComm_.SetSendCoreNumPerRank(tiliingDatainfo.aivNum / mteComm_.hcclContext_->rankDim);
    
    totalBlockNums_ = CeilDiv(xSliceSize_, xBlockBytes);
    mteComm_.round_ = totalBlockNums_ / mteComm_.sendCoreNumPerRank_;
    mteComm_.tailBlockNums_ = totalBlockNums_ % mteComm_.sendCoreNumPerRank_;
    mteComm_.ComputeTailAivId(tiliingDatainfo.aivNum);
    
    tPipe->Reset();
    tPipe->InitBuffer(xInQueue_, BUFFER_NUM, xBlockBytes);
    tPipe->InitBuffer(scaleInQue, BUFFER_NUM, UB_ALIGN_BYTES);
    tPipe->InitBuffer(sumBuf_, mteComm_.xNumPerBlock_ * sizeof(float));
    sumTensor_ = sumBuf_.Get<float>();

    mteComm_.SetBlockSize(mteComm_.xNumPerBlock_, tiliingDatainfo.aivNum, tailXNums_);
    vecComp_.SetBlockSize(mteComm_.xNumPerBlock_);  

    // [PUT模式+AIV分组] 初始化：每组核负责一个targetRank分区
    mteComm_.InitParamsForAIVGroup(xSliceSizeNums_);

    mteComm_.InitBuffer(tPipe);
    vecComp_.InitBuffer(tPipe);

    mteComm_.InitGMTensor(x, scales, output, xSize_, totalWinSize_);
}

template <TemplateTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateType>::ReadDataBlockReduceSum(uint64_t curXOffset,
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
__aicore__ inline void QuantReduceScatterMte<TemplateType>::ClearSumTensor()
{
    Duplicate<float>(sumTensor_, (float)0.0, mteComm_.xNumPerBlock_); // sumTensor 清零，使用动态块大小
}

template <TemplateTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateType>::ExecuteReduceScatter()
{   
    mteComm_.ReadStatus(); 
    
    if (mteComm_.xOffset_ >= xSliceSizeNums_) {
        PipeBarrier<PIPE_ALL>();
        return;
    }
    
    // [PUT模式] 获取本卡Win Area数据区地址
    // Win Area[本卡]存储所有rank对本卡分区的数据
    // 布局：rank j分区起始 = j * xSliceSizeNums
    GM_ADDR selfWinDataGm = mteComm_.GetWinDataAddrGm(mteComm_.hcclContext_->rankId, mteComm_.winBufferFlags_);
    remoteWinXTensor_.SetGlobalBuffer((__gm__ XType*)selfWinDataGm);
    
    // ReduceScatter Win Area大小 = xSize_（所有rank分区数据）
    // scale区起始偏移 = xSize_
    remoteWinScaleTensor_.SetGlobalBuffer((__gm__ ScalesType*)(selfWinDataGm + xSize_));
    
    uint32_t lastCoreIndexForRank = mteComm_.sendCoreNumPerRank_ - 1;
    
    // [关键] 遍历本核负责的数据块
    // 每核负责输出数据的一部分（对应remoteRankId_分区的一部分）
    for (uint64_t curBlock = 0; curBlock < mteComm_.assignedBlockNums_; ++curBlock) {
        // 分区内偏移：xOffset_来自InitParamsForAIVGroup，按coreInnerIndex_计算
        uint64_t curXOffset = mteComm_.xOffset_ + curBlock * mteComm_.xNumPerBlock_;
        uint64_t curScaleOffset = mteComm_.scaleOffset_ + curBlock * mteComm_.scaleNumsPerBlcok_;
        ClearSumTensor();
        
        // [ReduceScatter语义] 从所有rank分区读取对应位置数据并ReduceSum
        // 每个rank分区的大小 = xSliceSizeNums_
        // 读取位置：srcRankId分区 + curXOffset
        uint32_t startRankId = mteComm_.hcclContext_->rankId;
        for (uint32_t i = 0; i < mteComm_.hcclContext_->rankDim; ++i) {
            uint32_t srcRankId = (startRankId + i) % mteComm_.hcclContext_->rankDim;
            
            uint64_t winXOffset = srcRankId * xSliceSizeNums_ + curXOffset;
            uint64_t winScaleOffset = srcRankId * scaleSliceNums_ + curScaleOffset;
            ReadDataBlockReduceSum(winXOffset, winScaleOffset);
        }

        // 输出位置：本核负责的分区内偏移
        uint32_t copyBlockNum = mteComm_.xNumPerBlock_;
        if ((mteComm_.coreInnerIndex_ == lastCoreIndexForRank) && (curBlock == mteComm_.assignedBlockNums_ - 1)) {
            copyBlockNum = tailXNums_;
        }
        mteComm_.CopyResultToOutput(curXOffset, sumTensor_, copyBlockNum);
    }
}

template <TemplateTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateType>::Process()
{
    // 纯AIV过程
    if ASCEND_IS_AIC {
        return;
    }

    // 边界检查：xOffset_ >= xSliceSizeNums_ 时跳过整个流程（该核无数据处理）
    if (mteComm_.xOffset_ >= xSliceSizeNums_) {
        return;
    }

    // [PUT 模式 + AIV 分组] 本卡数据写入目标 rank 分区（AllToAll 语义）
    // 每核负责 remoteRankId_ 分区的数据写入到 remoteRankId_ 的 Win Area
    mteComm_.template PutDataToAllWin<true>(xSize_, xSliceSizeNums_, scaleSliceNums_);
    // 写入状态到状态区
    mteComm_.WriteStatusToWin();
    // 执行ReduceScatter过程：等待状态区同步，从本卡Win Area读取数据并进行反量化ReduceSum
    ExecuteReduceScatter();
}
} // QuantReduceScatterImpl
#endif  // QUANT_REDUCE_SCATTER_MTE_H