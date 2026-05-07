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
    __aicore__ inline void ClearSumTensor(uint32_t count);
    __aicore__ inline void ReadDataBlockReduceSum(uint64_t curXOffset, uint64_t curScaleOffset, uint32_t xCount, uint32_t scaleCount);
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

    /* quant_all_reduce自己的数据 */
    auto&& tiliingDatainfo = tilingData->quantAllReduceTilingInfo;
    totalWinSize_ = tiliingDatainfo.totalWinSize;
    xNums_ = tiliingDatainfo.bs * tiliingDatainfo.hiddenSize; // 总的x数据个数， bs * h
    xSize_ = xNums_ * sizeof(XType); // 总的x数据量，B
    tailXNums_ = BlockAlignMod(xNums_, X_PRE_BLOCK_NUM); // 计算最后一个数据块的大小
    totalBlockNums_ = CeilDiv(xSize_, X_BLOCK_BYTES); // 按每次搬运x的数据量分块，得到的总块数
    mteComm_.round_ = totalBlockNums_ / tiliingDatainfo.aivNum; // 计算总的数据分核搬运需要的轮次数
    mteComm_.tailBlockNums_ = totalBlockNums_ % tiliingDatainfo.aivNum; // 搬运的尾块数
    mteComm_.ComputeTailAivId(tiliingDatainfo.aivNum); // 计算最后一个核的id
    tPipe->Reset();
    tPipe->InitBuffer(xInQueue_, BUFFER_NUM, X_BLOCK_BYTES); // 每次拷贝 1024B x; 128 * 8
    tPipe->InitBuffer(scaleInQue, BUFFER_NUM, UB_ALIGN_BYTES); // 每次拷贝 32B scale；4 * 8
    tPipe->InitBuffer(sumBuf_, X_PRE_BLOCK_NUM * sizeof(float)); // 用于Reduce_sum 求和，1024 * 4 = 4k
    sumTensor_ = sumBuf_.Get<float>();

    // 设置切块大小
    mteComm_.SetBlockSize(X_PRE_BLOCK_NUM, tiliingDatainfo.aivNum, tailXNums_);
    vecComp_.SetBlockSize(X_PRE_BLOCK_NUM);  

    // 公共MTE搬运参数计算
    mteComm_.InitParams();

    // 初始化tPipe的各种buffer
    mteComm_.InitBuffer(tPipe);
    vecComp_.InitBuffer(tPipe);

    // 初始化GM上的Tensor，包括Win区
    mteComm_.InitGMTensor(x, scales, output, xSize_, totalWinSize_);
}

template <TemplateTypeClass>
__aicore__ inline void QuantAllReduceMteOneShot<TemplateType>::ReadDataBlockReduceSum(uint64_t curXOffset,
    uint64_t curScaleOffset, uint32_t xCount, uint32_t scaleCount)
{
    LocalTensor<XType> xTmpTensor = xInQueue_.AllocTensor<XType>();
    DataCopy(xTmpTensor, remoteWinXTensor_[curXOffset], xCount);
    xInQueue_.EnQue(xTmpTensor);
    xTmpTensor = xInQueue_.DeQue<XType>();

    LocalTensor<ScalesType> scaleTmpTensor = scaleInQue.AllocTensor<ScalesType>();
    DataCopy(scaleTmpTensor, remoteWinScaleTensor_[curScaleOffset], scaleCount);
    scaleInQue.EnQue(scaleTmpTensor);
    scaleTmpTensor = scaleInQue.DeQue<ScalesType>();

    vecComp_.DequantReduceSum(xTmpTensor, scaleTmpTensor, sumTensor_, xCount); 
    xInQueue_.FreeTensor(xTmpTensor);
    scaleInQue.FreeTensor(scaleTmpTensor);
}

template <TemplateTypeClass>
__aicore__ inline void QuantAllReduceMteOneShot<TemplateType>::ClearSumTensor(uint32_t count)
{
    Duplicate<float>(sumTensor_, (float)0.0, count);
}

template <TemplateTypeClass>
__aicore__ inline void QuantAllReduceMteOneShot<TemplateType>::ExecuteAllReduce()
{   
    mteComm_.ReadStatus();
    for (uint64_t curBlock = 0; curBlock < mteComm_.assignedBlockNums_; ++curBlock) {
        uint64_t curXOffset = mteComm_.xOffset_ + curBlock * X_PRE_BLOCK_NUM;
        uint64_t curScaleOffset = mteComm_.scaleOffset_ + curBlock * mteComm_.scaleNumsPerBlcok_;
        uint32_t xCount = X_PRE_BLOCK_NUM;
        uint32_t scaleCount = mteComm_.scaleNumsPerBlcok_;
        if ((mteComm_.aivId_ ==  mteComm_.lastAivId_) && (curBlock == mteComm_.assignedBlockNums_ - 1)) {
            xCount = tailXNums_;
            if constexpr (AscendC::IsSameType<ScalesType, fp8_e8m0_t>::value) {
                scaleCount = CeilDiv(tailXNums_, MX_SIZE);
            } else {
                scaleCount = CeilDiv(tailXNums_, PER_GROUP_SIZE);
            }
        }
        ClearSumTensor(xCount);
        uint32_t startRankId = mteComm_.hcclContext_->rankId;
        for (uint32_t i = 0; i < mteComm_.hcclContext_->rankDim; ++i) {
            uint32_t remoteRankId = (startRankId + i) % mteComm_.hcclContext_->rankDim;
            GM_ADDR remoteDataSpaceGm = mteComm_.GetWinDataAddrGm(remoteRankId, mteComm_.winBufferFlags_);
            remoteWinXTensor_.SetGlobalBuffer((__gm__ XType*)remoteDataSpaceGm);
            remoteWinScaleTensor_.SetGlobalBuffer((__gm__ ScalesType*)(remoteDataSpaceGm + xSize_));
            ReadDataBlockReduceSum(curXOffset, curScaleOffset, xCount, scaleCount);
        }
        mteComm_.CopyResultToOutput(curXOffset, sumTensor_, xCount);
    }
}

template <TemplateTypeClass>
__aicore__ inline void QuantAllReduceMteOneShot<TemplateType>::Process()
{
    if ASCEND_IS_AIC {
        return;
    }

    uint32_t tailScaleNums = 0;
    if constexpr (AscendC::IsSameType<ScalesType, fp8_e8m0_t>::value) {
        tailScaleNums = CeilDiv(tailXNums_, MX_SIZE);
    } else {
        tailScaleNums = CeilDiv(tailXNums_, PER_GROUP_SIZE);
    }
    mteComm_.CopyDataToWin(0, 0, tailScaleNums);
    mteComm_.WriteStatusToWin();
    ExecuteAllReduce();
}
} // QuantAllReduceImpl
#endif  // QUANT_ALL_REDUCE_MTE_ONE_SHOT_H