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
// 当前一次搬运一个x数据块，xDtype为8bit时对应1024个x数据。对于float4_e2m1或者float4_e1m2类型则需要另外算
constexpr static uint32_t X_PRE_BLOCK_NUM = 1024U;
// MX量化scales最后一维的大小
constexpr static uint64_t MX_SCALES_LAST_DIM = 2U;

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

    // 声明一个MTECommunication通信对象，调用MTE对象下面通信相关的函数实现
    MTECommunication<TemplateType> mteComm_;
    // 声明一个VectorCompute计算对象，调用Vector对象下面计算相关的函数实现
    VectorCompute<TemplateType> vecComp_;

    // GM上申请一块内存，作为远端win区
    GlobalTensor<XType> remoteWinXTensor_;
    GlobalTensor<ScalesType> remoteWinScaleTensor_;
    // 本地UB上申请一块内存，用于存储ReduceSum的结果
    LocalTensor<float> sumTensor_;

    // 创建CopyIn阶段的队列，用于读数据和反量化求和的通算并行
    TQue<QuePosition::VECIN, 1> xInQueue_, scaleInQue;
    // 用于Reduce_sum 求和，Tbuf所在的逻辑位置默认为VECCALC，用于UB上计算临时存储使用
    TBuf<> sumBuf_;

    // x总的数据量大小：b*s*h*sizeof(xDtype)
    uint64_t xSize_{0};
    // Win区总数据量大小，即HCCL_BUFFER_SIZE
    uint64_t totalWinSize_{0};
    // 按卡数均分后，每张卡的尾块中x的个数
    uint64_t tailXNums_{0};
    // xSliceSize_按照每个block为1024B切分，得到的总的块数
    uint32_t totalBlockNums_{0};
    uint64_t alignedXSize_{0};

    /* reduceScatter进行all2all过程，数据要按卡数进行均分，以下为计算切分相关的参数 */
    // scale总的数据量大小：b*s*h*sizeof(scaleDtype)
    uint64_t scaleSize_{0};
    // x按照卡数均分，每张卡上的数据量大小
    uint64_t xSliceSize_{0};
    // x按照卡数均分，每张卡上x的个数
    uint64_t xSliceNum_{0};
    // scale按照卡数均分，每张卡上scale的个数
    uint64_t scaleSliceNum_{0};
};

template <TemplateTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateType>::Init(GM_ADDR x, GM_ADDR scales,
    GM_ADDR output, TPipe *tPipe, const QuantReduceScatterTilingData *tilingData)
{
    // 初始化HcclContext
    mteComm_.InitHcclContext();
    /* quant_reduce_scatter自己的数据 */
    auto&& tilingDataInfo = tilingData->quantReduceScatterTilingInfo;
    totalWinSize_ = tilingDataInfo.totalWinSize;
    xSize_ = tilingDataInfo.bs * tilingDataInfo.hiddenSize * sizeof(XType);
    scaleSize_ = tilingDataInfo.bs * tilingDataInfo.scaleHiddenSize * sizeof(ScalesType);
    // MX量化模式下scale是三维，最后一维为2，总scales的数据量需要再乘以2
    if constexpr(AscendC::IsSameType<ScalesType, fp8_e8m0_t>::value) {
        scaleSize_ *= MX_SCALES_LAST_DIM;
    }
    // all2all过程，按卡数均分后每张卡上的x的数据量大小
    xSliceSize_ = xSize_ / mteComm_.hcclContext_->rankDim;
    // 按卡均分后，每张卡上x的个数
    xSliceNum_ = xSliceSize_ / sizeof(XType);
    // all2all过程，按卡数均分后每张卡上的scale的个数
    scaleSliceNum_ = scaleSize_ / (mteComm_.hcclContext_->rankDim * sizeof(ScalesType));
    // 每张卡上x的最后一个数据块的大小，计算结果要么是X_PRE_BLOCK_NUM，要么是取余数
    tailXNums_ = BlockAlignMod(xSliceNum_, X_PRE_BLOCK_NUM);
    // 每张卡上的x的数据量大小 / 每个block包含的x的数据量大小，得到需要搬运的x的总块数
    totalBlockNums_ = CeilDiv(xSliceSize_, X_BLOCK_BYTES);
    // 计算数据分核搬运需要的轮次数
    mteComm_.round_ = totalBlockNums_ / tilingDataInfo.aivNum;
    // 搬运的尾块数
    mteComm_.tailBlockNums_ = totalBlockNums_ % tilingDataInfo.aivNum;
    // 计算最后一个核的aivId
    mteComm_.ComputeTailAivId(tilingDataInfo.aivNum);
    tPipe->Reset();
    // 每次拷贝x: 128 * 8 = 1024B，其中groupSize是128
    tPipe->InitBuffer(xInQueue_, BUFFER_NUM, X_BLOCK_BYTES);
    // 每次拷贝scale: 4 * 8 = 32B，这里是按照32B对齐，float32对应4B，计算得到8个scale
    tPipe->InitBuffer(scaleInQue, BUFFER_NUM, UB_ALIGN_BYTES);
    // 用于ReduceSum求和，1024 * 4 = 4KB
    tPipe->InitBuffer(sumBuf_, X_PRE_BLOCK_NUM * sizeof(float));
    sumTensor_ = sumBuf_.Get<float>();
    // 设置quant_reduce_scatter切块大小
    mteComm_.SetBlockSize(X_PRE_BLOCK_NUM, tilingDataInfo.aivNum, tailXNums_);
    vecComp_.SetBlockSize(X_PRE_BLOCK_NUM);  
    // 公共MTE搬运参数计算
    mteComm_.InitParams();
    // 初始化tPipe的各种buffer
    mteComm_.InitBuffer(tPipe);
    vecComp_.InitBuffer(tPipe);
    // 初始化GM上的Tensor，包括Win区
    mteComm_.InitGMTensor(x, scales, output, xSize_, totalWinSize_);
}

/**
 * @brief 从win区读数据（x和scale）到本地UB，然后计算dequant和reduceSum
 */
template <TemplateTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateType>::ReadDataBlockReduceSum(
    uint64_t curXOffset, uint64_t curScaleOffset)
{
    /* 读取 x 从 Win -> UB */
    LocalTensor<XType> xTmpTensor = xInQueue_.AllocTensor<XType>();
    // 从远端win区读取x到UB
    DataCopy(xTmpTensor, remoteWinXTensor_[curXOffset], X_PRE_BLOCK_NUM);
    // 固定写法
    xInQueue_.EnQue(xTmpTensor);
    xTmpTensor = xInQueue_.DeQue<XType>();

    /* 读取 scale 从 Win -> UB */
    LocalTensor<ScalesType> scaleTmpTensor = scaleInQue.AllocTensor<ScalesType>();
    DataCopy(scaleTmpTensor, remoteWinScaleTensor_[curScaleOffset], mteComm_.scaleNumsPerBlcok_);
    scaleInQue.EnQue(scaleTmpTensor);
    scaleTmpTensor = scaleInQue.DeQue<ScalesType>();

    /* 反量化计算与ReduceSum求和 */
    vecComp_.DequantReduceSum(xTmpTensor, scaleTmpTensor, sumTensor_); 
    xInQueue_.FreeTensor(xTmpTensor);
    scaleInQue.FreeTensor(scaleTmpTensor);
}

/**
 * @brief sumTensor_清零
 */
template <TemplateTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateType>::ClearSumTensor()
{
    Duplicate<float>(sumTensor_, (float)0.0, X_PRE_BLOCK_NUM);
}

/**
 * @brief 执行ReduceScatter核心逻辑
 */
template <TemplateTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateType>::ExecuteReduceScatter()
{   
    // 读状态位，软同步
    mteComm_.ReadStatus();
    // 遍历需要搬运的数据块
    for (uint64_t curBlock = 0; curBlock < mteComm_.assignedBlockNums_; ++curBlock) {
        // 一次搬一个block，一个block大小为1024B
        uint64_t curXOffset = mteComm_.xOffset_ + curBlock * X_PRE_BLOCK_NUM;
        // 一块scale固定32B / sizeof(scaleType)
        uint64_t curScaleOffset = mteComm_.scaleOffset_ + curBlock * mteComm_.scaleNumsPerBlcok_;
        // sumTensor_清零
        ClearSumTensor();
        // 遍历每张卡，读取其Win区的数据，采取错卡序读取，从自己卡上读起
        uint32_t startRankId = mteComm_.hcclContext_->rankId;
        for (uint32_t i = 0; i < mteComm_.hcclContext_->rankDim; ++i) {
            // rank0: [0,1,2]; rank1: [1,2,0]; rank2: [2,0,1]
            uint32_t remoteRankId = (startRankId + i) % mteComm_.hcclContext_->rankDim;
            // 获取对端Win区中数据区相关的地址
            GM_ADDR remoteDataSpaceGm = mteComm_.GetWinDataAddrGm(remoteRankId, mteComm_.winBufferFlags_);

            remoteWinXTensor_.SetGlobalBuffer((__gm__ XType*)remoteDataSpaceGm);
            // 这里看起来是scale接在x地址后面
            remoteWinScaleTensor_.SetGlobalBuffer((__gm__ ScalesType*)(remoteDataSpaceGm + xSize_));

            /* 读取对端对应地址的x和scale数据，进行反量化和求和 */
            // xSliceNum_ = 总数据量/卡数/sizeof(xType)，表示每张卡上x的个数
            uint64_t curRankXOffset = curXOffset + mteComm_.hcclContext_->rankId * xSliceNum_;
            // scaleSliceNum_ = 总数据量/卡数/sizeof(scaleType)，表示每张卡上scale的个数
            uint64_t curRankScaleOffset = curScaleOffset + mteComm_.hcclContext_->rankId * scaleSliceNum_;
            // ReduceScatter过程，all2all仅需与rankId相关的数据，加上本卡偏移
            ReadDataBlockReduceSum(curRankXOffset, curRankScaleOffset);
        }

        /* 将计算好的数据拷贝到输出tensor */
        uint32_t copyBlockNum = X_PRE_BLOCK_NUM;
        // 如果是最后一个核，并且是最后这个核的最后一个数据块
        if ((mteComm_.aivId_ ==  mteComm_.lastAivId_) && (curBlock == mteComm_.assignedBlockNums_ - 1)) {
            // 最后尾块的搬运数
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

    // 一次性拷贝完所有数据到本地卡win区
    mteComm_.template CopyDataToWin<true>(xSliceNum_, scaleSliceNum_);
    // 写入状态到状态区
    mteComm_.WriteStatusToWin();
    // 执行ReduceScatter过程：等待状态区同步，读取数据并进行反量化ReduceSum
    ExecuteReduceScatter();
}

} // QuantReduceScatterImpl

#endif  // QUANT_REDUCE_SCATTER_MTE_H