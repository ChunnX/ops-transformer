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
    __aicore__ inline void ReadDataBlockReduceSum(uint64_t curXOffset, uint64_t curScaleOffset,
                                                    uint32_t xNum, uint32_t scaleNum);
    __aicore__ inline void ExecuteReduceScatter();
    __aicore__ inline void ParseTilingInfo(const QuantReduceScatterTilingData *tilingData);
    __aicore__ inline void ComputeXPerBlock(const QuantReduceScatterTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void ComputeBlockDistribution(uint64_t aivNum);
    __aicore__ inline void InitSubModules(GM_ADDR x, GM_ADDR scales, GM_ADDR output, TPipe *tPipe, uint64_t aivNum);

    MTECommunication<TemplateType> mteComm_; // MTE 通信相关实现
    VectorCompute<TemplateType> vecComp_; // vector 计算相关实现

    GlobalTensor<XType> remoteWinXTensor_;
    GlobalTensor<ScalesType> remoteWinScaleTensor_;
    LocalTensor<float> sumTensor_;

    TQue<QuePosition::VECIN, 1> xInQueue_, scaleInQue_; // 用于读数据和反量化求和的通算并行
    TBuf<> sumBuf_; // 用于Reduce_sum 求和

    uint64_t xSize_{0};
    uint64_t totalWinSize_{0};
    // reduceScater进行all2all过程，数据要按卡数进行均分，以下为计算切分相关的参数
    uint64_t scaleSize_{0};
    uint64_t xSliceSizeNums_{0};      // 按卡均分后每片x数据的个数
    uint64_t scaleSliceNums_{0};      // 按卡均分后每片scale数据的个数

    // 动态分块参数（基于xPerBlock_最大化UB利用）
    uint32_t xPerBlock_{0};           // 动态计算的每块数据元素数
    uint32_t scaleNumsPerBlock_{0};   // 每块对应的scale数量
    uint32_t totalBlockNums_{0};    // 总块数（基于xPerBlock_）
    uint32_t assignedBlockNums_{0}; // 当前核分配的块数
    uint32_t round_{0};             // 基础轮次数
    uint32_t tailBlockNums_{0};     // 尾块数（分配给前几个核）
    uint32_t lastAivId_{0};         // 最后一个核的ID
    uint64_t xOffset_{0};           // 当前核的起始x偏移
    uint64_t scaleOffset_{0};       // 当前核的起始scale偏移
    uint32_t tailXNums_{0};         // 尾块的x数量
};

template <TemplateTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateType>::Init(GM_ADDR x, GM_ADDR scales,
    GM_ADDR output, TPipe *tPipe, const QuantReduceScatterTilingData *tilingData)
{
    mteComm_.InitHcclContext();
    ParseTilingInfo(tilingData);
    tPipe->Reset();
    ComputeXPerBlock(tilingData, tPipe);
    uint64_t aivNum = tilingData->quantReduceScatterTilingInfo.aivNum;
    ComputeBlockDistribution(aivNum);
    InitSubModules(x, scales, output, tPipe, aivNum);
}

template <TemplateTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateType>::ParseTilingInfo(
    const QuantReduceScatterTilingData *tilingData)
{
    auto&& info = tilingData->quantReduceScatterTilingInfo;
    totalWinSize_ = info.totalWinSize;
    xSize_ = info.bs * info.hiddenSize * sizeof(XType);
    scaleSize_ = info.bs * info.scaleHiddenSize * sizeof(ScalesType);
    if constexpr(AscendC::IsSameType<ScalesType, fp8_e8m0_t>::value) {
        scaleSize_ *= MX_SCALES_LAST_DIM;
    }
    uint64_t xSliceSize = xSize_ / (mteComm_.hcclContext_->rankDim);
    xSliceSizeNums_ = xSliceSize / sizeof(XType);
    scaleSliceNums_ = scaleSize_ / (mteComm_.hcclContext_->rankDim * sizeof(ScalesType));
}

template <TemplateTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateType>::ComputeXPerBlock(
    const QuantReduceScatterTilingData *tilingData, TPipe *tPipe)
{
    auto&& info = tilingData->quantReduceScatterTilingInfo;

    // 固定开销：不随 xPerBlock_ 变化的 buffer
    uint64_t mteFixedUbCost = BUFFER_NUM * X_BLOCK_BYTES +      // scaleQueue_
                                  UB_ALIGN_BYTES +                  // winFlagsBuf_
                                  UB_ALIGN_BYTES +                  // writeStateBuf_
                                  mteComm_.hcclContext_->rankDim * UB_ALIGN_BYTES +  // readStateBuf_
                                  mteComm_.hcclContext_->rankDim * UB_ALIGN_BYTES;   // stateResetBuf_

    uint64_t dynUbPerX = BUFFER_NUM * sizeof(OutputType) +  // xOutQueue_
                            BUFFER_NUM * sizeof(XType) +       // xQueue_
                            sizeof(float) +                    // brcbBuf_
                            sizeof(float) +                    // xCastBuf_
                            sizeof(bfloat16_t) +               // xCastBf16Buf_
                            sizeof(float16_t) +                // xCastfp16Buf_
                            sizeof(float) +                    // sumBuf_
                            BUFFER_NUM * sizeof(XType);        // xInQueue_

    uint64_t availUbSpace = TOTAL_UB_SIZE - mteFixedUbCost;
    uint64_t ubMaxXCount;
    if constexpr (AscendC::IsSameType<ScalesType, fp8_e8m0_t>::value) {
        ubMaxXCount = FracDiv(availUbSpace, dynUbPerX, 4U, 1U);
    } else {
        ubMaxXCount = FracDiv(availUbSpace, dynUbPerX, 32U, 3U);
    }

    uint32_t hostXLimit = info.xPerBlock;
    if (hostXLimit != 0U && ubMaxXCount > hostXLimit) {
        ubMaxXCount = hostXLimit;
    }

    // 每卡分片数据量上限约束
    xPerBlock_ = static_cast<uint32_t>(ubMaxXCount < xSliceSizeNums_ ? ubMaxXCount : xSliceSizeNums_);

    // 对齐到 alignBlock
    uint32_t alignGranularity = info.alignBlock;
    xPerBlock_ = FloorAlign(xPerBlock_, alignGranularity);
    if (xPerBlock_ == 0) {
        xPerBlock_ = alignGranularity;
    }

    // 每块 scale 数量（real 值，GM 偏移用；UB 分配按 32B 对齐）
    if constexpr (AscendC::IsSameType<ScalesType, fp8_e8m0_t>::value) {
        scaleNumsPerBlock_ = xPerBlock_ / MX_SIZE;
    } else {
        scaleNumsPerBlock_ = xPerBlock_ / PER_GROUP_SIZE;
    }

    uint32_t scaleAlignUnit = UB_ALIGN_BYTES / sizeof(ScalesType);
    uint32_t scaleBufferCount = CeilAlignU32(scaleNumsPerBlock_, scaleAlignUnit);
    uint64_t scaleBufByteSize = scaleBufferCount * sizeof(ScalesType);

    tPipe->InitBuffer(xInQueue_, BUFFER_NUM, xPerBlock_ * sizeof(XType));
    tPipe->InitBuffer(scaleInQue_, BUFFER_NUM, scaleBufByteSize);
    tPipe->InitBuffer(sumBuf_, xPerBlock_ * sizeof(float));
    sumTensor_ = sumBuf_.Get<float>();
}

// 计算多核分块参数：每核分配块数、尾块归属、起始偏移
template <TemplateTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateType>::ComputeBlockDistribution(uint64_t aivNum)
{
    uint64_t aivId = GetBlockIdx();
    totalBlockNums_ = CeilDiv(xSliceSizeNums_, xPerBlock_);
    round_ = totalBlockNums_ / aivNum;
    tailBlockNums_ = totalBlockNums_ % aivNum;
    assignedBlockNums_ = aivId < tailBlockNums_ ? round_ + 1 : round_;
    lastAivId_ = (round_ == 0) ? tailBlockNums_ - 1 : aivNum - 1;

    uint64_t blockIdx = aivId * round_ + (aivId < tailBlockNums_ ? aivId : tailBlockNums_);
    xOffset_ = blockIdx * xPerBlock_;
    scaleOffset_ = blockIdx * scaleNumsPerBlock_;
    tailXNums_ = BlockAlignMod(xSliceSizeNums_, xPerBlock_);
}

// 初始化 vecComp、mteComm 子模块参数并分配其 UB buffer，绑定 GM Tensor
template <TemplateTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateType>::InitSubModules(
    GM_ADDR x, GM_ADDR scales, GM_ADDR output, TPipe *tPipe, uint64_t coreNum)
{
    vecComp_.SetBlockSize(xPerBlock_);
    vecComp_.SetScaleNums(scaleNumsPerBlock_);
    mteComm_.round_ = round_;
    mteComm_.tailBlockNums_ = tailBlockNums_;
    mteComm_.ComputeTailAivId(coreNum);
    mteComm_.SetBlockSize(xPerBlock_, coreNum, tailXNums_, scaleNumsPerBlock_);
    mteComm_.InitParams();
    mteComm_.InitBuffer(tPipe);
    vecComp_.InitBuffer(tPipe);
    mteComm_.InitGMTensor(x, scales, output, xSize_, totalWinSize_);
}

template <TemplateTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateType>::ReadDataBlockReduceSum(uint64_t curXOffset,
    uint64_t curScaleOffset, uint32_t xNum, uint32_t scaleNum)
{
    /* 读取 x 从 Win -> UB */
    LocalTensor<XType> xLocalTensor = xInQueue_.AllocTensor<XType>();
    DataCopy(xLocalTensor, remoteWinXTensor_[curXOffset], xNum);
    xInQueue_.EnQue(xLocalTensor);
    xLocalTensor = xInQueue_.DeQue<XType>();

    /* 读取 scale 从 Win -> UB（DataCopyPad 处理 sub-32B 对齐） */
    LocalTensor<ScalesType> scaleLocalTensor = scaleInQue_.AllocTensor<ScalesType>();
    DataCopyParams scaleDcpyParams;
    scaleDcpyParams.blockLen = scaleNum * sizeof(ScalesType);
    scaleDcpyParams.blockCount = 1;
    DataCopyPadParams scalePadCfg;
    DataCopyPad(scaleLocalTensor, remoteWinScaleTensor_[curScaleOffset], scaleDcpyParams, scalePadCfg);
    scaleInQue_.EnQue(scaleLocalTensor);
    scaleLocalTensor = scaleInQue_.DeQue<ScalesType>();

    /* 反量化计算与ReduceSum求和 */
    vecComp_.DequantReduceSum(xLocalTensor, scaleLocalTensor, sumTensor_);
    xInQueue_.FreeTensor(xLocalTensor);
    scaleInQue_.FreeTensor(scaleLocalTensor);
}


template <TemplateTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateType>::ClearSumTensor()
{
    Duplicate<float>(sumTensor_, (float)0.0, xPerBlock_); // sumTensor 清零
}

template <TemplateTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateType>::ExecuteReduceScatter()
{
    // 读状态位，软同步
    mteComm_.ReadStatus();
    uint64_t coreIdx = GetBlockIdx();

    // 遍历需要处理的数据块（基于动态xPerBlock_分块）
    for (uint64_t blkIdx = 0; blkIdx < assignedBlockNums_; ++blkIdx) {
        uint64_t curXOff = xOffset_ + blkIdx * xPerBlock_;
        uint64_t curScaleOff = scaleOffset_ + blkIdx * scaleNumsPerBlock_;

        // 计算当前块的x数量（尾块处理）
        uint32_t curXCnt = xPerBlock_;
        uint32_t curScaleCnt = scaleNumsPerBlock_;
        if ((coreIdx == lastAivId_) && (blkIdx == assignedBlockNums_ - 1)) {
            curXCnt = tailXNums_;
            // 尾块的scale数量：real 值（DataCopyPad 处理 sub-32B 拷贝，不再需要硬对齐）
            if constexpr (AscendC::IsSameType<ScalesType, fp8_e8m0_t>::value) {
                curScaleCnt = CeilDivU32(curXCnt, MX_SIZE);
            } else {
                curScaleCnt = CeilDivU32(curXCnt, PER_GROUP_SIZE);
            }
        }

        ClearSumTensor(); // sumTensor 清零

        // 遍历每张卡，读取其Win区的数据，采取错卡序读取，从自己卡上读起
        /* rank0: [0,1,2]; rank1: [1,2,0]; rank2: [2,0,1] */
        uint32_t initRankId = mteComm_.hcclContext_->rankId;
        for (uint32_t i = 0; i < mteComm_.hcclContext_->rankDim; ++i) {
            uint32_t peerRankId = (initRankId + i) % mteComm_.hcclContext_->rankDim;

            // 获取对端Win区中数据区相关的地址
            GM_ADDR peerDataGm = mteComm_.GetWinDataAddrGm(peerRankId, mteComm_.winBufferFlags_);

            remoteWinXTensor_.SetGlobalBuffer((__gm__ XType*)peerDataGm);
            remoteWinScaleTensor_.SetGlobalBuffer((__gm__ ScalesType*)(peerDataGm + xSize_));

            // 读取对端对应地址的 x 和 scale数据，进行反量化和求和
            // ReduceScatter过程，all2all仅需与rankId相关的数据，加上本卡偏移
            uint64_t curRankXOffset = curXOff + mteComm_.hcclContext_->rankId * xSliceSizeNums_;
            uint64_t curRankScaleOffset = curScaleOff + mteComm_.hcclContext_->rankId * scaleSliceNums_;
            ReadDataBlockReduceSum(curRankXOffset, curRankScaleOffset, curXCnt, curScaleCnt);
        }

        // 将计算好的数据拷贝到输出tensor
        mteComm_.CopyResultToOutput(curXOff, sumTensor_, curXCnt);
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
    mteComm_.template CopyDataToWin<true>(xSliceSizeNums_, scaleSliceNums_);
    // 写入状态到状态区
    mteComm_.WriteStatusToWin();
    // 执行ReduceScatter过程：等待状态区同步，读取数据并进行反量化ReduceSum
    ExecuteReduceScatter();
}
} // QuantReduceScatterImpl
#endif  // QUANT_REDUCE_SCATTER_MTE_H